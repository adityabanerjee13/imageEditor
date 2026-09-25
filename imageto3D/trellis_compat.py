"""Pure-PyTorch stand-ins for the two CUDA-only kernels TRELLIS imports, so DIRECT's pipeline runs
unmodified on this project's Intel XPU (and on CPU).

TRELLIS hard-imports `spconv.pytorch` (sparse 3-D convolution) in `modules/sparse/basic.py` and
`xformers.ops` (sparse attention) in `modules/sparse/attention/*`.  Neither has an XPU build, and
`modules/sparse/__init__.py` accepts only 'xformers' | 'flash_attn' for sparse attention - unlike
dense attention it has no sdpa fallback.  `install()` registers stand-in `spconv` and `xformers`
modules in `sys.modules` before trellis is imported, so no trellis or DIRECT source is touched.

Two things keep the surface small:

* Every `sp.SparseConv3d` the models build is `sp.SparseConv3d(in, out, 3)` or `(in, out, 1)` -
  stride 1, no padding - so trellis' wrapper only ever reaches spconv's `SubMConv3d`.  Down- and
  upsampling is `SparseDownsample`/`SparseUpsample` in `modules/sparse/spatial.py`, which is already
  pure PyTorch.  So only *submanifold* convolution (output coords == input coords) is needed.
* The xformers calls are just `memory_efficient_attention`, with and without a
  `BlockDiagonalMask`, which map onto `F.scaled_dot_product_attention`.

Submanifold convolution on a fully occupied grid is exactly `F.conv3d` with padding=(K-1)//2, which
is what `python -m object_edit.trellis_compat --self-test` checks the conv against.
"""
import importlib.util
import math
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["install", "SubMConv3d", "SparseConvTensor", "memory_efficient_attention"]

# Cap on the gathered neighbour buffer (bytes) before the conv splits its kernel offsets into groups.
GATHER_BUDGET = 256 << 20


# ============================================================================ spconv stand-in
class SparseConvTensor:
    """Stand-in for `spconv.pytorch.SparseConvTensor`, the container trellis' SparseTensor wraps.

    Only the attributes `modules/sparse/basic.py` touches are provided.  `features` is a property
    because trellis assigns `data._features = feats` to keep a non-flattened view (e.g. [N, 3, H, C]
    for packed qkv) alongside the flat [N, C] features the constructor is handed.
    """

    def __init__(self, features, indices, spatial_shape, batch_size,
                 grid=None, voxel_num=None, indice_dict=None):
        self._features = features
        self.indices = indices
        self.spatial_shape = [int(s) for s in spatial_shape]
        self.batch_size = int(batch_size)
        self.grid = grid
        self.voxel_num = voxel_num
        self.indice_dict = {} if indice_dict is None else indice_dict
        # Bookkeeping fields that SparseTensor.replace() copies across verbatim.
        self.benchmark = False
        self.benchmark_record = {}
        self.thrust_allocator = None
        self._timer = None
        self.force_algo = None
        self.int8_scale = None

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    def replace_feature(self, features):
        out = SparseConvTensor(features, self.indices, self.spatial_shape, self.batch_size,
                               self.grid, self.voxel_num, self.indice_dict)
        out.benchmark = self.benchmark
        out.benchmark_record = self.benchmark_record
        out.thrust_allocator = self.thrust_allocator
        out._timer = self._timer
        out.force_algo = self.force_algo
        out.int8_scale = self.int8_scale
        return out

    def dense(self, channels_first=True):
        """Scatter back to a dense [B, C, *spatial_shape] grid (spconv's default is channels-first)."""
        feats = self._features.reshape(self._features.shape[0], -1)
        out = feats.new_zeros((self.batch_size, *self.spatial_shape, feats.shape[1]))
        idx = self.indices.long()
        out[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = feats
        if channels_first:
            out = out.permute(0, 4, 1, 2, 3).contiguous()
        return out


class ConvAlgo:
    """spconv's algorithm enum; trellis only ever passes these through to the constructor."""
    Native = "Native"
    MaskImplicitGemm = "MaskImplicitGemm"
    MaskSplitImplicitGemm = "MaskSplitImplicitGemm"


def _neighbour_index(coords, offset, dims):
    """For each active voxel p, the row holding voxel p + `offset` (same batch), or -1 if inactive.

    Coordinates are packed into one int64 key per voxel and looked up with a sorted searchsorted,
    which keeps this a handful of vectorised ops rather than a Python loop over voxels.
    """
    b, xyz = coords[:, :1], coords[:, 1:]
    strides = torch.tensor([dims[1] * dims[2], dims[2], 1], device=coords.device, dtype=torch.int64)
    base = int(dims[0]) * int(dims[1]) * int(dims[2])

    keys = b.squeeze(1).long() * base + (xyz.long() * strides).sum(1)
    order = torch.argsort(keys)
    sorted_keys = keys[order]

    nbr = xyz.long() + torch.as_tensor(offset, device=coords.device, dtype=torch.int64)
    inside = ((nbr >= 0) & (nbr < torch.as_tensor(dims, device=coords.device, dtype=torch.int64))).all(1)
    nbr_keys = b.squeeze(1).long() * base + (nbr.clamp_min(0) * strides).sum(1)

    pos = torch.searchsorted(sorted_keys, nbr_keys).clamp_max(sorted_keys.numel() - 1)
    found = inside & (sorted_keys[pos] == nbr_keys)
    return torch.where(found, order[pos], torch.full_like(pos, -1))


class SubMConv3d(nn.Module):
    """Submanifold sparse 3-D convolution: the output keeps the input's coordinates, and each active
    voxel gathers only from neighbours that are themselves active.

        out[p] = sum_k W[k] @ in[p + k - (K-1)//2]

    which is PyTorch's (and spconv's) cross-correlation convention, so on a fully occupied grid this
    equals `F.conv3d(..., padding=(K-1)//2)`.  The weight keeps spconv 2.x's KRSC layout,
    [out_channels, kD, kH, kW, in_channels], which is how the released checkpoints store it.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0,
                 bias=True, indice_key=None, algo=None, **_):
        super().__init__()
        k = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
        if tuple(k) != (k[0],) * 3:
            raise NotImplementedError(f"only cubic kernels are supported, got {kernel_size}")
        d = dilation if isinstance(dilation, (list, tuple)) else (dilation,) * 3
        if tuple(d) != (1, 1, 1):
            raise NotImplementedError(f"only dilation 1 is supported, got {dilation}")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.indice_key = tuple(k), indice_key
        self.weight = nn.Parameter(torch.empty(out_channels, *self.kernel_size, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.reset_parameters()

        c = [(s - 1) // 2 for s in self.kernel_size]
        self._offsets = [(i - c[0], j - c[1], l - c[2])
                         for i in range(self.kernel_size[0])
                         for j in range(self.kernel_size[1])
                         for l in range(self.kernel_size[2])]

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight.reshape(self.out_channels, -1), a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        feats = x.features.reshape(x.features.shape[0], -1)
        coords = x.indices
        n = feats.shape[0]
        w = self.weight.to(feats.dtype)

        # 1x1x1 is a plain linear map - no neighbour search needed.
        if self.kernel_size == (1, 1, 1):
            out = feats @ w.reshape(self.out_channels, self.in_channels).t()
            if self.bias is not None:
                out = out + self.bias.to(out.dtype)
            return x.replace_feature(out)

        dims = [int(v) + 1 for v in coords[:, 1:].max(0).values.tolist()]
        # A kernel offset may point one voxel past the occupied extent; widen so it stays in range.
        dims = [v + max(s // 2 for s in self.kernel_size) + 1 for v in dims]

        # Gathering all K^3 neighbours at once is one big GEMM but needs n*K^3*Cin elements, so
        # split the offsets into groups that stay inside GATHER_BUDGET.
        per = max(1, GATHER_BUDGET // max(1, n * self.in_channels * feats.element_size()))
        w_flat = w.permute(1, 2, 3, 4, 0).reshape(-1, self.out_channels)   # [K^3*Cin, Cout]

        out = torch.zeros(n, self.out_channels, device=feats.device, dtype=feats.dtype)
        for start in range(0, len(self._offsets), per):
            group = self._offsets[start:start + per]
            gathered = feats.new_zeros(n, len(group), self.in_channels)
            for gi, off in enumerate(group):
                idx = _neighbour_index(coords, off, dims)
                hit = idx >= 0
                gathered[hit, gi] = feats[idx[hit]]
            out += gathered.reshape(n, -1) @ w_flat[start * self.in_channels:
                                                    (start + len(group)) * self.in_channels]
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return x.replace_feature(out)


class _UnsupportedConv(nn.Module):
    """Strided / inverse sparse convolution. TRELLIS' models never build these - down- and
    upsampling go through SparseDownsample/SparseUpsample - so this only guards the code path."""

    def __init__(self, *_, **__):
        raise NotImplementedError(
            f"{type(self).__name__} has no pure-PyTorch implementation in trellis_compat; "
            "TRELLIS' released models only use stride-1 submanifold convolution."
        )


class SparseConv3d(_UnsupportedConv):
    pass


class SparseInverseConv3d(_UnsupportedConv):
    pass


# ============================================================================ xformers stand-in
class BlockDiagonalMask:
    """Stand-in for `xformers.ops.fmha.BlockDiagonalMask`: attention restricted to each variable
    length segment of a concatenated batch."""

    def __init__(self, q_seqlen, kv_seqlen=None):
        self.q_seqlen = [int(s) for s in q_seqlen]
        self.kv_seqlen = self.q_seqlen if kv_seqlen is None else [int(s) for s in kv_seqlen]

    @classmethod
    def from_seqlens(cls, q_seqlen, kv_seqlen=None):
        return cls(q_seqlen, kv_seqlen)


def _sdpa(q, k, v, attn_mask=None, scale=None):
    """[B, M, H, C] -> [B, M, H, C], with SDPA's [B, H, M, C] layout in between."""
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                       attn_mask=attn_mask, scale=scale)
    return o.transpose(1, 2)


def _block_diagonal_attention(q, k, v, mask, scale=None):
    """q [Mq, H, C], k/v [Mkv, H, *] concatenated over segments -> [Mq, H, Cv]."""
    q_seqlen, kv_seqlen = mask.q_seqlen, mask.kv_seqlen
    b = len(q_seqlen)

    # Equal-length segments (the common case: fixed attention windows) reshape with no padding.
    if len(set(q_seqlen)) == 1 and len(set(kv_seqlen)) == 1:
        lq, lkv = q_seqlen[0], kv_seqlen[0]
        out = _sdpa(q.reshape(b, lq, *q.shape[1:]), k.reshape(b, lkv, *k.shape[1:]),
                    v.reshape(b, lkv, *v.shape[1:]), scale=scale)
        return out.reshape(-1, *out.shape[2:])

    lq, lkv = max(q_seqlen), max(kv_seqlen)
    dev = q.device
    q_valid = torch.arange(lq, device=dev)[None] < torch.tensor(q_seqlen, device=dev)[:, None]
    kv_valid = torch.arange(lkv, device=dev)[None] < torch.tensor(kv_seqlen, device=dev)[:, None]

    qp = q.new_zeros(b, lq, *q.shape[1:]); qp[q_valid] = q
    kp = k.new_zeros(b, lkv, *k.shape[1:]); kp[kv_valid] = k
    vp = v.new_zeros(b, lkv, *v.shape[1:]); vp[kv_valid] = v

    # [B, 1, 1, Lkv] keeps padded keys out; padded query rows are dropped by the q_valid gather.
    out = _sdpa(qp, kp, vp, attn_mask=kv_valid[:, None, None, :], scale=scale)
    return out[q_valid]


def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, scale=None, **_):
    """Stand-in for `xformers.ops.memory_efficient_attention` over [B, M, H, C] tensors."""
    if attn_bias is None:
        return _sdpa(query, key, value, scale=scale)
    if isinstance(attn_bias, BlockDiagonalMask):
        if query.shape[0] != 1:
            raise ValueError(f"BlockDiagonalMask expects a batch of 1, got {query.shape[0]}")
        return _block_diagonal_attention(query[0], key[0], value[0], attn_bias, scale).unsqueeze(0)
    return _sdpa(query, key, value, attn_mask=attn_bias, scale=scale)


# ============================================================================ cuda -> xpu redirect
# Tensor factories that trellis' representations call with a hardcoded device.
_FACTORIES = ("tensor", "zeros", "ones", "empty", "full", "arange", "linspace", "eye",
              "rand", "randn", "randint", "zeros_like", "ones_like", "empty_like", "full_like")
_redirected = False


def _is_cuda_spec(spec):
    if isinstance(spec, str):
        return spec == "cuda" or spec.startswith("cuda:")
    return isinstance(spec, torch.device) and spec.type == "cuda"


def redirect_cuda(device):
    """Point trellis' hardcoded `cuda` at `device`.

    `representations/gaussian/*` and `representations/mesh/cube2mesh.py` build their constant
    buffers with `device="cuda"` / `.cuda()` defaults that no argument threads through (e.g.
    `SLatMeshDecoder` constructs `SparseFeatures2Mesh()` without a device).  Rather than editing the
    vendored source, this remaps cuda device specs to `device` on the tensor factories and on
    `.to()` / `.cuda()`.  A no-op when the target really is CUDA.
    """
    global _redirected
    dev = torch.device(device)
    if dev.type == "cuda" or _redirected:
        return
    _redirected = True

    def remap(spec):
        return dev if _is_cuda_spec(spec) else spec

    for name in _FACTORIES:
        orig = getattr(torch, name, None)
        if orig is None:
            continue

        def wrapper(*args, _orig=orig, **kwargs):
            if "device" in kwargs:
                kwargs["device"] = remap(kwargs["device"])
            return _orig(*args, **kwargs)

        setattr(torch, name, wrapper)

    def make_to(orig):
        def to(self, *args, **kwargs):
            args = tuple(remap(a) if _is_cuda_spec(a) else a for a in args)
            if "device" in kwargs:
                kwargs["device"] = remap(kwargs["device"])
            return orig(self, *args, **kwargs)
        return to

    torch.Tensor.to = make_to(torch.Tensor.to)
    nn.Module.to = make_to(nn.Module.to)
    torch.Tensor.cuda = lambda self, *a, **k: self.to(dev)
    nn.Module.cuda = lambda self, *a, **k: self.to(dev)


# ============================================================================ kaolin stand-in
def check_tensor(tensor, shape=None, dtype=None, device=None, throw=True):
    """Faithful stand-in for `kaolin.utils.testing.check_tensor`.

    FlexiCubes (`representations/mesh/flexicubes/flexicubes.py`) imports this one helper from kaolin
    - a CUDA-only NVIDIA library - and uses it purely to assert the shapes of its own inputs, so the
    check is reimplemented here rather than stubbed out, and keeps validating.  A `None` entry in
    `shape` matches any size.
    """
    def fail(msg):
        if throw:
            raise ValueError(msg)
        return False

    if not torch.is_tensor(tensor):
        return fail(f"expected a tensor, got {type(tensor)}")
    if shape is not None:
        if len(tensor.shape) != len(shape):
            return fail(f"expected {len(shape)} dimensions, got {len(tensor.shape)}")
        for i, (got, want) in enumerate(zip(tensor.shape, shape)):
            if want is not None and got != want:
                return fail(f"expected size {want} at dim {i}, got {got}")
    if dtype is not None and tensor.dtype != dtype:
        return fail(f"expected dtype {dtype}, got {tensor.dtype}")
    if device is not None and torch.device(device).type != tensor.device.type:
        return fail(f"expected device {device}, got {tensor.device}")
    return True


# ============================================================================ installation
def _module(name, package=True):
    """A fresh stand-in module that behaves like an imported one.  `types.ModuleType` leaves
    `__spec__` as None, and `importlib.util.find_spec(name)` *raises* ValueError for a module in
    `sys.modules` with no spec - which is how libraries probe for optional packages (diffusers'
    `_is_package_available("xformers")` did, breaking every later diffusers import in a process that
    had run a 3-D job).  With a spec present the probe instead fails on the missing distribution
    metadata and correctly reports the package as unavailable."""
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    if package:
        mod.__path__ = []
        mod.__spec__.submodule_search_locations = mod.__path__
    return mod


def install():
    """Register the stand-in `spconv` and `xformers` modules. Call before importing trellis."""
    if "spconv" not in sys.modules:
        spconv = _module("spconv")
        pytorch = _module("spconv.pytorch")
        for name, obj in (("SparseConvTensor", SparseConvTensor), ("SubMConv3d", SubMConv3d),
                          ("SparseConv3d", SparseConv3d), ("SparseInverseConv3d", SparseInverseConv3d),
                          ("ConvAlgo", ConvAlgo)):
            setattr(pytorch, name, obj)
        spconv.pytorch = pytorch
        spconv.__version__ = "trellis_compat"
        sys.modules["spconv"] = spconv
        sys.modules["spconv.pytorch"] = pytorch

    if "kaolin" not in sys.modules:
        kaolin = _module("kaolin")
        utils = _module("kaolin.utils")
        testing = _module("kaolin.utils.testing")
        testing.check_tensor = check_tensor
        utils.testing = testing
        kaolin.utils = utils
        kaolin.__version__ = "trellis_compat"
        sys.modules["kaolin"] = kaolin
        sys.modules["kaolin.utils"] = utils
        sys.modules["kaolin.utils.testing"] = testing

    if "xformers" not in sys.modules:
        xformers = _module("xformers")
        ops = _module("xformers.ops")
        fmha = _module("xformers.ops.fmha")
        fmha.BlockDiagonalMask = BlockDiagonalMask
        ops.fmha = fmha
        ops.memory_efficient_attention = memory_efficient_attention
        ops.BlockDiagonalMask = BlockDiagonalMask
        xformers.ops = ops
        xformers.__version__ = "trellis_compat"
        sys.modules["xformers"] = xformers
        sys.modules["xformers.ops"] = ops
        sys.modules["xformers.ops.fmha"] = fmha


# ============================================================================ self-test
def _self_test(device="cpu"):
    """Check the stand-ins against references: submanifold conv on a fully occupied grid must equal
    F.conv3d with padding, and block-diagonal attention must equal per-segment SDPA."""
    torch.manual_seed(0)
    dev = torch.device(device)
    ok = True

    for k in (1, 3):
        cin, cout, r = 5, 7, 6
        conv = SubMConv3d(cin, cout, k, bias=True).to(dev).double()
        nn.init.normal_(conv.weight); nn.init.normal_(conv.bias)

        # Every voxel of an r^3 grid active -> submanifold conv == dense conv with padding.
        g = torch.stack(torch.meshgrid(*[torch.arange(r, device=dev)] * 3, indexing="ij"), -1).reshape(-1, 3)
        coords = torch.cat([torch.zeros(len(g), 1, device=dev, dtype=torch.int32), g.int()], 1)
        feats = torch.randn(len(g), cin, device=dev, dtype=torch.float64)

        got = conv(SparseConvTensor(feats, coords, [r, r, r], 1)).features
        dense = feats.reshape(r, r, r, cin).permute(3, 0, 1, 2)[None]
        ref = F.conv3d(dense, conv.weight.permute(0, 4, 1, 2, 3), conv.bias, padding=(k - 1) // 2)
        ref = ref[0].permute(1, 2, 3, 0).reshape(-1, cout)
        err = (got - ref).abs().max().item()
        ok &= err < 1e-9
        print(f"  SubMConv3d k={k} vs F.conv3d: max |diff| = {err:.2e}")

    # A sparse subset must match the dense conv computed with inactive voxels zeroed.
    cin, cout, r = 4, 6, 8
    conv = SubMConv3d(cin, cout, 3, bias=False).to(dev).double()
    nn.init.normal_(conv.weight)
    g = torch.stack(torch.meshgrid(*[torch.arange(r, device=dev)] * 3, indexing="ij"), -1).reshape(-1, 3)
    keep = torch.rand(len(g), device=dev) < 0.3
    coords = torch.cat([torch.zeros(int(keep.sum()), 1, device=dev, dtype=torch.int32), g[keep].int()], 1)
    feats = torch.randn(int(keep.sum()), cin, device=dev, dtype=torch.float64)
    got = conv(SparseConvTensor(feats, coords, [r, r, r], 1)).features
    dense = torch.zeros(r, r, r, cin, device=dev, dtype=torch.float64)
    dense[coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()] = feats
    ref = F.conv3d(dense.permute(3, 0, 1, 2)[None], conv.weight.permute(0, 4, 1, 2, 3), padding=1)
    ref = ref[0].permute(1, 2, 3, 0)[coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()]
    err = (got - ref).abs().max().item()
    ok &= err < 1e-9
    print(f"  SubMConv3d sparse (30% occupancy) vs masked F.conv3d: max |diff| = {err:.2e}")

    # Attention: block-diagonal (ragged and equal-length) vs per-segment SDPA.
    for seqlens in ([5, 9, 2], [7, 7, 7]):
        h, c = 3, 16
        m = sum(seqlens)
        q, k_, v = (torch.randn(1, m, h, c, device=dev, dtype=torch.float64) for _ in range(3))
        got = memory_efficient_attention(q, k_, v, BlockDiagonalMask.from_seqlens(seqlens))[0]
        parts, s = [], 0
        for n in seqlens:
            parts.append(_sdpa(q[:, s:s + n], k_[:, s:s + n], v[:, s:s + n])[0]); s += n
        err = (got - torch.cat(parts)).abs().max().item()
        ok &= err < 1e-10
        print(f"  BlockDiagonalMask {seqlens} vs per-segment SDPA: max |diff| = {err:.2e}")

    print("self-test PASSED" if ok else "self-test FAILED")
    return ok


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="TRELLIS CUDA-free compatibility shims.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if _self_test(a.device) else 1)
    ap.print_help()

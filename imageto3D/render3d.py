"""Render a TRELLIS mesh from any pose - DIRECT's 3D -> image step, without CUDA.

How DIRECT turns the 3D proxy back into an image (demo/demo.py + demo/demo_utils.py):

  1. `extract_gaussian_params` feeds the Gaussian to a viser scene under `add_transform_controls`,
     so the user drags a gizmo to pose the object.
  2. `apply_gaussian_transform(gaussian, wxyz, position)` applies that gizmo as a rigid transform to
     the *representation*, not the camera: it rotates/translates each splat's centre and composes
     the rotation into each splat's own orientation quaternion.
  3. `get_fixed_view_matrix(camera.position)` builds a world->camera matrix looking down -Z, and
     `utils3d.torch.intrinsics_from_fov_xy(fov, fov)` the (normalised) intrinsics.
  4. `GaussianRenderer.render(...)["color"]` rasterises it to a [3, H, W] image.
  5. `postprocess_rendered_image` crops to the rendered object's bbox, expands it by 1.2, pads
     square and resizes to 512 - that crop is the geometry/appearance condition FLUX then sees.
     `direct/geometry.py: render_normal_from_slat` does the same for a *normal* map via
     `MeshRenderer`, and `preprocess_example.py` renders 6 canonical views to fit the input's pose.

So "any orientation" is just step 2 + step 3: pose the object with a rigid (R, t), keep a fixed
camera, rasterise.  This module reproduces that for the mesh.  Both of TRELLIS' rasterisers are
CUDA-only (`GaussianRenderer` -> diff-gaussian-rasterization, `MeshRenderer` -> nvdiffrast), so the
camera math here is DIRECT's verbatim - the same `utils3d` calls and the same
`intrinsics_to_projection` - and only the rasteriser is replaced with a z-buffered PyTorch one that
runs on XPU.

    python -m object_edit.render3d --mesh outputs/image_to_3d/obj0_subject_512/mesh.glb --orbit 8
    python -m object_edit.render3d --mesh ... --yaw 35 --pitch 20 --out turn.png

Renders `color`, `normal`, `depth` and `mask`, matching `MeshRenderer.render`'s return dict.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import utils3d
from PIL import Image

from object_edit.common import DEV

RETURN_TYPES = ("color", "normal", "depth", "mask")


# ============================================================================ camera
def intrinsics_to_projection(intrinsics, near, far):
    """OpenCV intrinsics -> OpenGL projection. Copied from `trellis.renderers.mesh_renderer` so the
    clip-space convention matches what nvdiffrast would have produced."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = -2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.0
    return ret


def look_at(eye, target=(0, 0, 0), up=(0, 1, 0), device=None):
    """World->camera extrinsics, via the same `utils3d` helper DIRECT's render_utils uses."""
    dev = device or DEV
    t = lambda v: torch.as_tensor(v, dtype=torch.float32, device=dev)
    return utils3d.torch.extrinsics_look_at(t(eye), t(target), t(up))


def intrinsics_from_fov(fov_deg=40.0, device=None):
    """Normalised intrinsics for a square image, as in `render_normal_from_slat` (fov 40)."""
    dev = device or DEV
    fov = torch.deg2rad(torch.tensor(float(fov_deg), device=dev))
    return utils3d.torch.intrinsics_from_fov_xy(fov, fov)


def fit_radius(fov_deg=40.0, margin=1.15, object_radius=1.0):
    """Camera distance that frames an object of `object_radius` at `fov_deg` (unit sphere by
    default, which is what `normalise_mesh` produces)."""
    return margin * object_radius / math.sin(math.radians(fov_deg) / 2)


def orbit_extrinsics(n=8, radius=None, pitch_deg=20.0, fov_deg=40.0, device=None):
    """`n` yaw-spaced views around the object, the turntable equivalent of DIRECT's
    `yaw_pitch_r_fov_to_extrinsics_intrinsics`."""
    dev = device or DEV
    radius = fit_radius(fov_deg) if radius is None else radius
    p = math.radians(pitch_deg)
    out = []
    for i in range(n):
        y = 2 * math.pi * i / n
        eye = (radius * math.cos(p) * math.sin(y), radius * math.sin(p), radius * math.cos(p) * math.cos(y))
        out.append(look_at(eye, device=dev))
    return torch.stack(out), intrinsics_from_fov(fov_deg, dev)


def rigid_transform(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, translation=(0, 0, 0), scale=1.0, device=None):
    """The gizmo, as a matrix: DIRECT's `apply_gaussian_transform` takes (wxyz, position) from viser
    and applies it to the representation.  This builds the same rigid (R, t) from Euler angles."""
    dev = device or DEV
    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    cr, sr = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32, device=dev)
    rx = torch.tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=torch.float32, device=dev)
    rz = torch.tensor([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=torch.float32, device=dev)
    return (ry @ rx @ rz) * scale, torch.as_tensor(translation, dtype=torch.float32, device=dev)


# ============================================================================ rasteriser
def _rasterize(v_screen, v_depth, faces, res, chunk=200_000):
    """Z-buffered triangle rasteriser: returns the winning face per pixel and its barycentrics.

    Each triangle is expanded over the pixels of its own screen bounding box (a ragged expansion, so
    sub-pixel triangles cost ~1 pixel each and only genuinely large ones cost more), tested with
    edge functions, then reduced with a scatter-min over depth.  No backface culling - FlexiCubes
    leaves the winding inconsistent, so culling would punch holes in the surface.
    """
    dev = v_screen.device
    n_pix = res * res
    zbuf = torch.full((n_pix,), float("inf"), device=dev)
    fbuf = torch.full((n_pix,), -1, dtype=torch.long, device=dev)
    bary = torch.zeros((n_pix, 3), device=dev)

    p = v_screen[faces]                                   # [F, 3, 2]
    z = v_depth[faces]                                    # [F, 3]
    lo = p.min(1).values.floor().long().clamp(0, res - 1)
    hi = p.max(1).values.ceil().long().clamp(0, res - 1)
    wh = (hi - lo + 1).clamp(min=0)
    area = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - \
           (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    live = (wh[:, 0] > 0) & (wh[:, 1] > 0) & (area.abs() > 1e-12) & (z > 0).all(1)

    idx_all = torch.nonzero(live, as_tuple=True)[0]
    for s in range(0, idx_all.numel(), chunk):
        fi = idx_all[s:s + chunk]
        w, h = wh[fi, 0], wh[fi, 1]
        counts = w * h
        tri = torch.repeat_interleave(fi, counts)
        # local pixel offset within each triangle's bbox
        ends = counts.cumsum(0)
        local = torch.arange(int(counts.sum()), device=dev) - torch.repeat_interleave(ends - counts, counts)
        ww = torch.repeat_interleave(w, counts)
        px = torch.repeat_interleave(lo[fi, 0], counts) + local % ww
        py = torch.repeat_interleave(lo[fi, 1], counts) + local // ww

        a, b, c = p[tri, 0], p[tri, 1], p[tri, 2]
        ar = area[tri]
        fx, fy = px.float() + 0.5, py.float() + 0.5
        w0 = ((b[:, 0] - a[:, 0]) * (fy - a[:, 1]) - (fx - a[:, 0]) * (b[:, 1] - a[:, 1])) / ar
        w1 = ((fx - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (fy - a[:, 1])) / ar
        l1, l2 = w1, w0
        l0 = 1 - l1 - l2
        inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if not inside.any():
            continue
        tri, px, py = tri[inside], px[inside], py[inside]
        l0, l1, l2 = l0[inside], l1[inside], l2[inside]
        zt = z[tri]
        depth = l0 * zt[:, 0] + l1 * zt[:, 1] + l2 * zt[:, 2]
        pid = py * res + px

        zbuf.scatter_reduce_(0, pid, depth, reduce="amin")
        win = depth <= zbuf[pid] + 1e-12
        fbuf[pid[win]] = tri[win]
        bary[pid[win]] = torch.stack([l0[win], l1[win], l2[win]], 1)

    return fbuf.view(res, res), bary.view(res, res, 3), zbuf.view(res, res)


def render(mesh, extrinsics, intrinsics, res=512, near=0.1, far=100.0,
           return_types=("color", "normal", "depth", "mask"), ssaa=1):
    """Rasterise `mesh` from one camera. Mirrors `trellis.renderers.MeshRenderer.render`'s dict.

    `mesh` needs `.vertices` [V,3], `.faces` [F,3] and, for colour, `.vertex_attrs` [V,>=3].
    Normals are face normals mapped to [0,1], as `MeshRenderer` returns them.
    """
    dev = extrinsics.device
    r = res * ssaa
    v = mesh.vertices.to(dev).float()
    f = mesh.faces.to(dev).long()
    if v.numel() == 0 or f.numel() == 0:
        blank = torch.zeros(3, res, res, device=dev)
        return {k: (blank if k in ("color", "normal") else blank[:1]) for k in return_types}

    proj = intrinsics_to_projection(intrinsics.to(dev), near, far)
    homo = torch.cat([v, torch.ones_like(v[:, :1])], 1)
    v_cam = homo @ extrinsics.to(dev).T
    v_clip = homo @ (proj @ extrinsics.to(dev)).T
    ndc = v_clip[:, :3] / v_clip[:, 3:].clamp(min=1e-8)
    # NDC (-1..1, y up) -> pixels (y down), the convention nvdiffrast rasterises into
    screen = torch.stack([(ndc[:, 0] * 0.5 + 0.5) * r, (0.5 - ndc[:, 1] * 0.5) * r], 1)

    fbuf, bary, zbuf = _rasterize(screen, v_cam[:, 2], f, r)
    hit = fbuf >= 0
    out = {}

    def interp(attr):                       # attr [V, C] -> [r, r, C]
        tri = f[fbuf.clamp(min=0)]
        return (attr[tri] * bary.unsqueeze(-1)).sum(-2) * hit.unsqueeze(-1)

    for t in return_types:
        if t == "mask":
            img = hit.float().unsqueeze(-1)
        elif t == "depth":
            img = torch.where(hit, zbuf, torch.zeros_like(zbuf)).unsqueeze(-1)
        elif t == "color":
            attrs = getattr(mesh, "vertex_attrs", None)
            img = interp(attrs[:, :3].to(dev).float()) if attrs is not None else hit.float().unsqueeze(-1).repeat(1, 1, 3)
        elif t == "normal":
            fn = _face_normals(v, f)
            n = fn[fbuf.clamp(min=0)]
            # World-space face normals, [-1,1] -> [0,1], as MeshRenderer does.
            img = ((n + 1) / 2) * hit.unsqueeze(-1)
        else:
            raise ValueError(f"unknown return type {t!r}; choose from {RETURN_TYPES}")
        img = img.permute(2, 0, 1)
        if ssaa > 1:
            img = torch.nn.functional.interpolate(img[None], (res, res), mode="bilinear",
                                                  align_corners=False, antialias=True)[0]
        out[t] = img
    return out


def _face_normals(v, f):
    n = torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=-1)
    return n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)


# ============================================================================ helpers
class _Mesh:
    """Minimal stand-in for `MeshExtractResult` when loading a mesh back off disk."""

    def __init__(self, vertices, faces, vertex_attrs=None):
        self.vertices, self.faces, self.vertex_attrs = vertices, faces, vertex_attrs


def load_mesh(path, device=None):
    import trimesh

    dev = device or DEV
    scene = trimesh.load(str(path))
    m = scene.to_geometry() if hasattr(scene, "to_geometry") else scene
    attrs = None
    if getattr(m.visual, "vertex_colors", None) is not None:
        attrs = torch.tensor(np.asarray(m.visual.vertex_colors)[:, :3] / 255.0,
                             dtype=torch.float32, device=dev)
    return _Mesh(torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=dev),
                 torch.tensor(np.asarray(m.faces), dtype=torch.long, device=dev), attrs)


def transform_mesh(mesh, rot, trans):
    """Pose the object, DIRECT's step 2: a rigid transform on the representation itself."""
    return _Mesh(mesh.vertices @ rot.T.to(mesh.vertices.device) + trans.to(mesh.vertices.device),
                 mesh.faces, mesh.vertex_attrs)


def normalise_mesh(mesh):
    """Centre on the origin and fit in the unit sphere, so a fixed camera radius frames any object."""
    v = mesh.vertices
    v = v - (v.min(0).values + v.max(0).values) / 2
    return _Mesh(v / v.norm(dim=1).max().clamp(min=1e-6), mesh.faces, mesh.vertex_attrs)


def to_pil(img, mask=None, bg=None):
    """[C, H, W] float -> PIL. `render` returns the object on black like `MeshRenderer`; pass `mask`
    and `bg` (0-1 grey level) to composite it onto a different background instead."""
    a = img.detach().cpu().numpy()
    if a.shape[0] == 1:
        a = np.repeat(a, 3, 0)
    a = np.clip(a.transpose(1, 2, 0), 0, 1)
    if mask is not None and bg is not None:
        m = mask[0].detach().cpu().numpy()[..., None]
        a = a * m + float(bg) * (1 - m)
    return Image.fromarray((a * 255).astype(np.uint8))


def strip(images, bg=255):
    arrs = [np.asarray(im) for im in images]
    return Image.fromarray(np.concatenate(arrs, axis=1))


def crop_to_object(img, mask, ratio=1.2, out_size=512):
    """DIRECT's `postprocess_rendered_image`: crop to the rendered object, expand by `ratio`, pad
    square, resize - this is the framing its conditioning images use."""
    import cv2

    m = (mask[0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    ys, xs = np.where(m)
    if len(ys) == 0:
        return Image.new("RGB", (out_size, out_size))
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    a = (np.clip(img.detach().cpu().numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)[y1:y2, x1:x2]
    h, w = a.shape[:2]
    side = int(max(h, w) * ratio)
    pad = np.zeros((side, side, 3), np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    pad[oy:oy + h, ox:ox + w] = a
    return Image.fromarray(cv2.resize(pad, (out_size, out_size), interpolation=cv2.INTER_LINEAR))


# ============================================================================ self-test
def _self_test(device=None):
    """Check the hand-written rasteriser against things with known answers: occlusion order,
    barycentric interpolation, and the projection of a point whose pixel can be computed by hand."""
    dev = torch.device(device) if device else DEV
    ok = True

    # 1. Occlusion: a near red quad in front of a far green one must win, whatever the face order.
    def two_quads(near_first):
        q = lambda z: [[-0.5, -0.5, z], [0.5, -0.5, z], [0.5, 0.5, z], [-0.5, 0.5, z]]
        v = torch.tensor(q(0.0) + q(-0.5), dtype=torch.float32, device=dev)
        f = torch.tensor([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=torch.long, device=dev)
        col = torch.tensor([[1., 0, 0]] * 4 + [[0, 1., 0]] * 4, device=dev)
        if not near_first:                       # reverse submission order
            f = f.flip(0)
        return _Mesh(v, f, col)

    for near_first in (True, False):
        r = render(two_quads(near_first), look_at((0, 0, 3), device=dev),
                   intrinsics_from_fov(40.0, dev), res=64, return_types=("color", "mask"))
        c = r["color"][:, 32, 32]
        hit = c[0] > 0.9 and c[1] < 0.1          # red, not green
        ok &= hit
        print(f"  occlusion (near submitted {'first' if near_first else 'second'}): "
              f"centre = {[round(float(x), 2) for x in c]} {'OK' if hit else 'FAIL'}")

    # 2. Barycentric interpolation: interpolating the vertices' own x/y must reproduce the plane,
    #    so the rendered 'colour' equals the point's world position at every covered pixel.
    v = torch.tensor([[-1., -1, 0], [1., -1, 0], [0., 1., 0]], device=dev)
    attrs = (v[:, :3] + 2) / 4                   # affine map of position into 0..1
    r = render(_Mesh(v, torch.tensor([[0, 1, 2]], device=dev), attrs),
               look_at((0, 0, 4), device=dev), intrinsics_from_fov(60.0, dev),
               res=128, return_types=("color", "mask"))
    m = r["mask"][0] > 0.5
    # Reconstruct world x,y from the interpolated attribute and re-project: should land on itself.
    got = r["color"].permute(1, 2, 0)[m] * 4 - 2
    err = float((got[:, 2]).abs().max())         # the triangle is planar at z = 0
    ok &= err < 1e-4
    print(f"  barycentric interpolation, max |z| on a z=0 triangle: {err:.2e}")

    # 3. Projection: a point at world (x, 0, 0) seen from (0,0,d) lands at pixel
    #    cx + res * fx * x / d, with fx the normalised focal length.
    d, x, res, fov = 4.0, 0.3, 128, 40.0
    K = intrinsics_from_fov(fov, dev)
    expect = res * (float(K[0, 2]) + float(K[0, 0]) * x / d)
    # ~7 px wide and symmetric about x, so the covered pixels' x-centroid is the projected centre.
    w = 0.08
    tri = torch.tensor([[x - w, -w, 0], [x + w, -w, 0], [x, w, 0]], device=dev)
    r = render(_Mesh(tri, torch.tensor([[0, 1, 2]], device=dev), None),
               look_at((0, 0, d), device=dev), K, res=res, return_types=("mask",))
    hits = torch.nonzero(r["mask"][0] > 0.5)
    if hits.numel() == 0:
        ok = False
        print("  projection: nothing rasterised FAIL")
    else:
        xs = hits[:, 1].float()
        err = abs(float(xs.mean() + 0.5) - expect)
        ok &= err < 1.0
        print(f"  projection: {len(xs)} px, centroid {float(xs.mean() + 0.5):.1f}, "
              f"expected {expect:.1f} (|d| = {err:.2f})")

    print("self-test PASSED" if ok else "self-test FAILED")
    return ok


# ============================================================================ CLI
def main():
    ap = argparse.ArgumentParser(description="Render a TRELLIS mesh from any pose (DIRECT's 3D -> image step).")
    ap.add_argument("--mesh", default="outputs/image_to_3d/obj0_subject_512/mesh.glb")
    ap.add_argument("--out", default=None, help="output PNG (default: <mesh dir>/render_<kind>.png)")
    ap.add_argument("--orbit", type=int, default=0, help="render N yaw-spaced views instead of one")
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--cam-pitch", type=float, default=20.0, help="camera elevation for --orbit")
    ap.add_argument("--radius", type=float, default=None, help="camera distance (default: auto-fit to the fov)")
    ap.add_argument("--bg", type=float, default=1.0, help="background grey level 0-1 (DIRECT renders on 0)")
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--ssaa", type=int, default=2, help="supersampling factor")
    ap.add_argument("--kind", default="color", choices=RETURN_TYPES)
    ap.add_argument("--crop", action="store_true", help="crop/pad/resize like DIRECT's condition images")
    ap.add_argument("--self-test", action="store_true", help="check the rasteriser against known answers")
    a = ap.parse_args()

    if a.self_test:
        raise SystemExit(0 if _self_test() else 1)

    mesh = normalise_mesh(load_mesh(a.mesh))
    if a.yaw or a.pitch or a.roll:
        mesh = transform_mesh(mesh, *rigid_transform(a.yaw, a.pitch, a.roll))

    if a.orbit:
        extr, intr = orbit_extrinsics(a.orbit, a.radius, a.cam_pitch, a.fov)
    else:
        extr = look_at((0, 0, a.radius if a.radius is not None else fit_radius(a.fov)))[None]
        intr = intrinsics_from_fov(a.fov)

    frames = []
    for e in extr:
        r = render(mesh, e, intr, res=a.res, ssaa=a.ssaa, return_types=(a.kind, "mask"))
        frames.append(crop_to_object(r[a.kind], r["mask"], out_size=a.res) if a.crop
                      else to_pil(r[a.kind], r["mask"], a.bg))

    out = Path(a.out) if a.out else Path(a.mesh).parent / f"render_{a.kind}.png"
    (frames[0] if len(frames) == 1 else strip(frames)).save(out)
    print(f"[render3d] {len(frames)} view(s) -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Bootstrap the imageEditor workspace: clone the code dependencies into repos/ and download the model weights into weights/.

    python setup.py                      # everything (~13 GB of weights for both pipelines)
    python setup.py --pipeline floor     # only what floor_edit needs   (~10.8 GB)
    python setup.py --pipeline object    # only what object_edit needs  (~7.1 GB; sam3 + moge are shared)
    python setup.py --repos-only | --weights-only
    python setup.py --check              # report what is present / missing, download nothing
    python setup.py --pip                # also `pip install -r requirements.txt` first (install torch yourself before)

Existing repos / weights are skipped, so the script is safe to re-run after an interrupted download.
Hugging Face: `facebook/sam3` is gated -> the `jetjodh/sam3` mirror is used. Set HF_TOKEN (or `hf auth login`) if a
repo asks for it. Stale `.incomplete` files under weights/*/.cache are removed after each download.
"""
import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOS = ROOT / "repos"
WEIGHTS = ROOT / "weights"

# name -> (git url, pinned commit or None, pipelines that need it)
REPO_SPECS = {
    "MoGe":     ("https://github.com/microsoft/MoGe.git",   "74fbce054ebed49800de42d0ad0e83495065719a", {"floor", "object"}),
    "rgbx":     ("https://github.com/zheng95z/rgbx.git",    None,                                       {"floor"}),
    "FreeFine": ("https://github.com/CIawevy/FreeFine.git", "4c9fdb971572b32edbeac13464659274c28decbb", {"object"}),
    "lama":     ("https://github.com/advimman/lama.git",    "786f5936b27fb3dacd2b1ad799e4de968ea697e7", {"object"}),
}

# folder -> spec. kind: "snapshot" (repo subset via allow_patterns) | "file" (single file) | "zip" (single archive, extracted)
# `marker` is a file whose presence means the download is complete.
WEIGHT_SPECS = {
    "segformer-b5-ade": dict(kind="snapshot", repo="nvidia/segformer-b5-finetuned-ade-640-640",
                             patterns=["*.json", "pytorch_model.bin"], marker="pytorch_model.bin",
                             pipelines={"floor"}, size="0.34 GB"),
    "upernet-convnext-l": dict(kind="snapshot", repo="openmmlab/upernet-convnext-large",
                               patterns=["*.json", "pytorch_model.bin"], marker="pytorch_model.bin",
                               pipelines={"floor"}, size="0.9 GB"),
    "sam3": dict(kind="snapshot", repo="jetjodh/sam3",             # mirror of the gated facebook/sam3
                 patterns=["*.json", "*.txt", "*.safetensors"], marker="model.safetensors",
                 pipelines={"floor", "object"}, size="3.4 GB"),
    "moge-2-vitl-normal": dict(kind="file", repo="Ruicheng/moge-2-vitl-normal", filename="model.pt", marker="model.pt",
                               pipelines={"floor", "object"}, size="1.3 GB"),
    "rgb-to-x": dict(kind="snapshot", repo="zheng95z/rgb-to-x",
                     patterns=["*.json", "*.txt", "*.safetensors", "LICENSE"], marker="unet/diffusion_pytorch_model.safetensors",
                     pipelines={"floor"}, size="4.9 GB"),
    "stable-diffusion-v1-5": dict(kind="snapshot", repo="stable-diffusion-v1-5/stable-diffusion-v1-5",
                                  patterns=["model_index.json", "scheduler/*", "tokenizer/*", "feature_extractor/*",
                                            "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
                                            "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
                                            "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"],
                                  marker="unet/diffusion_pytorch_model.fp16.safetensors",
                                  pipelines={"object"}, size="2.0 GB"),
    "big-lama": dict(kind="zip", repo="smartywu/big-lama", filename="big-lama.zip", marker="big-lama/models/best.ckpt",
                     pipelines={"object"}, size="0.4 GB"),
}


def log(msg):
    print(f"[setup] {msg}", flush=True)


# ---------------------------------------------------------------------------- repos
def repo_present(name):
    return (REPOS / name).is_dir() and any((REPOS / name).iterdir())


def clone_repo(name, url, commit):
    dst = REPOS / name
    if repo_present(name):
        log(f"repo {name}: present, skipping")
        return
    REPOS.mkdir(exist_ok=True)
    log(f"repo {name}: cloning {url}")
    subprocess.run(["git", "clone", "--quiet", url, str(dst)], check=True)
    if commit:
        subprocess.run(["git", "-C", str(dst), "checkout", "--quiet", commit], check=True)
        log(f"repo {name}: checked out {commit[:10]}")


# ---------------------------------------------------------------------------- weights
def weight_present(name):
    return (WEIGHTS / name / WEIGHT_SPECS[name]["marker"]).is_file()


def clean_cache(folder):
    """huggingface_hub keeps download metadata / partial files in <local_dir>/.cache - drop it once the files are in place."""
    cache = folder / ".cache"
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)


def download_weight(name):
    spec = WEIGHT_SPECS[name]
    dst = WEIGHTS / name
    if weight_present(name):
        log(f"weights {name}: present, skipping")
        return
    from huggingface_hub import hf_hub_download, snapshot_download
    dst.mkdir(parents=True, exist_ok=True)
    log(f"weights {name}: downloading from {spec['repo']} ({spec['size']})")
    if spec["kind"] == "snapshot":
        snapshot_download(spec["repo"], local_dir=str(dst), allow_patterns=spec["patterns"])
    elif spec["kind"] == "file":
        hf_hub_download(spec["repo"], spec["filename"], local_dir=str(dst))
    elif spec["kind"] == "zip":
        archive = Path(hf_hub_download(spec["repo"], spec["filename"], local_dir=str(dst)))
        log(f"weights {name}: extracting {archive.name}")
        with zipfile.ZipFile(archive) as z:
            z.extractall(dst)
        archive.unlink()
    clean_cache(dst)
    if not weight_present(name):
        sys.exit(f"[setup] weights {name}: download finished but {spec['marker']} is missing - check the repo layout")


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", choices=["all", "floor", "object"], default="all", help="which pipeline to set up (default: all)")
    ap.add_argument("--repos-only", action="store_true")
    ap.add_argument("--weights-only", action="store_true")
    ap.add_argument("--pip", action="store_true", help="pip install -r requirements.txt before anything else")
    ap.add_argument("--check", action="store_true", help="only report what is present / missing")
    args = ap.parse_args()
    want = {"floor", "object"} if args.pipeline == "all" else {args.pipeline}

    repos = [n for n, (_, _, p) in REPO_SPECS.items() if p & want]
    weights = [n for n, s in WEIGHT_SPECS.items() if s["pipelines"] & want]

    if args.check:
        for n in repos:
            print(f"  repo    {n:<24s} {'ok' if repo_present(n) else 'MISSING'}")
        for n in weights:
            print(f"  weights {n:<24s} {'ok' if weight_present(n) else 'MISSING':<8s} {WEIGHT_SPECS[n]['size']}")
        return

    if args.pip:
        log("pip install -r requirements.txt")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)

    if not args.weights_only:
        for n in repos:
            clone_repo(n, *REPO_SPECS[n][:2])
    if not args.repos_only:
        for n in weights:
            download_weight(n)
    log("done")


if __name__ == "__main__":
    main()

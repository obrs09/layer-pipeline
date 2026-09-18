"""Download SAM2 / LaMa / SD1.5 weights into model/ (gitignored)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAM2_DIR = ROOT / "model" / "sam2"
LAMA_DIR = ROOT / "model" / "lama"
SD15_DIR = ROOT / "model" / "sd15"

SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
SAM2_NAME = "sam2.1_hiera_large.pt"
LAMA_REPO = "okaris/simple-lama"
LAMA_FILE = "big-lama.pt"
SD15_REPO = "stable-diffusion-v1-5/stable-diffusion-inpainting"


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"skip existing {dest}")
        return
    import urllib.request

    print(f"downloading {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"saved {dest} ({dest.stat().st_size} bytes)")


def _hf_file(repo: str, filename: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(dest_dir),
    )
    return Path(path)


def _hf_snapshot(repo: str, dest_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        local_dir=str(dest_dir),
        ignore_patterns=["*.md", "*.txt", ".gitattributes"],
    )


def main() -> int:
    print("SAM2…")
    try:
        _download_url(SAM2_URL, SAM2_DIR / SAM2_NAME)
    except Exception as exc:
        print(f"direct SAM2 failed ({exc}); trying Hugging Face")
        _hf_file("facebook/sam2.1-hiera-large", SAM2_NAME, SAM2_DIR)

    print("LaMa…")
    try:
        _hf_file(LAMA_REPO, LAMA_FILE, LAMA_DIR)
    except Exception as exc:
        print(f"{LAMA_REPO} failed ({exc}); trying fashn-ai/LaMa")
        _hf_file("fashn-ai/LaMa", LAMA_FILE, LAMA_DIR)

    print("SD1.5 inpaint…")
    _hf_snapshot(SD15_REPO, SD15_DIR)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

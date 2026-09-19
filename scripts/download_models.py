"""Download detect / SAM / inpaint weights into model/ (gitignored)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAM2_DIR = ROOT / "model" / "sam2"
LAMA_DIR = ROOT / "model" / "lama"
SD15_DIR = ROOT / "model" / "sd15"
ANISEG_DIR = ROOT / "model" / "aniseg"
WD_DIR = ROOT / "model" / "wdtagger"
DINO_DIR = ROOT / "model" / "grounding_dino"
SAM3_DIR = ROOT / "model" / "sam3"

SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
SAM2_NAME = "sam2.1_hiera_large.pt"
LAMA_REPO = "okaris/simple-lama"
LAMA_FILE = "big-lama.pt"
SD15_REPO = "stable-diffusion-v1-5/stable-diffusion-inpainting"
ANISEG_REPO = "skytnt/anime-seg"
ANISEG_FILE = "isnetis.onnx"
WD_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
DINO_REPO = "IDEA-Research/grounding-dino-tiny"
SAM3_REPO = "facebook/sam3"
SAM3_FILES = ("sam3.pt", "config.json")
DWPOSE_DIR = ROOT / "model" / "dwpose"
DWPOSE_REPO = "fashn-ai/DWPose"
DWPOSE_FILES = ("yolox_l.onnx", "dw-ll_ucoco_384.onnx")


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


def _download_detect() -> None:
    print("anime-segmentation (SkyTNT isnetis)…")
    _hf_file(ANISEG_REPO, ANISEG_FILE, ANISEG_DIR)

    print("WDTagger…")
    _hf_file(WD_REPO, "model.onnx", WD_DIR)
    _hf_file(WD_REPO, "selected_tags.csv", WD_DIR)

    print("Grounding DINO…")
    _hf_snapshot(DINO_REPO, DINO_DIR)

    print("SAM3…")
    try:
        for name in SAM3_FILES:
            _hf_file(SAM3_REPO, name, SAM3_DIR)
    except Exception as exc:
        print(
            f"SAM3 download failed ({exc}). "
            "Request access at https://huggingface.co/facebook/sam3 then `hf auth login`."
        )

    print("DWPose…")
    try:
        for name in DWPOSE_FILES:
            _hf_file(DWPOSE_REPO, name, DWPOSE_DIR)
    except Exception as exc:
        print(f"{DWPOSE_REPO} failed ({exc}); trying IDEA-Research/DWPose filenames")
        try:
            _hf_file("yzd-v/DWPose", "yolox_l.onnx", DWPOSE_DIR)
            _hf_file("yzd-v/DWPose", "dw-ll_ucoco_384.onnx", DWPOSE_DIR)
        except Exception as exc2:
            print(f"DWPose download failed ({exc2}). Put yolox_l.onnx and dw-ll_ucoco_384.onnx in model/dwpose/")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only-detect",
        action="store_true",
        help="anime-segmentation / WDTagger / DINO / SAM3 / DWPose only (skip SAM2, LaMa, SD1.5)",
    )
    parser.add_argument(
        "--only-dwpose",
        action="store_true",
        help="YOLOX + DWPose ONNX only",
    )
    args = parser.parse_args()

    if args.only_dwpose:
        print("DWPose…")
        try:
            for name in DWPOSE_FILES:
                _hf_file(DWPOSE_REPO, name, DWPOSE_DIR)
        except Exception as exc:
            print(f"{DWPOSE_REPO} failed ({exc}); trying yzd-v/DWPose")
            _hf_file("yzd-v/DWPose", "yolox_l.onnx", DWPOSE_DIR)
            _hf_file("yzd-v/DWPose", "dw-ll_ucoco_384.onnx", DWPOSE_DIR)
        print("done")
        return 0

    if not args.only_detect:
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

    _download_detect()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

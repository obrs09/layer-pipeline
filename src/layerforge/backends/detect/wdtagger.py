from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from layerforge.config import resolve_path
from layerforge.taxonomy import _norm_tag


class WdTagger:
    """Danbooru WD tagger. Inventory only — does not cut pixels."""

    name = "wdtagger"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._session = None
        self._tags: list[str] = []

    def _files(self) -> tuple[Path, Path]:
        paths = self.cfg.get("paths") or {}
        search = resolve_path(paths.get("wdtagger_dir", "model/wdtagger"))
        onnx = paths.get("wdtagger_checkpoint")
        onnx_path = resolve_path(onnx) if onnx else None
        if onnx_path is None or not onnx_path.exists():
            found = sorted(search.glob("*.onnx"))
            onnx_path = found[0] if found else None
        csv_path = search / "selected_tags.csv"
        if csv_path.exists() is False:
            alt = sorted(search.glob("*.csv"))
            csv_path = alt[0] if alt else csv_path
        if onnx_path is None or not Path(onnx_path).exists() or not csv_path.exists():
            raise RuntimeError(
                "WDTagger weights missing. Put model.onnx and selected_tags.csv "
                "in model/wdtagger/ (see scripts/download_models.py)."
            )
        return Path(onnx_path), csv_path

    def tag(self, image: np.ndarray) -> dict[str, float]:
        onnx_path, csv_path = self._files()
        if self._session is None:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(onnx_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._tags = _load_tags(csv_path)
        rgb = image[:, :, :3]
        blob = _preprocess(rgb, 448)
        inp = self._session.get_inputs()[0]
        probs = np.squeeze(self._session.run(None, {inp.name: blob})[0])
        out: dict[str, float] = {}
        for name, prob in zip(self._tags, probs.tolist()):
            if prob >= 0.15:
                out[_norm_tag(name)] = float(prob)
        return out


def _load_tags(csv_path: Path) -> list[str]:
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    tags: list[str] = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        tags.append(parts[1].strip().strip('"'))
    return tags


def _preprocess(rgb: np.ndarray, size: int) -> np.ndarray:
    """WD SwinV2 v3 ONNX: white square pad, bicubic 448, BGR, float32 0-255, NHWC."""
    image = Image.fromarray(rgb[:, :, :3]).convert("RGB")
    w, h = image.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    if side != size:
        canvas = canvas.resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(canvas, dtype=np.float32)
    arr = arr[:, :, ::-1]
    return arr[None]

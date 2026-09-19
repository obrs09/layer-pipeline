from __future__ import annotations

from pathlib import Path

import numpy as np

from layerforge.config import resolve_path
from layerforge.ops.normalize import foreground_mask, to_rgba


class AniSegCut:
    """Whole-character cut. ONNX isnet (skytnt/anime-seg) if present."""

    name = "aniseg"

    def __init__(self, cfg: dict, *, allow_luma: bool = False) -> None:
        self.cfg = cfg
        self.allow_luma = allow_luma
        self._net = None
        self._input_size = 1024

    def _onnx_path(self) -> Path | None:
        paths = self.cfg.get("paths") or {}
        explicit = paths.get("aniseg_checkpoint")
        if explicit:
            path = resolve_path(explicit)
            return path if path.exists() else None
        search = resolve_path(paths.get("aniseg_dir", "model/aniseg"))
        found = sorted(search.glob("*.onnx"))
        return found[0] if found else None

    def cut(self, image: np.ndarray) -> np.ndarray:
        onnx_path = self._onnx_path()
        if onnx_path is None:
            if self.allow_luma:
                return foreground_mask(to_rgba(image))
            raise RuntimeError(
                "AniSeg ONNX missing. Put isnetis.onnx in model/aniseg/ "
                "(see scripts/download_models.py)."
            )
        rgb = to_rgba(image)[:, :, :3]
        h, w = rgb.shape[:2]
        blob = _letterbox(rgb, self._input_size)
        mask = self._run(onnx_path, blob)
        mask = _unletterbox(mask, h, w, self._input_size)
        return ((mask > 0.5).astype(np.uint8) * 255)

    def _run(self, onnx_path: Path, blob: np.ndarray) -> np.ndarray:
        try:
            import onnxruntime as ort
        except ImportError:
            import cv2

            net = cv2.dnn.readNetFromONNX(str(onnx_path))
            net.setInput(blob)
            out = net.forward()
            return np.squeeze(out).astype(np.float32)
        if self._net is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._net = ort.InferenceSession(str(onnx_path), providers=providers)
        inp = self._net.get_inputs()[0]
        out = self._net.run(None, {inp.name: blob})[0]
        return np.squeeze(out).astype(np.float32)


def _letterbox(rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    import cv2

    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized
    chw = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw[None]


def _unletterbox(mask: np.ndarray, h: int, w: int, size: int) -> np.ndarray:
    import cv2

    mask = np.squeeze(mask)
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    crop = mask[:nh, :nw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

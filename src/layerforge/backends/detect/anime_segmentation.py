from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from layerforge.config import resolve_path
from layerforge.ops.normalize import foreground_mask, to_rgba


class AnimeSegmentationCut:
    """Whole-character cut using SkyTNT anime-segmentation (isnetis ONNX).

    Preprocess matches official inference.py: RGB /255, aspect resize, center pad.
    """

    name = "anime_segmentation"

    def __init__(self, cfg: dict, *, allow_luma: bool = False) -> None:
        self.cfg = cfg
        self.allow_luma = allow_luma
        self._net = None
        self._input_size = int((cfg.get("cascade") or {}).get("character_size", 1024))

    def _onnx_path(self) -> Path | None:
        paths = self.cfg.get("paths") or {}
        for key in ("anime_segmentation_checkpoint", "aniseg_checkpoint"):
            explicit = paths.get(key)
            if explicit:
                path = resolve_path(explicit)
                if path.exists():
                    return path
        for key, default in (
            ("anime_segmentation_dir", "model/anime_segmentation"),
            ("aniseg_dir", "model/aniseg"),
        ):
            search = resolve_path(paths.get(key, default))
            if not search.exists():
                continue
            found = sorted(search.glob("*.onnx"))
            if found:
                return found[0]
        return None

    def cut(self, image: np.ndarray) -> np.ndarray:
        onnx_path = self._onnx_path()
        if onnx_path is None:
            if self.allow_luma:
                return foreground_mask(to_rgba(image))
            raise RuntimeError(
                "anime-segmentation ONNX missing. Put isnetis.onnx in "
                "model/aniseg/ or model/anime_segmentation/ "
                "(see scripts/download_models.py)."
            )
        rgb = to_rgba(image)[:, :, :3]
        blob, meta = pack_square(rgb, self._input_size)
        pred = self._run(onnx_path, blob)
        mask = unpack_square(pred, meta)
        return ((mask > 0.5).astype(np.uint8) * 255)

    def _run(self, onnx_path: Path, blob: np.ndarray) -> np.ndarray:
        try:
            import onnxruntime as ort
        except ImportError:
            net = cv2.dnn.readNetFromONNX(str(onnx_path))
            net.setInput(blob)
            out = net.forward()
            return np.squeeze(out).astype(np.float32)
        if self._net is None:
            self._net = ort.InferenceSession(
                str(onnx_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        inp = self._net.get_inputs()[0]
        out = self._net.run(None, {inp.name: blob})[0]
        return np.squeeze(out).astype(np.float32)


class AniSegCut(AnimeSegmentationCut):
    """Old name. Same SkyTNT isnetis weights and preprocess."""

    name = "aniseg"


def pack_square(rgb: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int, int, int, int, int]]:
    """Center-pad like SkyTNT/anime-segmentation inference.get_mask."""
    h0, w0 = rgb.shape[:2]
    if h0 > w0:
        nh, nw = size, max(1, int(size * w0 / h0))
    else:
        nh, nw = max(1, int(size * h0 / w0)), size
    top = (size - nh) // 2
    left = (size - nw) // 2
    resized = cv2.resize(rgb.astype(np.float32) / 255.0, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    canvas[top : top + nh, left : left + nw] = resized
    blob = np.transpose(canvas, (2, 0, 1))[None]
    return blob, (h0, w0, nh, nw, top, left)


def unpack_square(mask: np.ndarray, meta: tuple[int, int, int, int, int, int]) -> np.ndarray:
    h0, w0, nh, nw, top, left = meta
    mask = np.squeeze(mask).astype(np.float32)
    if mask.ndim == 3:
        mask = mask[0] if mask.shape[0] <= 3 else mask[:, :, 0]
    crop = mask[top : top + nh, left : left + nw]
    return cv2.resize(crop, (w0, h0), interpolation=cv2.INTER_LINEAR)

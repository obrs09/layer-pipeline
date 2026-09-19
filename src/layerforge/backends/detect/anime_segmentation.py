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
        mask, _prob = self.predict(image, seed=0)
        return mask

    def predict(self, image: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Return (uint8 mask, float32 probability). seed picks a retry variant."""
        onnx_path = self._onnx_path()
        if onnx_path is None:
            if self.allow_luma:
                mask = foreground_mask(to_rgba(image))
                return mask, (mask > 0).astype(np.float32)
            raise RuntimeError(
                "anime-segmentation ONNX missing. Put isnetis.onnx in "
                "model/aniseg/ or model/anime_segmentation/ "
                "(see scripts/download_models.py)."
            )
        size, threshold, noise = _seed_variant(seed, self._fixed_size() or self._input_size)
        rgb = to_rgba(image)[:, :, :3].copy()
        if noise > 0:
            rng = np.random.default_rng(int(seed))
            rgb = np.clip(
                rgb.astype(np.float32) + rng.normal(0.0, noise, rgb.shape),
                0,
                255,
            ).astype(np.uint8)
        blob, meta = pack_square(rgb, size)
        pred = self._run(onnx_path, blob)
        prob = np.clip(unpack_square(pred, meta), 0.0, 1.0)
        return ((prob > threshold).astype(np.uint8) * 255), prob.astype(np.float32)

    def _fixed_size(self) -> int | None:
        if self._net is None:
            return None
        try:
            dim = self._net.get_inputs()[0].shape[2]
        except Exception:
            return None
        if isinstance(dim, int) and dim > 0:
            return dim
        return None

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


def _seed_variant(seed: int, default_size: int) -> tuple[int, float, float]:
    """ONNX is deterministic; seed selects size/threshold/input noise."""
    variants = (
        (default_size, 0.50, 0.0),
        (default_size, 0.42, 1.5),
        (default_size, 0.58, 2.5),
    )
    return variants[int(seed) % len(variants)]

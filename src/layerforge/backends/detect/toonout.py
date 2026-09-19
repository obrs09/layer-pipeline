"""ToonOut (BiRefNet anime) as a CharacterCut peer for consistency QA."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from layerforge.config import resolve_path
from layerforge.ops.normalize import to_rgba

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ToonOutCut:
    name = "toonout"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._net = None
        self._size = 1024

    def cut(self, image: np.ndarray) -> np.ndarray:
        path = self._onnx_path()
        if path is None:
            raise RuntimeError(
                "ToonOut ONNX missing. Put birefnet-toonout-fp16.onnx in "
                "model/toonout/ (see scripts/download_models.py --only-character-qa)."
            )
        rgb = to_rgba(image)[:, :, :3]
        blob = _pack(rgb, self._size)
        pred = self._run(path, blob)
        mask = cv2.resize(
            _as_prob(pred),
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return ((mask > 0.5).astype(np.uint8) * 255)

    def _onnx_path(self) -> Path | None:
        paths = self.cfg.get("paths") or {}
        explicit = paths.get("toonout_checkpoint")
        if explicit:
            path = resolve_path(explicit)
            if path.exists():
                return path
        search = resolve_path(paths.get("toonout_dir", "model/toonout"))
        if not search.exists():
            return None
        for name in ("birefnet-toonout-fp16.onnx", "birefnet-toonout.onnx"):
            cand = search / name
            if cand.exists():
                return cand
        found = sorted(search.glob("*.onnx"))
        return found[0] if found else None

    def _run(self, path: Path, blob: np.ndarray) -> np.ndarray:
        import onnxruntime as ort

        if self._net is None:
            self._net = ort.InferenceSession(
                str(path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        names = [item.name for item in self._net.get_inputs()]
        key = "image" if "image" in names else names[0]
        inp = next(item for item in self._net.get_inputs() if item.name == key)
        blob = blob.astype(np.float16 if inp.type == "tensor(float16)" else np.float32)
        out = self._net.run(None, {key: blob})[-1]
        return np.squeeze(out).astype(np.float32)


def _pack(rgb: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(x, (2, 0, 1))[None]


def _as_prob(pred: np.ndarray) -> np.ndarray:
    mask = np.squeeze(pred).astype(np.float32)
    if mask.ndim == 3:
        mask = mask[0] if mask.shape[0] <= 3 else mask[:, :, 0]
    if float(mask.max(initial=0.0)) > 1.5 or float(mask.min(initial=0.0)) < 0.0:
        mask = 1.0 / (1.0 + np.exp(-np.clip(mask, -20.0, 20.0)))
    return np.clip(mask, 0.0, 1.0)

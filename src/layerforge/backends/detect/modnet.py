"""MODNet as a lightweight CharacterCut peer (Anime-MODNet architecture).

SkyTNT trains a MODNet on anime-seg but did not publish ONNX. The available
ONNX is Xenova/modnet (same architecture, photographic weights).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from layerforge.config import resolve_path
from layerforge.ops.normalize import to_rgba


class ModnetCut:
    name = "modnet"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._net = None
        self._ref = 512

    def cut(self, image: np.ndarray) -> np.ndarray:
        path = self._onnx_path()
        if path is None:
            raise RuntimeError(
                "MODNet ONNX missing. Put model.onnx in model/modnet/onnx/ "
                "or model/modnet/ (see scripts/download_models.py --only-character-qa)."
            )
        rgb = to_rgba(image)[:, :, :3]
        blob, size = _pack(rgb, self._ref)
        pred = self._run(path, blob)
        mask = cv2.resize(
            _as_prob(pred),
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return ((mask > 0.5).astype(np.uint8) * 255)

    def _onnx_path(self) -> Path | None:
        paths = self.cfg.get("paths") or {}
        explicit = paths.get("modnet_checkpoint")
        if explicit:
            path = resolve_path(explicit)
            if path.exists():
                return path
        search = resolve_path(paths.get("modnet_dir", "model/modnet"))
        if not search.exists():
            return None
        for rel in ("onnx/model.onnx", "model.onnx", "modnet.onnx"):
            cand = search / rel
            if cand.exists():
                return cand
        found = sorted(search.rglob("*.onnx"))
        preferred = [p for p in found if p.name == "model.onnx"]
        return (preferred or found)[0] if found else None

    def _run(self, path: Path, blob: np.ndarray) -> np.ndarray:
        import onnxruntime as ort

        if self._net is None:
            self._net = ort.InferenceSession(
                str(path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        inp = self._net.get_inputs()[0]
        out = self._net.run(None, {inp.name: blob})[-1]
        return np.squeeze(out).astype(np.float32)


def _pack(rgb: np.ndarray, ref: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = rgb.shape[:2]
    scale = ref / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    nh = max(32, (nh + 31) // 32 * 32)
    nw = max(32, (nw + 31) // 32 * 32)
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return np.transpose(x, (2, 0, 1))[None], (nh, nw)


def _as_prob(pred: np.ndarray) -> np.ndarray:
    mask = np.squeeze(pred).astype(np.float32)
    if mask.ndim == 3:
        mask = mask[0] if mask.shape[0] <= 3 else mask[:, :, 0]
    if float(mask.max(initial=0.0)) > 1.5 or float(mask.min(initial=0.0)) < 0.0:
        mask = 1.0 / (1.0 + np.exp(-np.clip(mask, -20.0, 20.0)))
    return np.clip(mask, 0.0, 1.0)

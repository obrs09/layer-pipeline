from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from layerforge.config import resolve_path


class LamaInpaint:
    name = "lama"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._model = None

    def _checkpoint(self) -> Path:
        paths = self.cfg.get("paths") or {}
        explicit = paths.get("lama_checkpoint")
        if explicit:
            path = resolve_path(explicit)
            if path.exists():
                return path
        search = resolve_path(paths.get("lama_dir", "model/lama"))
        found = (
            sorted(search.glob("*.pt"))
            + sorted(search.glob("*.pth"))
            + sorted(search.glob("*.ckpt"))
        )
        if not found:
            raise RuntimeError(f"No LaMa checkpoint under {search}. Put weights in model/lama/.")
        return found[0]

    def _load(self):
        if self._model is not None:
            return
        ckpt = self._checkpoint()
        try:
            from simple_lama_inpainting import SimpleLama
        except ImportError as exc:
            raise RuntimeError("LaMa extra is not installed (simple-lama-inpainting).") from exc
        self._model = SimpleLama(device="cuda")
        self._ckpt = ckpt

    def inpaint(self, image: np.ndarray, mask: np.ndarray, prompt: str) -> np.ndarray:
        self._load()
        rgb = Image.fromarray(image[:, :, :3])
        hole = Image.fromarray(((mask > 0) * 255).astype(np.uint8))
        filled = np.array(self._model(rgb, hole))
        out = image.copy()
        out[:, :, :3] = filled[:, :, :3]
        if out.shape[2] == 4:
            alpha = out[:, :, 3]
            alpha[mask > 0] = 255
            out[:, :, 3] = alpha
        return out

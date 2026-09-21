from __future__ import annotations

from pathlib import Path

import numpy as np

from layerforge.config import resolve_path


class AnimeHairParse:
    """Optional front/back hair parser. Skips unless a local checkpoint is present."""

    name = "anime_hair_parse"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._processor = None
        self._model = None

    def _model_dir(self) -> Path:
        paths = self.cfg.get("paths") or {}
        return resolve_path(paths.get("anime_parse_dir", "model/anime_parse"))

    def available(self) -> bool:
        path = self._model_dir()
        if not path.exists():
            return False
        return any(path.glob("*.onnx")) or any(path.glob("config.json")) or any(path.glob("*.pt"))

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (front_coarse, back_coarse) uint8 masks, or None if no usable weights."""
        if not self.available():
            return None
        return None

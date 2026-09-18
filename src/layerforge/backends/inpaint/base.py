from __future__ import annotations

from typing import Protocol

import numpy as np


class InpaintBackend(Protocol):
    name: str

    def inpaint(self, image: np.ndarray, mask: np.ndarray, prompt: str) -> np.ndarray:
        """Inpaint RGB/RGBA `image` where `mask` > 0. Must not be required to preserve visible pixels; reproject does that."""
        ...

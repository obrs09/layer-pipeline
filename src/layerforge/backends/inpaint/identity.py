from __future__ import annotations

import numpy as np


class IdentityInpaint:
    name = "identity"

    def inpaint(self, image: np.ndarray, mask: np.ndarray, prompt: str) -> np.ndarray:
        return image.copy()

from __future__ import annotations

import numpy as np


def morph_open(mask: np.ndarray, px: int) -> np.ndarray:
    """Drop 1–2px rims and hairline bridges. Overlay callers should skip this."""
    if px <= 0 or mask is None:
        return mask
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(px) + 1, 2 * int(px) + 1))
    opened = cv2.morphologyEx((mask > 0).astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    return opened


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or mask is None:
        return mask
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(px) + 1, 2 * int(px) + 1))
    return cv2.dilate((mask > 0).astype(np.uint8) * 255, kernel)

from __future__ import annotations

import numpy as np


def crop_to_mask(
    image: np.ndarray,
    mask: np.ndarray,
    pad: int = 32,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Tight crop around the inpaint hole, padded for context."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        h, w = mask.shape[:2]
        return image.copy(), mask.copy(), (0, h, 0, w)
    y0 = max(0, int(ys.min()) - pad)
    x0 = max(0, int(xs.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + 1 + pad)
    x1 = min(mask.shape[1], int(xs.max()) + 1 + pad)
    return image[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy(), (y0, y1, x0, x1)


def paste_crop(
    image: np.ndarray,
    crop: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    y0, y1, x0, x1 = box
    h, w = y1 - y0, x1 - x0
    if crop.shape[0] != h or crop.shape[1] != w:
        from PIL import Image

        crop = np.array(Image.fromarray(crop).resize((w, h), Image.Resampling.LANCZOS))
    out = image.copy()
    out[y0:y1, x0:x1] = crop
    return out

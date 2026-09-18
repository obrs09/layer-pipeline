from __future__ import annotations

import numpy as np


def reproject(
    inpainted: np.ndarray,
    source: np.ndarray,
    visible: np.ndarray,
    occluded: np.ndarray,
) -> np.ndarray:
    """Keep original pixels wherever the layer is visible. Allow inpaint only in occluded."""
    if inpainted.shape != source.shape:
        raise ValueError("inpainted and source must match shape")
    vis = visible > 0
    occ = occluded > 0
    out = np.zeros_like(source)
    out[occ] = inpainted[occ]
    out[vis] = source[vis]
    if out.shape[-1] == 4:
        alpha = np.zeros(out.shape[:2], dtype=np.uint8)
        alpha[vis | occ] = 255
        out[..., 3] = alpha
        out[vis, :3] = source[vis, :3]
    return out

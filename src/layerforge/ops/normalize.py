from __future__ import annotations

import numpy as np


def to_rgba(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError("expected HxWxC image")
    if image.shape[2] == 4:
        return image
    if image.shape[2] == 3:
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
        return np.dstack([image, alpha])
    raise ValueError(f"unsupported channel count {image.shape[2]}")


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    try:
        import cv2
    except ImportError:
        from PIL import Image

        scale = max_side / longest
        size = (int(round(w * scale)), int(round(h * scale)))
        return np.array(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))
    scale = max_side / longest
    size = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def foreground_mask(image: np.ndarray, background_luma: int = 250) -> np.ndarray:
    rgba = to_rgba(image)
    alpha = rgba[:, :, 3]
    if float((alpha < 8).mean()) > 0.02:
        return ((alpha > 8).astype(np.uint8) * 255)
    rgb = rgba[:, :, :3].astype(np.int16)
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    bright = luma >= background_luma
    try:
        import cv2
    except ImportError:
        fg = ~bright
        return (fg.astype(np.uint8) * 255)
    num, labels = cv2.connectedComponents(bright.astype(np.uint8), connectivity=8)
    h, w = labels.shape
    border = np.unique(
        np.concatenate(
            [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
        )
    )
    bg_ids = {int(v) for v in border if v != 0}
    bg = np.isin(labels, list(bg_ids)) if bg_ids else bright
    fg = ~bg
    return (fg.astype(np.uint8) * 255)

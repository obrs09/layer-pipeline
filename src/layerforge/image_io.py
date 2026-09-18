from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def open_rgba(path: Path) -> np.ndarray:
    image = Image.open(path)
    return np.array(image.convert("RGBA"), dtype=np.uint8)


def open_rgb(path: Path) -> np.ndarray:
    image = Image.open(path)
    return np.array(image.convert("RGB"), dtype=np.uint8)


def save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        Image.fromarray(array, mode="L").save(path)
        return
    if array.shape[2] == 3:
        Image.fromarray(array, mode="RGB").save(path)
        return
    Image.fromarray(array, mode="RGBA").save(path)


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return [x0, y0, x1 - x0, y1 - y0]


def alpha_over(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    out = base.astype(np.float32)
    src = overlay.astype(np.float32)
    a = (src[..., 3:4] / 255.0)
    out[..., :3] = src[..., :3] * a + out[..., :3] * (1.0 - a)
    out[..., 3:4] = src[..., 3:4] + out[..., 3:4] * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)

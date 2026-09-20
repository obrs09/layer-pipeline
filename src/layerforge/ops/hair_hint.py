from __future__ import annotations

import cv2
import numpy as np

from layerforge.backends.segment.base import LayerMask
from layerforge.ops.morph import dilate_mask


def fill_box(shape: tuple[int, int], box: list[float]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 255
    return mask


def seed_region(
    character: np.ndarray,
    hair_box: list[float],
    *,
    face_box: list[float] | None = None,
    claimed: np.ndarray | None = None,
    face_dilate_px: int = 8,
) -> np.ndarray:
    """Hair-box pixels that are not face-box / already-claimed clothes."""
    region = fill_box(character.shape, hair_box) > 0
    region &= character > 0
    if face_box is not None:
        face = fill_box(character.shape, face_box)
        if face_dilate_px > 0:
            face = dilate_mask(face, face_dilate_px)
        region &= face == 0
    if claimed is not None:
        region &= claimed == 0
    return region.astype(np.uint8) * 255


def drop_skin_pixels(image: np.ndarray, region: np.ndarray, *, a_min: int = 134, l_lo: int = 70, l_hi: int = 210) -> np.ndarray:
    """Drop warm mid-tone pixels so the seed prefers hair over cheek."""
    vis = region > 0
    if int(vis.sum()) < 8:
        return region
    bgr = image[:, :, :3]
    if bgr.shape[2] == 3:
        lab = cv2.cvtColor(bgr, cv2.COLOR_RGB2LAB)
    else:
        return region
    skin = vis & (lab[..., 1] >= a_min) & (lab[..., 0] >= l_lo) & (lab[..., 0] <= l_hi)
    kept = vis & ~skin
    if int(kept.sum()) < 8:
        return region
    return kept.astype(np.uint8) * 255


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    vis = mask > 0
    if not vis.any():
        return None
    ys, xs = np.nonzero(vis)
    return (float(xs.mean()), float(ys.mean()))


def claimed_from(others: list[LayerMask], roles: tuple[str, ...]) -> np.ndarray | None:
    claimed = None
    for layer in others:
        if layer.role not in roles:
            continue
        vis = layer.visible > 0
        if claimed is None:
            claimed = np.zeros(vis.shape, dtype=bool)
        claimed |= vis
    return claimed


def hair_positive(
    image: np.ndarray,
    character: np.ndarray,
    hair_box: list[float],
    others: list[LayerMask],
    *,
    face_dilate_px: int = 8,
    drop_skin: bool = True,
) -> tuple[float, float] | None:
    face = next((layer for layer in others if layer.role == "face"), None)
    face_box = list(face.bbox) if face is not None and int((face.visible > 0).sum()) else None
    if face_box is not None:
        x, y, bw, bh = face_box
        face_box = [float(x), float(y), float(x + bw), float(y + bh)]
    claimed = claimed_from(others, ("clothes", "body"))
    region = seed_region(
        character,
        hair_box,
        face_box=face_box,
        claimed=claimed,
        face_dilate_px=face_dilate_px,
    )
    if drop_skin:
        region = drop_skin_pixels(image, region)
    return centroid(region)

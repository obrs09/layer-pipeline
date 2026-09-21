from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from layerforge.ops.morph import dilate_mask


@dataclass
class HairSplitResult:
    strategy: str
    front: np.ndarray
    back: np.ndarray
    notes: str
    extras: dict = field(default_factory=dict)


def _as_bool(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=bool)
    return mask > 0


def fill_box(shape: tuple[int, int], box: list[float]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 255
    return mask


def bangs_region(
    face: np.ndarray,
    keypoints: list[dict] | None = None,
    *,
    forehead_frac: float = 0.42,
    bangs_up_frac: float = 0.35,
    min_score: float = 0.2,
) -> np.ndarray:
    """Forehead + eye band where front hair usually sits. Landmarks first, face box fallback."""
    h, w = face.shape[:2]
    region = np.zeros((h, w), dtype=np.uint8)
    brows: list[tuple[float, float]] = []
    eyes: list[tuple[float, float]] = []
    for item in keypoints or []:
        name = str(item.get("name") or "")
        if float(item.get("score") or 0.0) < min_score:
            continue
        x, y = float(item.get("x", 0.0)), float(item.get("y", 0.0))
        if name.startswith("face_"):
            try:
                idx = int(name.split("_")[1])
            except ValueError:
                continue
            # 68-pt: 17-26 brows, 36-47 eyes
            if 17 <= idx <= 26:
                brows.append((x, y))
            elif 36 <= idx <= 47:
                eyes.append((x, y))
    anchors = brows or eyes
    if len(anchors) >= 3:
        xs = [p[0] for p in anchors]
        ys = [p[1] for p in anchors]
        x0, x1 = min(xs), max(xs)
        y_brow = min(ys)
        width = max(8.0, x1 - x0)
        pad = width * 0.25
        face_h = 0.0
        vis = face > 0
        if vis.any():
            fy = np.where(vis)[0]
            face_h = float(fy.max() - fy.min() + 1)
        up = max(8.0, (face_h or width) * bangs_up_frac)
        y0 = max(0.0, y_brow - up)
        y1 = min(float(h), max(ys) + width * 0.15)
        return fill_box((h, w), [x0 - pad, y0, x1 + pad, y1])
    vis = face > 0
    if not vis.any():
        return region
    ys, xs = np.where(vis)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    split = y0 + max(4, int((y1 - y0) * forehead_frac))
    band = np.zeros((h, w), dtype=np.uint8)
    band[max(0, y0 - int((y1 - y0) * bangs_up_frac)) : split, x0:x1] = 255
    halo = dilate_mask(face, 10) > 0
    return ((band > 0) & halo).astype(np.uint8) * 255


def lower_face(face: np.ndarray, bangs: np.ndarray) -> np.ndarray:
    return ((face > 0) & (bangs == 0)).astype(np.uint8) * 255


def grow_geodesic(seeds: np.ndarray, walkable: np.ndarray, *, max_dist: int) -> np.ndarray:
    """Dilate seeds through walkable pixels, stopping at barriers. max_dist is in pixels."""
    if max_dist <= 0:
        return ((seeds > 0) & (walkable > 0)).astype(np.uint8) * 255
    walk = walkable > 0
    front = (seeds > 0) & walk
    if not front.any() and (seeds > 0).any():
        front = seeds > 0
        walk = walk | front
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    current = front.astype(np.uint8) * 255
    for _ in range(int(max_dist)):
        grown = cv2.dilate(current, kernel) > 0
        nxt = (grown & walk).astype(np.uint8) * 255
        if np.array_equal(nxt, current):
            break
        current = nxt
    return current


def split_occlusion(
    hair: np.ndarray,
    face: np.ndarray,
    *,
    clothes: np.ndarray | None = None,
    body: np.ndarray | None = None,
    keypoints: list[dict] | None = None,
    forehead_frac: float = 0.42,
    bangs_up_frac: float = 0.35,
    grow_px: int = 48,
    barrier_dilate_px: int = 2,
    min_front_px: int = 32,
) -> HairSplitResult:
    """Front hair covers the face; back is the leftover. Body/clothes block wrap-around."""
    shape = hair.shape[:2]
    hair_b = _as_bool(hair, shape)
    face_b = _as_bool(face, shape)
    bangs = bangs_region(
        face if face is not None else np.zeros(shape, dtype=np.uint8),
        keypoints,
        forehead_frac=forehead_frac,
        bangs_up_frac=bangs_up_frac,
    )
    covering = hair_b & face_b & (bangs > 0)
    seeds = (hair_b & (bangs > 0)) | covering
    barrier = _as_bool(clothes, shape) | _as_bool(body, shape) | (lower_face(face, bangs) > 0)
    if barrier_dilate_px > 0 and barrier.any():
        barrier = dilate_mask(barrier.astype(np.uint8) * 255, barrier_dilate_px) > 0
    walkable = hair_b & ~barrier
    walkable |= seeds
    front = grow_geodesic(seeds.astype(np.uint8) * 255, walkable.astype(np.uint8) * 255, max_dist=grow_px) > 0
    front &= hair_b
    if int(front.sum()) < min_front_px:
        front = covering
    back = hair_b & ~front
    return HairSplitResult(
        strategy="occlusion",
        front=(front.astype(np.uint8) * 255),
        back=(back.astype(np.uint8) * 255),
        notes="occlusion inversion: bangs seeds, body barrier, back = hair \\ front",
        extras={
            "bangs_px": int((bangs > 0).sum()),
            "seed_px": int(seeds.sum()),
            "cover_px": int(covering.sum()),
            "front_px": int(front.sum()),
            "back_px": int(back.sum()),
        },
    )


def split_depth(
    hair: np.ndarray,
    face: np.ndarray,
    depth: np.ndarray,
    *,
    bangs: np.ndarray | None = None,
    eps: float = 0.04,
    min_front_px: int = 32,
) -> HairSplitResult:
    """Slice hair on the face depth plane. Front is the side that matches the bangs prior."""
    hair_b = hair > 0
    face_b = face > 0
    if depth.shape[:2] != hair.shape[:2]:
        depth = cv2.resize(depth.astype(np.float32), (hair.shape[1], hair.shape[0]), interpolation=cv2.INTER_LINEAR)
    span = float(np.percentile(depth, 95) - np.percentile(depth, 5)) or 1.0
    cut = max(float(eps) * span, 1e-6)
    bangs_m = bangs if bangs is not None else bangs_region(face)
    z_face = float(np.median(depth[face_b])) if face_b.any() else float(np.median(depth))
    low = hair_b & (depth < (z_face - cut))
    high = hair_b & (depth > (z_face + cut))
    bangs_b = bangs_m > 0
    if int((high & bangs_b).sum()) > int((low & bangs_b).sum()):
        front, back, sign = high, low, -1
    else:
        front, back, sign = low, high, 1
    mid = hair_b & ~front & ~back
    front = front | (mid & bangs_b)
    back = hair_b & ~front
    if int(front.sum()) < min_front_px:
        front = hair_b & face_b
        back = hair_b & ~front
    return HairSplitResult(
        strategy="depth",
        front=(front.astype(np.uint8) * 255),
        back=(back.astype(np.uint8) * 255),
        notes=f"depth plane at face median; sign={sign}",
        extras={
            "z_face": z_face,
            "sign": sign,
            "cut": cut,
            "front_px": int(front.sum()),
            "back_px": int(back.sum()),
        },
    )


def erode_core(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or mask is None:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(px) + 1, 2 * int(px) + 1))
    return cv2.erode((mask > 0).astype(np.uint8) * 255, kernel)


def sample_points(mask: np.ndarray, count: int) -> list[tuple[float, float]]:
    vis = mask > 0
    if not vis.any() or count <= 0:
        return []
    ys, xs = np.where(vis)
    if xs.size <= count:
        return [(float(x), float(y)) for x, y in zip(xs, ys)]
    rng = np.random.default_rng(0)
    pick = rng.choice(xs.size, size=int(count), replace=False)
    return [(float(xs[i]), float(ys[i])) for i in pick]


def split_parsing(
    hair: np.ndarray,
    front_coarse: np.ndarray,
    back_coarse: np.ndarray,
    *,
    erode_px: int = 8,
    min_front_px: int = 32,
) -> HairSplitResult:
    """Keep the high-confidence core of a coarse parser, clipped to whole hair."""
    hair_b = hair > 0
    front_core = erode_core(front_coarse, erode_px) > 0
    back_core = erode_core(back_coarse, erode_px) > 0
    front = hair_b & front_core & ~back_core
    if int(front.sum()) < min_front_px:
        front = hair_b & (front_coarse > 0) & ~(back_core)
    back = hair_b & ~front
    return HairSplitResult(
        strategy="parsing",
        front=(front.astype(np.uint8) * 255),
        back=(back.astype(np.uint8) * 255),
        notes="parser core eroded then clipped to whole hair",
        extras={
            "front_px": int(front.sum()),
            "back_px": int(back.sum()),
            "pos_points": sample_points(front_core.astype(np.uint8) * 255, 6),
            "neg_points": sample_points(back_core.astype(np.uint8) * 255, 4),
        },
    )

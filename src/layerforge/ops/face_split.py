from __future__ import annotations

import numpy as np

from layerforge.backends.segment.base import LayerMask
from layerforge.ops.morph import dilate_mask
from layerforge.taxonomy import Taxonomy


def split_face_colors(
    image: np.ndarray,
    layers: list[LayerMask],
    taxonomy: Taxonomy,
    cfg: dict | None = None,
) -> tuple[list[LayerMask], dict]:
    """Punch eyes/mouth out of face, then split leftover pixels by color.

    Skin grows from cheek seeds through 3x3 neighbors and stops at Canny lineart
    (or a global Lab ball if skin_mode=global). Hair-colored leftover joins
    hair_front. Neck split is off unless cut_neck is true.
    """
    cfg = dict(cfg or {})
    report: dict = {"enabled": bool(cfg.get("enabled", True)), "applied": False}
    if not cfg or not report["enabled"]:
        report["enabled"] = False
        return layers, report
    faces = [layer for layer in layers if layer.role == "face" and int((layer.visible > 0).sum()) > 0]
    if not faces:
        return layers, report
    face = max(faces, key=lambda m: int((m.visible > 0).sum()))
    punch_roles = list(cfg.get("punch_roles") or ("eye_l", "eye_r", "mouth"))
    neck_role = str(cfg.get("neck_role") or "neck")
    hair_role = str(cfg.get("hair_role") or "hair_front")
    hair_fallback = str(cfg.get("hair_fallback") or "hair_back")
    expand_px = int(cfg.get("expand_px", 1))
    skin_dist = float(cfg.get("skin_dist", 24))
    hair_dist = float(cfg.get("hair_dist", 24))
    hair_slack = float(cfg.get("hair_slack", 8))
    l_weight = float(cfg.get("l_weight", 0.08))
    jaw_frac = float(cfg.get("jaw_frac", 0.82))
    jaw_pad_px = int(cfg.get("jaw_pad_px", 4))
    neck_width_frac = float(cfg.get("neck_width_frac", 0.62))
    min_neck_px = int(cfg.get("min_neck_px", 64))
    min_hair_px = int(cfg.get("min_hair_px", 32))
    cheek_dilate_px = int(cfg.get("cheek_dilate_px", 24))
    skin_mode = str(cfg.get("skin_mode") or "grow")
    local_dist = float(cfg.get("local_dist", 20))
    cut_neck = bool(cfg.get("cut_neck", False))
    edge_stop = bool(cfg.get("edge_stop", True))
    canny_low = int(cfg.get("canny_low", 50))
    canny_high = int(cfg.get("canny_high", 120))

    rgb = image[:, :, :3]
    lab = _to_lab(rgb)
    punched = _union_roles(layers, punch_roles)
    remaining = (face.visible > 0) & ~punched
    report["punched_roles"] = [role for role in punch_roles if any(layer.role == role for layer in layers)]
    report["remaining_px"] = int(remaining.sum())
    if int(remaining.sum()) < 32:
        face.visible = remaining.astype(np.uint8) * 255
        face.notes = _join(face.notes, "face_split punched overlays")
        report["applied"] = True
        return layers, report

    skin_seed, seed_mask = _skin_seed(
        lab, remaining, layers, punch_roles, cheek_dilate_px, [hair_role, hair_fallback]
    )
    if skin_seed is None:
        return layers, report
    d_skin = _lab_dist(lab, skin_seed, l_weight)
    hair_seed = _hair_seed(lab, remaining, remaining & (d_skin <= skin_dist), layers, hair_role, hair_fallback)
    if hair_seed is not None:
        d_hair = _lab_dist(lab, hair_seed, l_weight)
        cap = remaining & (d_skin <= skin_dist) & (d_skin < d_hair + hair_slack)
        hair_like = remaining & (d_hair <= hair_dist) & (d_hair <= d_skin)
    else:
        cap = remaining & (d_skin <= skin_dist)
        hair_like = np.zeros_like(remaining)

    if skin_mode == "grow":
        warm = _warm_skin(lab, remaining)
        start = (warm & cap) | (seed_mask & remaining)
        edges = _lineart_edges(rgb, canny_low, canny_high) if edge_stop else np.zeros_like(remaining)
        walk = (cap | punched | start) & ~(edges & ~start)
        grown = _grow_skin(lab, walk, start, skin_seed, local_dist, l_weight)
        skin = grown & remaining
        if int(skin.sum()) < 32:
            skin = cap
        report["skin_mode"] = "grow"
        report["edge_stop"] = bool(edge_stop)
    else:
        skin = cap
        report["skin_mode"] = "global"

    already_neck = any(
        layer.role == neck_role and int((layer.visible > 0).sum()) >= min_neck_px for layer in layers
    )
    if not cut_neck or already_neck:
        jaw_y = 0
        neck = np.zeros_like(remaining)
        face_skin = skin
        report["neck_skipped"] = True
        report["cut_neck"] = False
    else:
        jaw_y = _jaw_y(skin, layers, jaw_frac, jaw_pad_px, neck_width_frac)
        neck = _neck_from_skin(skin, jaw_y, min_neck_px)
        face_skin = skin & ~neck
        report["cut_neck"] = True
    if expand_px > 0:
        face_skin = (dilate_mask(face_skin.astype(np.uint8) * 255, expand_px) > 0) & remaining & ~punched
        if int(neck.sum()) > 0:
            neck = (dilate_mask(neck.astype(np.uint8) * 255, expand_px) > 0) & remaining & ~punched
        overlap = face_skin & neck
        if overlap.any():
            ys = np.arange(remaining.shape[0])[:, None]
            face_skin = face_skin & ((ys <= jaw_y) | ~overlap)
            neck = neck & ~face_skin

    leftover = remaining & ~face_skin & ~neck
    if hair_seed is not None:
        hair = leftover & hair_like
    else:
        hair = np.zeros_like(leftover)
    unclaimed = leftover & ~hair

    face.visible = face_skin.astype(np.uint8) * 255
    face.notes = _join(face.notes, "face_split color")
    report.update(
        {
            "applied": True,
            "face_px": int(face_skin.sum()),
            "neck_px": int(neck.sum()),
            "hair_px": int(hair.sum()),
            "unclaimed_px": int(unclaimed.sum()),
            "jaw_y": int(jaw_y),
        }
    )

    if int(neck.sum()) >= min_neck_px and neck_role in taxonomy.roles:
        existing = [layer for layer in layers if layer.role == neck_role]
        if existing:
            target = max(existing, key=lambda m: int((m.visible > 0).sum()))
            target.visible = ((target.visible > 0) | neck).astype(np.uint8) * 255
            target.notes = _join(target.notes, "face_split neck")
        else:
            layers.append(
                LayerMask(
                    role=neck_role,
                    label=neck_role,
                    visible=neck.astype(np.uint8) * 255,
                    source="sam",
                    notes="face_split neck",
                )
            )
        report["neck_role"] = neck_role
    elif int(neck.sum()) > 0:
        face.visible = ((face.visible > 0) | neck).astype(np.uint8) * 255
        report["neck_px"] = 0
        report["face_px"] = int((face.visible > 0).sum())

    if int(hair.sum()) >= min_hair_px:
        dest = _pick_hair_dest(layers, hair_role, hair_fallback, taxonomy)
        report["hair_role"] = dest
        found = [layer for layer in layers if layer.role == dest]
        if found:
            target = max(found, key=lambda m: int((m.visible > 0).sum()))
            target.visible = ((target.visible > 0) | hair).astype(np.uint8) * 255
            target.notes = _join(target.notes, "face_split hair")
        else:
            layers.append(
                LayerMask(
                    role=dest,
                    label=dest,
                    visible=hair.astype(np.uint8) * 255,
                    source="sam",
                    notes="face_split hair",
                )
            )
    elif int(hair.sum()) > 0:
        report["hair_px"] = int(hair.sum())
        report["unclaimed_px"] = int((unclaimed | hair).sum())

    return layers, report


def _lineart_edges(rgb: np.ndarray, canny_low: int, canny_high: int) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(blur, int(canny_low), int(canny_high)) > 0


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _lab_dist(lab: np.ndarray, seed: np.ndarray, l_weight: float) -> np.ndarray:
    delta = lab - seed.reshape((1, 1, 3))
    return np.sqrt(l_weight * delta[:, :, 0] ** 2 + delta[:, :, 1] ** 2 + delta[:, :, 2] ** 2)


def _median_lab(lab: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    pix = lab[mask]
    if pix.size == 0:
        return None
    return np.median(pix, axis=0)


def _union_roles(layers: list[LayerMask], roles: list[str]) -> np.ndarray:
    want = set(roles)
    mask = None
    for layer in layers:
        if layer.role not in want:
            continue
        vis = layer.visible > 0
        mask = vis if mask is None else (mask | vis)
    if mask is None:
        h, w = layers[0].visible.shape
        return np.zeros((h, w), dtype=bool)
    return mask


def _skin_seed(
    lab: np.ndarray,
    remaining: np.ndarray,
    layers: list[LayerMask],
    punch_roles: list[str],
    cheek_dilate_px: int,
    hair_roles: list[str],
) -> tuple[np.ndarray | None, np.ndarray]:
    empty = np.zeros_like(remaining)
    hair = _union_roles(layers, hair_roles)
    keep = remaining & ~hair
    eyes = _union_roles(layers, [role for role in punch_roles if role.startswith("eye")])
    warm_all = _warm_skin(lab, keep if keep.any() else remaining)
    cheek = empty
    if eyes.any() and cheek_dilate_px > 0:
        band = (dilate_mask(eyes.astype(np.uint8) * 255, cheek_dilate_px) > 0) & keep & ~eyes
        cheek = _warm_skin(lab, band)
    seed_src = cheek if int(cheek.sum()) >= 32 else warm_all
    seed = _median_lab(lab, seed_src)
    if seed is None:
        return None, empty
    return seed, cheek if cheek.any() else seed_src


def _grow_skin(
    lab: np.ndarray,
    walk: np.ndarray,
    seed_mask: np.ndarray,
    seed_color: np.ndarray,
    local_dist: float,
    l_weight: float,
    max_iter: int = 2048,
) -> np.ndarray:
    """8-connected grow. A pixel joins if its Lab is within local_dist of grown 3x3 neighbors."""
    grown = (seed_mask & walk).copy()
    if not grown.any():
        d = _lab_dist(lab, seed_color, l_weight)
        grown = walk & (d <= local_dist)
    if not grown.any():
        return walk
    h, w = walk.shape
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    for _ in range(max_iter):
        acc = np.zeros((h, w, 3), dtype=np.float32)
        cnt = np.zeros((h, w), dtype=np.float32)
        for dy, dx in offsets:
            y0, y1 = max(0, dy), h + min(0, dy)
            x0, x1 = max(0, dx), w + min(0, dx)
            sy0, sy1 = max(0, -dy), h - max(0, dy)
            sx0, sx1 = max(0, -dx), w - max(0, dx)
            src = grown[sy0:sy1, sx0:sx1]
            acc[y0:y1, x0:x1] += lab[sy0:sy1, sx0:sx1] * src[..., None]
            cnt[y0:y1, x0:x1] += src.astype(np.float32)
        border = walk & ~grown & (cnt > 0)
        if not border.any():
            break
        mean = acc / np.maximum(cnt, 1.0)[..., None]
        delta = lab - mean
        local = np.sqrt(l_weight * delta[:, :, 0] ** 2 + delta[:, :, 1] ** 2 + delta[:, :, 2] ** 2)
        add = border & (local <= local_dist)
        if not add.any():
            break
        grown |= add
    return grown


def _warm_skin(lab: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """OpenCV Lab a≈128 is gray; anime skin sits on the red side. Drop bangs from the seed."""
    if not mask.any():
        return mask
    warm = mask & (lab[:, :, 1] >= 132)
    if int(warm.sum()) >= 32:
        return warm
    return mask


def _hair_seed(
    lab: np.ndarray,
    remaining: np.ndarray,
    skin: np.ndarray,
    layers: list[LayerMask],
    hair_role: str,
    hair_fallback: str,
) -> np.ndarray | None:
    hair = _union_roles(layers, [hair_role, hair_fallback])
    seed = _median_lab(lab, hair)
    if seed is not None:
        return seed
    other = remaining & ~skin
    ys, xs = np.where(other)
    if ys.size == 0:
        return None
    y_cut = int(np.quantile(ys, 0.45))
    upper = other.copy()
    upper[y_cut + 1 :] = False
    return _median_lab(lab, upper if upper.any() else other)


def _jaw_y(
    skin: np.ndarray,
    layers: list[LayerMask],
    jaw_frac: float,
    jaw_pad_px: int,
    neck_width_frac: float,
) -> int:
    """Jaw is where skin narrows to a neck, searched below the eyes/mouth."""
    widths = skin.sum(axis=1).astype(np.int32)
    ys = np.where(widths > 0)[0]
    if ys.size == 0:
        return 0
    y0, y1 = int(ys[0]), int(ys[-1])
    fallback = y0 + int(round(jaw_frac * max(1, y1 - y0)))
    cheek_peak = float(max(int(widths[ys].max()), 1))
    peak_y = int(ys[int(np.argmax(widths[ys]))])
    start = peak_y
    eyes = _union_roles(layers, ["eye_l", "eye_r"])
    if eyes.any():
        eye_ys = np.where(eyes.any(axis=1))[0]
        if eye_ys.size:
            start = max(start, int(eye_ys[-1]) + 2)
    mouths = [layer for layer in layers if layer.role == "mouth" and int((layer.visible > 0).sum()) > 0]
    if mouths:
        _x, y, _w, h = mouths[0].bbox
        start = max(start, int(y + h + jaw_pad_px))
    search = widths[start : y1 + 1]
    if search.size == 0:
        return fallback
    pinch = np.where(search <= neck_width_frac * cheek_peak)[0]
    if pinch.size:
        return start + int(pinch[0])
    return max(fallback, start)


def _neck_from_skin(skin: np.ndarray, jaw_y: int, min_neck_px: int) -> np.ndarray:
    import cv2

    below = skin.copy()
    below[: max(0, int(jaw_y) + 1)] = False
    if int(below.sum()) < min_neck_px:
        return np.zeros_like(skin)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(below.astype(np.uint8), connectivity=8)
    if n <= 1:
        return below
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[best, cv2.CC_STAT_AREA]) < min_neck_px:
        return np.zeros_like(skin)
    return labels == best


def _pick_hair_dest(
    layers: list[LayerMask],
    hair_role: str,
    hair_fallback: str,
    taxonomy: Taxonomy,
) -> str:
    del layers
    if hair_role in taxonomy.roles:
        return hair_role
    if hair_fallback in taxonomy.roles:
        return hair_fallback
    return hair_role


def _join(notes: str, extra: str) -> str:
    if not notes:
        return extra
    if extra in notes:
        return notes
    return f"{notes}; {extra}"

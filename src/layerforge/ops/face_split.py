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

    Skin above the jaw stays face. Skin below becomes neck. Hair-colored leftover
    joins hair_front (else hair_back). Anything else is dropped so later stages
    treat it as unclaimed. Expand skin selections by expand_px before the hair grab.
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
    expand_px = int(cfg.get("expand_px", 3))
    skin_dist = float(cfg.get("skin_dist", 18))
    hair_dist = float(cfg.get("hair_dist", 24))
    l_weight = float(cfg.get("l_weight", 0.15))
    jaw_frac = float(cfg.get("jaw_frac", 0.82))
    jaw_pad_px = int(cfg.get("jaw_pad_px", 4))
    neck_width_frac = float(cfg.get("neck_width_frac", 0.62))
    min_neck_px = int(cfg.get("min_neck_px", 64))
    min_hair_px = int(cfg.get("min_hair_px", 32))
    cheek_dilate_px = int(cfg.get("cheek_dilate_px", 12))

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

    skin_seed = _skin_seed(lab, remaining, layers, punch_roles, cheek_dilate_px)
    if skin_seed is None:
        return layers, report
    d_skin = _lab_dist(lab, skin_seed, l_weight)
    hair_seed = _hair_seed(lab, remaining, remaining & (d_skin <= skin_dist), layers, hair_role, hair_fallback)
    if hair_seed is not None:
        d_hair = _lab_dist(lab, hair_seed, l_weight)
        skin = remaining & (d_skin <= skin_dist) & (d_skin < d_hair)
        hair_like = remaining & (d_hair <= hair_dist) & (d_hair <= d_skin)
    else:
        skin = remaining & (d_skin <= skin_dist)
        hair_like = np.zeros_like(remaining)

    jaw_y = _jaw_y(skin, layers, jaw_frac, jaw_pad_px, neck_width_frac)
    neck = _neck_from_skin(skin, jaw_y, min_neck_px)
    face_skin = skin & ~neck
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
) -> np.ndarray | None:
    eyes = _union_roles(layers, [role for role in punch_roles if role.startswith("eye")])
    if eyes.any() and cheek_dilate_px > 0:
        band = (dilate_mask(eyes.astype(np.uint8) * 255, cheek_dilate_px) > 0) & remaining & ~eyes
        seed = _median_lab(lab, band)
        if seed is not None:
            return seed
    ys, xs = np.where(remaining)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    cy0 = y0 + int(0.25 * max(1, y1 - y0))
    cy1 = y0 + int(0.65 * max(1, y1 - y0))
    cx0 = x0 + int(0.25 * max(1, x1 - x0))
    cx1 = x0 + int(0.75 * max(1, x1 - x0))
    center = np.zeros_like(remaining)
    center[cy0:cy1, cx0:cx1] = remaining[cy0:cy1, cx0:cx1]
    return _median_lab(lab, center if center.any() else remaining)


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
    peak = float(max(int(search.max()), 1))
    pinch = np.where(search <= neck_width_frac * peak)[0]
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
    have = {layer.role for layer in layers}
    if hair_role in have or hair_role in taxonomy.roles:
        if hair_role in have:
            return hair_role
        if hair_fallback in have:
            return hair_fallback
        return hair_role
    if hair_fallback in have or hair_fallback in taxonomy.roles:
        return hair_fallback
    return hair_role


def _join(notes: str, extra: str) -> str:
    if not notes:
        return extra
    if extra in notes:
        return notes
    return f"{notes}; {extra}"

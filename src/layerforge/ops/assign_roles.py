from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from layerforge.backends.segment.base import LayerMask
from layerforge.image_io import bbox_from_mask
from layerforge.taxonomy import Taxonomy


@dataclass
class _Feat:
    layer: LayerMask
    area: int
    area_frac: float
    cx: float
    cy: float
    skin: float
    compact: float
    aspect: float


def assign_roles(
    layers: list[LayerMask],
    source: np.ndarray,
    taxonomy: Taxonomy,
) -> list[LayerMask]:
    """Label SAM `unknown_*` fragments with taxonomy roles using geometry + skin heuristics.

    Already-mapped roles (Imagine aliases, SAM body box) are left alone.
    Not a vision model — leftovers become `acc`.
    """
    unlabeled = [layer for layer in layers if _needs_role(layer, taxonomy)]
    if not unlabeled:
        return layers
    rgb = source[:, :, :3]
    char = np.zeros(rgb.shape[:2], dtype=bool)
    for layer in layers:
        char |= layer.visible > 0
    if int(char.sum()) < 32:
        return layers
    box = bbox_from_mask((char.astype(np.uint8) * 255))
    char_area = max(1, int(char.sum()))
    feats = [_features(layer, rgb, box, char_area) for layer in unlabeled]
    taken = {layer.role for layer in layers if layer.role in taxonomy.roles}

    face = _pick(feats, taken, "face", _face_score)
    hair_feats = [f for f in feats if f.layer.role.startswith("unknown")]
    _assign_hair(hair_feats, face, taken, taxonomy)
    _pick(feats, taken, "clothes", _clothes_score)
    # Second clothes piece is allowed.
    _pick(feats, taken, "clothes", _clothes_score, unique=False)
    _pick(feats, taken, "arm_l", lambda f: _arm_score(f, left=True))
    _pick(feats, taken, "arm_r", lambda f: _arm_score(f, left=False))

    for feat in feats:
        if not feat.layer.role.startswith("unknown"):
            continue
        feat.layer.role = "acc"
        feat.layer.notes = _note(feat.layer, "acc")
    return layers


def _needs_role(layer: LayerMask, taxonomy: Taxonomy) -> bool:
    if layer.role in taxonomy.roles:
        return False
    return str(layer.role).startswith("unknown")


def _features(layer: LayerMask, rgb: np.ndarray, box: list[int], char_area: int) -> _Feat:
    vis = layer.visible > 0
    area = int(vis.sum())
    ys, xs = np.where(vis)
    bx, by, bw, bh = box
    cx = float((xs.mean() - bx) / max(1, bw)) if xs.size else 0.5
    cy = float((ys.mean() - by) / max(1, bh)) if ys.size else 0.5
    x, y, w, h = layer.bbox
    bbox_area = max(1, w * h)
    return _Feat(
        layer=layer,
        area=area,
        area_frac=area / char_area,
        cx=cx,
        cy=cy,
        skin=_skin_ratio(rgb, vis),
        compact=area / bbox_area,
        aspect=h / max(1, w),
    )


def _skin_ratio(rgb: np.ndarray, vis: np.ndarray) -> float:
    n = int(vis.sum())
    if n == 0:
        return 0.0
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    skin = (
        (r > 90)
        & (g > 55)
        & (b > 35)
        & (r >= g)
        & ((r - b) >= 18)
        & ((r - g) < 90)
        & ((g - b) < 70)
    )
    return float((skin & vis).sum()) / n


def _note(layer: LayerMask, role: str) -> str:
    extra = f"role={role} heuristic"
    if layer.notes:
        return f"{layer.notes}; {extra}"
    return extra


def _pick(
    feats: list[_Feat],
    taken: set[str],
    role: str,
    score_fn,
    unique: bool = True,
    min_score: float = 0.35,
) -> _Feat | None:
    if unique and role in taken:
        return None
    best: tuple[float, _Feat] | None = None
    for feat in feats:
        if not feat.layer.role.startswith("unknown"):
            continue
        if feat.area_frac > 0.55:
            continue
        score = float(score_fn(feat))
        if score < min_score:
            continue
        if best is None or score > best[0]:
            best = (score, feat)
    if best is None:
        return None
    feat = best[1]
    feat.layer.role = role
    feat.layer.notes = _note(feat.layer, role)
    taken.add(role)
    return feat


def _face_score(feat: _Feat) -> float:
    if not (0.05 <= feat.cy <= 0.45):
        return -1.0
    if feat.skin < 0.28:
        return -1.0
    if not (0.012 <= feat.area_frac <= 0.28):
        return -1.0
    cy = 1.0 - min(1.0, abs(feat.cy - 0.22) / 0.22)
    return 0.5 * feat.skin + 0.3 * cy + 0.2 * feat.compact


def _hair_score(feat: _Feat) -> float:
    if feat.cy > 0.52:
        return -1.0
    if feat.skin > 0.42:
        return -1.0
    if feat.area_frac < 0.04:
        return -1.0
    return 0.4 * (1.0 - feat.skin) + 0.4 * (1.0 - feat.cy) + 0.2 * min(feat.area_frac / 0.25, 1.0)


def _clothes_score(feat: _Feat) -> float:
    if feat.skin > 0.42:
        return -1.0
    if feat.area_frac < 0.03:
        return -1.0
    if feat.cy < 0.22:
        return -1.0
    cy = 1.0 - min(1.0, abs(feat.cy - 0.55) / 0.4)
    return 0.35 * (1.0 - feat.skin) + 0.3 * cy + 0.35 * min(feat.area_frac / 0.3, 1.0)


def _arm_score(feat: _Feat, *, left: bool) -> float:
    if feat.aspect < 1.25:
        return -1.0
    if not (0.008 <= feat.area_frac <= 0.16):
        return -1.0
    if feat.cy < 0.28:
        return -1.0
    if left and feat.cx > 0.42:
        return -1.0
    if not left and feat.cx < 0.58:
        return -1.0
    side = (0.42 - feat.cx) if left else (feat.cx - 0.58)
    return 0.4 * min(feat.aspect / 2.5, 1.0) + 0.3 * min(max(side, 0.0) / 0.3, 1.0) + 0.3 * min(feat.skin + 0.2, 1.0)


def _assign_hair(feats: list[_Feat], face: _Feat | None, taken: set[str], taxonomy: Taxonomy) -> None:
    del taxonomy
    scored = []
    for feat in feats:
        if not feat.layer.role.startswith("unknown"):
            continue
        score = _hair_score(feat)
        if score >= 0.35:
            scored.append((score, feat))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return
    if face is not None:
        fx, fy, fw, fh = face.layer.bbox
        face_box = np.zeros_like(face.layer.visible, dtype=bool)
        face_box[fy : fy + fh, fx : fx + fw] = True
        front = None
        back = None
        for _, feat in scored:
            vis = feat.layer.visible > 0
            hit = int((vis & face_box).sum()) / max(1, int(vis.sum()))
            if hit >= 0.08 and front is None:
                front = feat
            elif back is None:
                back = feat
        if back is None and front is not None and len(scored) == 1:
            back = front
            front = None
        if back is not None and "hair_back" not in taken:
            back.layer.role = "hair_back"
            back.layer.notes = _note(back.layer, "hair_back")
            taken.add("hair_back")
        if front is not None and front is not back and "hair_front" not in taken:
            front.layer.role = "hair_front"
            front.layer.notes = _note(front.layer, "hair_front")
            taken.add("hair_front")
        return
    first = scored[0][1]
    first.layer.role = "hair_back"
    first.layer.notes = _note(first.layer, "hair_back")
    taken.add("hair_back")
    if len(scored) > 1:
        second = scored[1][1]
        second.layer.role = "hair_front"
        second.layer.notes = _note(second.layer, "hair_front")
        taken.add("hair_front")

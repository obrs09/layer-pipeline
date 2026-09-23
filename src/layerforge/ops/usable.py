from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from layerforge.backends.segment.base import LayerMask
from layerforge.taxonomy import RoleSpec


@dataclass
class UsableResult:
    ok: bool
    score: float
    reasons: list[str]
    mask: np.ndarray


def drop_crumbs(mask: np.ndarray, spec: RoleSpec, ref_area: int) -> np.ndarray:
    vis = (mask > 0).astype(np.uint8)
    if int(vis.sum()) == 0:
        return vis * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(vis, connectivity=8)
    min_area = max(16, int(spec.crumb_frac * max(1, ref_area)))
    sizes = [(int(stats[idx, cv2.CC_STAT_AREA]), idx) for idx in range(1, num)]
    sizes.sort(reverse=True)
    kept_ids: list[int] = []
    for area, idx in sizes:
        if len(kept_ids) >= spec.max_components:
            break
        if area < min_area and kept_ids:
            continue
        kept_ids.append(idx)
    if not kept_ids:
        return np.zeros_like(vis)
    keep = np.isin(labels, kept_ids)
    return keep.astype(np.uint8) * 255


def usable(
    mask: np.ndarray,
    spec: RoleSpec,
    character: np.ndarray,
    others: list[LayerMask],
    box: list[float] | None = None,
) -> UsableResult:
    reasons: list[str] = []
    char = character > 0
    char_area = max(1, int(char.sum()))
    cleaned = drop_crumbs(mask, spec, char_area)
    vis = (cleaned > 0) & char
    cleaned = vis.astype(np.uint8) * 255
    area = int(vis.sum())
    frac = area / char_area
    score = 1.0
    if area < 16:
        return UsableResult(False, 0.0, ["empty"], cleaned)
    if frac < spec.min_area_frac:
        reasons.append(f"too_small:{frac:.4f}")
        score -= 0.4
    if frac > spec.max_area_frac:
        reasons.append(f"too_large:{frac:.4f}")
        score -= 0.4
    num, _ = cv2.connectedComponents((cleaned > 0).astype(np.uint8), connectivity=8)
    components = max(0, num - 1)
    if components > spec.max_components:
        reasons.append(f"components:{components}")
        score -= 0.3
    for other in others:
        if other.role not in spec.exclude_roles:
            continue
        if other.source == "placeholder":
            continue
        other_vis = other.visible > 0
        other_area = int(other_vis.sum())
        if other_area == 0:
            continue
        overlap = int((vis & other_vis).sum()) / other_area
        if overlap > spec.max_overlap_frac:
            reasons.append(f"overlap_{other.role}:{overlap:.3f}")
            score -= 0.35
    if box is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        h, w = vis.shape
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 > x0 and y1 > y0:
            box_mask = np.zeros_like(vis)
            box_mask[y0:y1, x0:x1] = True
            box_area = max(1, int(box_mask.sum()))
            coverable = box_mask.copy()
            if spec.cover_in_character or spec.cover_minus_exclude:
                coverable &= char
            if spec.cover_minus_exclude:
                for other in others:
                    if other.role not in spec.exclude_roles:
                        continue
                    coverable &= other.visible == 0
            leftover = int(coverable.sum())
            if leftover >= 16 and (spec.cover_in_character or spec.cover_minus_exclude):
                box_area = leftover
            cover = int((vis & box_mask).sum()) / box_area
            out_frac = 1.0 - (int((vis & box_mask).sum()) / max(1, area))
            if cover < spec.min_box_cover:
                reasons.append(f"box_cover:{cover:.3f}")
                score -= 0.25
            if out_frac > spec.max_out_of_box:
                reasons.append(f"out_of_box:{out_frac:.3f}")
                score -= 0.25
    ok = len(reasons) == 0
    return UsableResult(ok, max(0.0, score), reasons, cleaned)

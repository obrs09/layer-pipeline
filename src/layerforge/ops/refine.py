from __future__ import annotations

import numpy as np

from layerforge.backends.segment.base import LayerMask
from layerforge.ops.morph import morph_open
from layerforge.ops.normalize import foreground_mask
from layerforge.taxonomy import Taxonomy


def refine_masks(
    layers: list[LayerMask],
    taxonomy: Taxonomy,
    min_area: int = 64,
    mutex: bool = True,
    overlay_iou: float = 0.5,
    overlay_contain: float = 0.75,
    morph_open_px: int = 0,
) -> list[LayerMask]:
    kept: list[LayerMask] = []
    for layer in layers:
        area = int((layer.visible > 0).sum())
        if area < min_area:
            continue
        kept.append(layer)
    if not mutex or not kept:
        return kept
    kept = _drop_small_same_role(kept, taxonomy)
    if not kept:
        return kept
    h, w = kept[0].visible.shape
    claimed = np.zeros((h, w), dtype=bool)
    # Specific (small) sprites win over nested larger dumps of the same pixels.
    non_overlay = [layer for layer in kept if not taxonomy.spec(layer.role).overlay]
    overlay = [layer for layer in kept if taxonomy.spec(layer.role).overlay]
    non_overlay.sort(
        key=lambda m: (
            -taxonomy.spec(m.role).cut_priority,
            int((m.visible > 0).sum()),
        )
    )
    for layer in non_overlay:
        pix = layer.visible > 0
        visible = pix & ~claimed
        claimed |= pix
        layer.visible = visible.astype(np.uint8) * 255
        if morph_open_px > 0:
            layer.visible = morph_open(layer.visible, morph_open_px)
            visible = layer.visible > 0
        if layer.rgba is not None:
            rgba = layer.rgba.copy()
            rgba[~visible, 3] = 0
            layer.rgba = rgba
    out = [layer for layer in non_overlay if int((layer.visible > 0).sum()) >= min_area]
    out.extend(_dedupe_overlay(overlay, overlay_iou, overlay_contain))
    return sorted(out, key=lambda m: (taxonomy.spec(m.role).order, -int((m.visible > 0).sum())))


def _drop_small_same_role(layers: list[LayerMask], taxonomy: Taxonomy) -> list[LayerMask]:
    """Drop tiny same-role fragments already covered by a much larger sprite (shoulders inside body)."""
    by_role: dict[str, list[LayerMask]] = {}
    for layer in layers:
        by_role.setdefault(layer.role, []).append(layer)
    drop: set[int] = set()
    for role, items in by_role.items():
        if taxonomy.spec(role).overlay:
            continue
        items = sorted(items, key=lambda m: int((m.visible > 0).sum()), reverse=True)
        if not items:
            continue
        largest = int((items[0].visible > 0).sum())
        for extra in items[1:]:
            if int((extra.visible > 0).sum()) < 0.25 * largest:
                drop.add(id(extra))
    return [layer for layer in layers if id(layer) not in drop]


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = int(np.logical_and(aa, bb).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(aa, bb).sum())
    return inter / union


def _containment(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of the smaller mask that sits inside the other."""
    aa = a > 0
    bb = b > 0
    inter = int(np.logical_and(aa, bb).sum())
    if inter == 0:
        return 0.0
    smaller = min(int(aa.sum()), int(bb.sum()))
    return inter / smaller


def _dedupe_overlay(
    layers: list[LayerMask],
    iou_thresh: float,
    contain_thresh: float = 0.75,
) -> list[LayerMask]:
    """Keep one overlay sprite per role when Imagine crops land on the same pixels."""
    if (iou_thresh <= 0 and contain_thresh <= 0) or len(layers) < 2:
        return layers
    by_role: dict[str, list[LayerMask]] = {}
    for layer in layers:
        by_role.setdefault(layer.role, []).append(layer)
    kept: list[LayerMask] = []
    for items in by_role.values():
        items = sorted(
            items,
            key=lambda m: (int((m.visible > 0).sum()), m.score),
            reverse=True,
        )
        chosen: list[LayerMask] = []
        for cand in items:
            duplicate = False
            for prev in chosen:
                if _mask_iou(cand.visible, prev.visible) >= iou_thresh:
                    duplicate = True
                    break
                if contain_thresh > 0 and _containment(cand.visible, prev.visible) >= contain_thresh:
                    duplicate = True
                    break
            if not duplicate:
                chosen.append(cand)
        kept.extend(chosen)
    return kept


def assign_residual_to_body(
    layers: list[LayerMask],
    source: np.ndarray,
    min_area: int = 64,
    background_luma: int = 250,
    max_frac: float = 0.04,
    domain: np.ndarray | None = None,
) -> list[LayerMask]:
    """Fold tiny unclaimed crumbs into body. Do not dump leftover hair/clothes."""
    if not layers:
        return layers
    fg = foreground_mask(source, background_luma) > 0
    if domain is not None:
        fg = fg & (domain > 0)
    claimed = np.zeros(fg.shape, dtype=bool)
    for layer in layers:
        claimed |= layer.visible > 0
    residual = fg & ~claimed
    leftover = int(residual.sum())
    if leftover < min_area:
        return layers
    fg_area = max(1, int(fg.sum()))
    if leftover / fg_area > max_frac:
        return layers
    bodies = [layer for layer in layers if layer.role == "body"]
    if bodies:
        target = max(bodies, key=lambda m: int((m.visible > 0).sum()))
        vis = (target.visible > 0) | residual
        target.visible = vis.astype(np.uint8) * 255
        target.notes = f"{target.notes}; residual fg folded into body".strip("; ")
        return layers
    layers.append(
        LayerMask(
            role="body",
            label="residual_fg",
            visible=residual.astype(np.uint8) * 255,
            source="silhouette",
            notes="unclaimed foreground after mutex",
        )
    )
    return layers


def unclaimed_in_domain(
    layers: list[LayerMask],
    domain: np.ndarray,
    taxonomy: Taxonomy,
    source: np.ndarray | None = None,
    background_luma: int = 250,
) -> np.ndarray:
    """Character pixels no non-overlay layer owns. Overlay sprites do not count as cover."""
    want = domain > 0
    if source is not None:
        want &= foreground_mask(source, background_luma) > 0
    claimed = np.zeros(want.shape, dtype=bool)
    for layer in layers:
        if taxonomy.spec(layer.role).overlay:
            continue
        claimed |= layer.visible > 0
    return want & ~claimed


def fill_unclaimed_domain(
    layers: list[LayerMask],
    domain: np.ndarray,
    taxonomy: Taxonomy,
    source: np.ndarray | None = None,
    background_luma: int = 250,
    min_area: int = 64,
    max_dist: float = 24.0,
    skip_roles: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[LayerMask], dict]:
    """Every character pixel ends up in a non-overlay layer so the stack has no holes.

    Thin leftovers (never farther than max_dist from a layer) join the nearest
    non-overlay layer, like seam fill. Anything with real interior becomes body:
    unclaimed flesh, hands, or hair the detectors missed. Returns (layers, report).
    skip_roles (eyes/mouth/acc) stay punched out; the overlay sprite covers them.
    """
    import cv2

    residual = unclaimed_in_domain(layers, domain, taxonomy, source, background_luma)
    hole_before = int(residual.sum())
    skip = {str(name) for name in (skip_roles or ())}
    skip.add("acc")
    if skip:
        for layer in layers:
            if layer.role in skip:
                residual &= ~(layer.visible > 0)
    domain_px = int((domain > 0).sum())
    report: dict = {
        "domain_px": domain_px,
        "hole_px_before": hole_before,
        "seam_px": 0,
        "body_px": 0,
        "blobs": [],
    }
    if not residual.any():
        after = unclaimed_in_domain(layers, domain, taxonomy, source, background_luma)
        report["hole_px_after"] = int(after.sum())
        report["hole_frac_after"] = float(after.sum() / max(1, domain_px))
        return layers, report

    candidates = [idx for idx, layer in enumerate(layers) if not taxonomy.spec(layer.role).overlay]
    claimed = np.zeros(residual.shape, dtype=bool)
    for idx in candidates:
        claimed |= layers[idx].visible > 0
    if claimed.any():
        dist_claimed = cv2.distanceTransform((~claimed).astype(np.uint8), cv2.DIST_L2, 5)
    else:
        dist_claimed = np.full(residual.shape, np.inf, dtype=np.float32)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(residual.astype(np.uint8), connectivity=8)
    seam = np.zeros(residual.shape, dtype=bool)
    blob = np.zeros(residual.shape, dtype=bool)
    for k in range(1, n):
        comp = labels == k
        area = int(stats[k, cv2.CC_STAT_AREA])
        depth = float(dist_claimed[comp].max())
        if area >= min_area and depth > float(max_dist):
            blob |= comp
            x, y, w, h = (int(v) for v in stats[k, :4])
            report["blobs"].append({"area": area, "bbox": [x, y, w, h], "depth_px": round(depth, 1)})
        else:
            seam |= comp

    if seam.any():
        if candidates:
            best_d = np.full(residual.shape, np.inf, dtype=np.float32)
            best_i = np.full(residual.shape, -1, dtype=np.int32)
            for idx in candidates:
                inv = np.ones(residual.shape, dtype=np.uint8)
                inv[layers[idx].visible > 0] = 0
                dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
                better = dist < best_d
                best_d[better] = dist[better]
                best_i[better] = idx
            for idx in candidates:
                add = seam & (best_i == idx)
                if not add.any():
                    continue
                layers[idx].visible = ((layers[idx].visible > 0) | add).astype(np.uint8) * 255
                layers[idx].notes = _join(layers[idx].notes, "seam filled")
            report["seam_px"] = int(seam.sum())
        else:
            blob |= seam

    if blob.any():
        bodies = [layer for layer in layers if layer.role == "body"]
        if bodies:
            target = max(bodies, key=lambda m: int((m.visible > 0).sum()))
            target.visible = ((target.visible > 0) | blob).astype(np.uint8) * 255
            target.notes = _join(target.notes, "unclaimed character pixels folded into body")
        else:
            layers.append(
                LayerMask(
                    role="body",
                    label="unclaimed",
                    visible=blob.astype(np.uint8) * 255,
                    source="silhouette",
                    notes="unclaimed character pixels; no body layer was cut",
                )
            )
            layers.sort(key=lambda m: (taxonomy.spec(m.role).order, -int((m.visible > 0).sum())))
        report["body_px"] = int(blob.sum())

    after = unclaimed_in_domain(layers, domain, taxonomy, source, background_luma)
    report["hole_px_after"] = int(after.sum())
    report["hole_frac_after"] = float(after.sum() / max(1, domain_px))
    return layers, report


def _join(notes: str, extra: str) -> str:
    if not notes:
        return extra
    if extra in notes:
        return notes
    return f"{notes}; {extra}"


def assign_unclaimed_seams(
    layers: list[LayerMask],
    source: np.ndarray,
    taxonomy: Taxonomy,
    background_luma: int = 250,
    max_dist: float = 24.0,
    domain: np.ndarray | None = None,
) -> list[LayerMask]:
    """Fill unclaimed fg that sits within max_dist of an existing non-overlay layer.

    Catches punch/mutex gaps. Far leftover blobs stay unassigned.
    Pass `domain` (character mask) so night/sky pixels outside the cut cannot join a layer.
    """
    if not layers or max_dist <= 0:
        return layers
    import cv2

    fg = foreground_mask(source, background_luma) > 0
    if domain is not None:
        fg = fg & (domain > 0)
    claimed = np.zeros(fg.shape, dtype=bool)
    for layer in layers:
        claimed |= layer.visible > 0
    residual = fg & ~claimed
    if int(residual.sum()) == 0:
        return layers
    dist_claimed = cv2.distanceTransform((~claimed).astype(np.uint8), cv2.DIST_L2, 5)
    seam = residual & (dist_claimed <= float(max_dist))
    if int(seam.sum()) == 0:
        return layers
    candidates = [
        idx for idx, layer in enumerate(layers) if not taxonomy.spec(layer.role).overlay
    ]
    if not candidates:
        candidates = list(range(len(layers)))
    best_d = np.full(fg.shape, np.inf, dtype=np.float32)
    best_i = np.full(fg.shape, -1, dtype=np.int32)
    for idx in candidates:
        inv = np.ones(fg.shape, dtype=np.uint8)
        inv[layers[idx].visible > 0] = 0
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
        better = dist < best_d
        best_d[better] = dist[better]
        best_i[better] = idx
    for idx in candidates:
        add = seam & (best_i == idx)
        if int(add.sum()) == 0:
            continue
        vis = (layers[idx].visible > 0) | add
        layers[idx].visible = vis.astype(np.uint8) * 255
        layers[idx].notes = f"{layers[idx].notes}; seam filled".strip("; ")
    return layers

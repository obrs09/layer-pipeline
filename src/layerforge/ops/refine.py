from __future__ import annotations

import numpy as np

from layerforge.backends.segment.base import LayerMask
from layerforge.ops.normalize import foreground_mask
from layerforge.taxonomy import Taxonomy


def refine_masks(
    layers: list[LayerMask],
    taxonomy: Taxonomy,
    min_area: int = 64,
    mutex: bool = True,
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
    non_overlay.sort(key=lambda m: int((m.visible > 0).sum()))
    for layer in non_overlay:
        pix = layer.visible > 0
        visible = pix & ~claimed
        claimed |= pix
        layer.visible = visible.astype(np.uint8) * 255
        if layer.rgba is not None:
            rgba = layer.rgba.copy()
            rgba[~visible, 3] = 0
            layer.rgba = rgba
    out = [layer for layer in non_overlay if int((layer.visible > 0).sum()) >= min_area]
    out.extend(overlay)
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


def assign_residual_to_body(
    layers: list[LayerMask],
    source: np.ndarray,
    min_area: int = 64,
    background_luma: int = 250,
) -> list[LayerMask]:
    """Give unclaimed foreground pixels to body so hair wisps / hems stay on a layer."""
    if not layers:
        return layers
    fg = foreground_mask(source, background_luma) > 0
    claimed = np.zeros(fg.shape, dtype=bool)
    for layer in layers:
        claimed |= layer.visible > 0
    residual = fg & ~claimed
    if int(residual.sum()) < min_area:
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

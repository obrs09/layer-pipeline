from __future__ import annotations

import cv2
import numpy as np

from layerforge.backends.segment.base import LayerMask
from layerforge.ops.normalize import foreground_mask
from layerforge.taxonomy import Taxonomy


def plan_occlusion(
    layers: list[LayerMask],
    source: np.ndarray,
    taxonomy: Taxonomy,
    background_luma: int = 250,
    seam_dilate_px: int = 0,
) -> dict[int, np.ndarray]:
    """Return occluded masks keyed by index into `layers`.

    Holes stay under occluder visibles, and only under layers that composite
    above this one. A hole never lands on a lower layer's visible pixels:
    this layer would paint inpaint over that original and light up
    preview/diff.png. Seam dilation obeys the same bounds.
    """
    if not layers:
        return {}
    h, w = layers[0].visible.shape
    fg = foreground_mask(source, background_luma) > 0
    orders = [taxonomy.spec(layer.role).order for layer in layers]
    occluded: dict[int, np.ndarray] = {}
    kernel = np.ones((3, 3), np.uint8)
    for i, layer in enumerate(layers):
        spec = taxonomy.spec(layer.role)
        if not spec.complete or spec.expand_px <= 0:
            occluded[i] = np.zeros((h, w), dtype=np.uint8)
            continue
        want = cv2.dilate(layer.visible, kernel, iterations=int(spec.expand_px)) > 0
        occluders = np.zeros((h, w), dtype=bool)
        lower = np.zeros((h, w), dtype=bool)
        for j, other in enumerate(layers):
            if j == i:
                continue
            if orders[j] > spec.order and other.role in spec.occluded_by:
                occluders |= other.visible > 0
            elif orders[j] < spec.order:
                lower |= other.visible > 0
        allowed = occluders & ~(layer.visible > 0) & fg & ~lower
        hole = want & allowed
        if seam_dilate_px > 0 and hole.any():
            dilated = cv2.dilate(
                hole.astype(np.uint8) * 255,
                kernel,
                iterations=int(seam_dilate_px),
            )
            hole = (dilated > 0) & allowed
        occluded[i] = hole.astype(np.uint8) * 255
    return occluded


def occluded_over_lower_visible(
    layers: list[LayerMask],
    occluded: dict[int, np.ndarray],
    taxonomy: Taxonomy,
) -> int:
    """Pixels where a layer's hole sits on a lower layer's visible. Must be 0."""
    orders = [taxonomy.spec(layer.role).order for layer in layers]
    total = 0
    for i, hole in occluded.items():
        if hole is None or not (hole > 0).any():
            continue
        for j, other in enumerate(layers):
            if orders[j] < orders[i]:
                total += int(((hole > 0) & (other.visible > 0)).sum())
    return total

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

    Holes stay under occluder visibles. Seam dilation must not spill into a
    neighbor's visible pixels — those would composite on top of original
    source and light up preview/diff.png.
    """
    if not layers:
        return {}
    h, w = layers[0].visible.shape
    fg = foreground_mask(source, background_luma) > 0
    role_union: dict[str, np.ndarray] = {}
    for layer in layers:
        role_union.setdefault(layer.role, np.zeros((h, w), dtype=bool))
        role_union[layer.role] |= layer.visible > 0
    occluded: dict[int, np.ndarray] = {}
    kernel = np.ones((3, 3), np.uint8)
    for i, layer in enumerate(layers):
        spec = taxonomy.spec(layer.role)
        if not spec.complete or spec.expand_px <= 0:
            occluded[i] = np.zeros((h, w), dtype=np.uint8)
            continue
        want = cv2.dilate(layer.visible, kernel, iterations=int(spec.expand_px)) > 0
        occluders = np.zeros((h, w), dtype=bool)
        for other_role in spec.occluded_by:
            if other_role in role_union:
                occluders |= role_union[other_role]
        hole = want & occluders & ~(layer.visible > 0) & fg
        if seam_dilate_px > 0 and hole.any():
            dilated = cv2.dilate(
                hole.astype(np.uint8) * 255,
                kernel,
                iterations=int(seam_dilate_px),
            )
            hole = (dilated > 0) & occluders & ~(layer.visible > 0) & fg
        occluded[i] = hole.astype(np.uint8) * 255
    return occluded

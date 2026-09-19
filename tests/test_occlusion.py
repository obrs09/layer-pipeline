from __future__ import annotations

import numpy as np

from layerforge.backends.segment.base import LayerMask
from layerforge.image_io import alpha_over
from layerforge.ops.occlusion import plan_occlusion
from layerforge.ops.reproject import reproject
from layerforge.taxonomy import load_taxonomy


def _layer(role: str, visible: np.ndarray) -> LayerMask:
    return LayerMask(role=role, label=role, visible=visible, source="sam")


def _source(h: int = 48, w: int = 48) -> np.ndarray:
    image = np.zeros((h, w, 4), dtype=np.uint8)
    image[..., :3] = 10
    image[..., 3] = 255
    return image


def test_seam_stays_under_occluders():
    source = _source()
    hair = np.zeros(source.shape[:2], dtype=np.uint8)
    hair[:, :22] = 255
    face = np.zeros(source.shape[:2], dtype=np.uint8)
    face[8:40, 18:34] = 255
    layers = [_layer("hair_back", hair), _layer("face", face)]
    occ = plan_occlusion(layers, source, load_taxonomy(), seam_dilate_px=4)
    hair_occ = occ[0] > 0
    assert hair_occ.any()
    assert not np.any(hair_occ & (hair > 0))
    assert not np.any(hair_occ & ~(face > 0))


def test_later_layer_seam_does_not_cover_earlier_visible_in_composite():
    """Body completing under a badge must not seam-dilate into hair visibles.

    That is the sam_flat1 diff.png outline: later occ composited over earlier
    original pixels.
    """
    source = _source()
    hair = np.zeros(source.shape[:2], dtype=np.uint8)
    hair[:, :24] = 255
    body = np.zeros(source.shape[:2], dtype=np.uint8)
    body[:, 24:] = 255
    badge = np.zeros(source.shape[:2], dtype=np.uint8)
    badge[16:32, 20:28] = 255
    layers = [
        _layer("hair_back", hair),
        _layer("body", body),
        _layer("acc", badge),
    ]
    occ = plan_occlusion(layers, source, load_taxonomy(), seam_dilate_px=3)
    body_occ = occ[1] > 0
    assert body_occ.any()
    assert not np.any(body_occ & ~(badge > 0))
    # Seam used to spill into hair visibles that are not the badge.

    inpainted = np.zeros_like(source)
    inpainted[..., :3] = 200
    inpainted[..., 3] = 255
    composite = np.zeros_like(source)
    for idx, layer in enumerate(layers):
        filled = reproject(inpainted, source, layer.visible, occ[idx])
        composite = alpha_over(composite, filled)
    vis = (hair > 0) | (body > 0) | (badge > 0)
    assert np.all(composite[vis, :3] == source[vis, :3])

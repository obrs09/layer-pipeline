from __future__ import annotations

import numpy as np

from layerforge.backends.segment.base import LayerMask
from layerforge.image_io import alpha_over
from layerforge.ops.occlusion import occluded_over_lower_visible, plan_occlusion
from layerforge.ops.reproject import reproject
from layerforge.taxonomy import RoleSpec, Taxonomy, load_taxonomy


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
    hair[:, :20] = 255
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
    assert not np.any(body_occ & (hair > 0))
    assert occluded_over_lower_visible(layers, occ, load_taxonomy()) == 0

    inpainted = np.zeros_like(source)
    inpainted[..., :3] = 200
    inpainted[..., 3] = 255
    composite = np.zeros_like(source)
    for idx, layer in enumerate(layers):
        filled = reproject(inpainted, source, layer.visible, occ[idx])
        composite = alpha_over(composite, filled)
    vis = (hair > 0) | (body > 0) | (badge > 0)
    assert np.all(composite[vis, :3] == source[vis, :3])


def test_clothes_hole_never_covers_lower_arm_visible():
    """9083053e: clothes completed under arm_r, then composited above it."""
    source = _source(64, 64)
    body = np.zeros(source.shape[:2], dtype=np.uint8)
    body[40:64, :] = 255
    arm = np.zeros(source.shape[:2], dtype=np.uint8)
    arm[10:40, 28:36] = 255
    clothes = np.zeros(source.shape[:2], dtype=np.uint8)
    clothes[10:40, :28] = 255
    clothes[10:40, 36:] = 255
    taxonomy = load_taxonomy()
    layers = [_layer("body", body), _layer("arm_r", arm), _layer("clothes", clothes)]
    occ = plan_occlusion(layers, source, taxonomy, seam_dilate_px=2)
    clothes_occ = occ[2] > 0
    assert not np.any(clothes_occ & (arm > 0))
    assert not np.any(clothes_occ & (body > 0))
    arm_occ = occ[1] > 0
    assert arm_occ.any()
    assert not np.any(arm_occ & (body > 0))
    assert occluded_over_lower_visible(layers, occ, taxonomy) == 0

    inpainted = np.zeros_like(source)
    inpainted[..., :3] = 200
    inpainted[..., 3] = 255
    composite = np.zeros_like(source)
    for idx, layer in enumerate(layers):
        filled = reproject(inpainted, source, layer.visible, occ[idx])
        composite = alpha_over(composite, filled)
    vis = (body > 0) | (arm > 0) | (clothes > 0)
    assert np.all(composite[vis, :3] == source[vis, :3])


def _spec(name: str, order: int, occluded_by: tuple[str, ...], overlay: bool = False) -> RoleSpec:
    return RoleSpec(
        name=name,
        order=order,
        complete=not overlay,
        overlay=overlay,
        expand_px=0 if overlay else 6,
        occluded_by=occluded_by,
        aliases=(),
    )


def test_plan_ignores_occluders_listed_below_the_layer():
    """Even a bad taxonomy cannot make a layer paint over the one beneath it."""
    taxonomy = Taxonomy(
        roles={
            "under": _spec("under", 10, ("mid", "over")),
            "mid": _spec("mid", 20, ("under", "over")),
            "over": _spec("over", 30, ()),
            "pin": _spec("pin", 90, (), overlay=True),
        },
        pair_roles={},
        pair_aliases={},
        alias_to_role={},
        pair_queries={},
    )
    source = _source(40, 60)
    under = np.zeros(source.shape[:2], dtype=np.uint8)
    under[:, :20] = 255
    mid = np.zeros(source.shape[:2], dtype=np.uint8)
    mid[:, 20:40] = 255
    over = np.zeros(source.shape[:2], dtype=np.uint8)
    over[:, 40:] = 255
    pin = np.zeros(source.shape[:2], dtype=np.uint8)
    pin[10:30, 15:45] = 255
    layers = [_layer("under", under), _layer("mid", mid), _layer("over", over), _layer("pin", pin)]
    occ = plan_occlusion(layers, source, taxonomy, seam_dilate_px=2)
    assert (occ[0] > 0).any()
    assert (occ[1] > 0).any()
    assert not np.any((occ[1] > 0) & (under > 0))
    assert occluded_over_lower_visible(layers, occ, taxonomy) == 0
    for i, hole in occ.items():
        for j, other in enumerate(layers):
            if taxonomy.spec(other.role).order < taxonomy.spec(layers[i].role).order:
                assert not np.any((hole > 0) & (other.visible > 0))

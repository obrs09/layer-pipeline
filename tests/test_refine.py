from __future__ import annotations

import numpy as np

from layerforge.backends.segment.base import LayerMask
from layerforge.ops.refine import (
    assign_residual_to_body,
    assign_unclaimed_seams,
    fill_unclaimed_domain,
    refine_masks,
    unclaimed_in_domain,
)
from layerforge.taxonomy import load_taxonomy


def _mask(role: str, label: str, y0: int, x0: int, h: int = 8, w: int = 8) -> LayerMask:
    visible = np.zeros((32, 40), dtype=np.uint8)
    visible[y0 : y0 + h, x0 : x0 + w] = 255
    return LayerMask(role=role, label=label, visible=visible, source="imagine_part")


def test_duplicate_acc_same_pixels_keeps_one():
    a = _mask("acc", "gold-jewelry", 4, 4)
    b = _mask("acc", "purple-gemstones", 4, 4)
    out = refine_masks([a, b], load_taxonomy())
    acc = [layer for layer in out if layer.role == "acc"]
    assert len(acc) == 1


def test_separate_acc_keeps_both():
    left = _mask("acc", "white-high-heels", 20, 4)
    right = _mask("acc", "white-high-heels", 20, 24)
    out = refine_masks([left, right], load_taxonomy())
    acc = [layer for layer in out if layer.role == "acc"]
    assert len(acc) == 2


def test_acc_inside_larger_acc_is_dropped():
    big = _mask("acc", "gold-embroidery", 4, 4, h=16, w=16)
    small = _mask("acc", "purple-gemstones", 6, 6, h=6, w=6)
    out = refine_masks([big, small], load_taxonomy())
    acc = [layer for layer in out if layer.role == "acc"]
    assert len(acc) == 1
    assert int((acc[0].visible > 0).sum()) == 16 * 16


def test_eyes_are_not_merged_across_roles():
    left = _mask("eye_l", "pink-eyes", 4, 4)
    right = _mask("eye_r", "pink-eyes", 4, 24)
    out = refine_masks([left, right], load_taxonomy())
    roles = {layer.role for layer in out}
    assert roles == {"eye_l", "eye_r"}


def test_missing_body_is_not_replaced_by_residual():
    clothes = _mask("clothes", "clothes", 4, 4, h=20, w=20)
    source = np.full((32, 40, 3), 10, dtype=np.uint8)
    source[0, :] = 255
    source[-1, :] = 255
    source[:, 0] = 255
    source[:, -1] = 255
    out = assign_residual_to_body([clothes], source, min_area=8, max_frac=0.9)
    assert all(layer.role != "body" for layer in out)


def test_torso_skin_survives_morph_open():
    body = _mask("body", "body", 8, 8, h=2, w=20)
    body.notes = "figure minus head and acc; torso skin"
    face = _mask("face", "face", 0, 0, h=4, w=4)
    out = refine_masks([body, face], load_taxonomy(), morph_open_px=3, min_area=64)
    kept = [layer for layer in out if layer.role == "body"]
    assert kept
    assert "torso skin" in kept[0].notes
    assert "unclaimed foreground" not in kept[0].notes

    body = _mask("body", "body", 20, 4, h=8, w=8)
    source = np.full((32, 40, 3), 255, dtype=np.uint8)
    source[2:30, 2:38] = 10
    out = assign_residual_to_body([body], source, min_area=64, max_frac=0.04)
    assert int((out[0].visible > 0).sum()) == 8 * 8
    assert "folded" not in out[0].notes


def test_small_unclaimed_fg_folds_into_body():
    body = _mask("body", "body", 8, 8, h=16, w=16)
    source = np.full((32, 40, 3), 255, dtype=np.uint8)
    source[8:24, 8:24] = 10
    source[8:12, 24:28] = 10
    out = assign_residual_to_body([body], source, min_area=8, max_frac=0.2)
    assert int((out[0].visible > 0).sum()) > 16 * 16
    assert "folded" in out[0].notes


def test_unclaimed_seam_goes_to_nearest_layer():
    hair = _mask("hair_back", "hair", 2, 4, h=10, w=20)
    body = _mask("body", "body", 16, 4, h=10, w=20)
    source = np.full((32, 40, 3), 255, dtype=np.uint8)
    source[2:26, 4:24] = 10
    out = assign_unclaimed_seams([hair, body], source, load_taxonomy(), max_dist=8)
    claimed = (out[0].visible > 0) | (out[1].visible > 0)
    assert claimed[14, 10]
    assert "seam filled" in out[0].notes or "seam filled" in out[1].notes


def test_far_unclaimed_blob_not_seam_filled():
    body = _mask("body", "body", 20, 4, h=8, w=8)
    source = np.full((32, 40, 3), 255, dtype=np.uint8)
    source[20:28, 4:12] = 10
    source[2:8, 28:36] = 10
    out = assign_unclaimed_seams([body], source, load_taxonomy(), max_dist=4)
    assert int((out[0].visible[2:8, 28:36] > 0).sum()) == 0


def test_seam_fill_stays_inside_character_domain():
    body = _mask("body", "body", 8, 8, h=8, w=8)
    source = np.full((32, 40, 3), 10, dtype=np.uint8)
    domain = np.zeros((32, 40), dtype=np.uint8)
    domain[8:16, 8:16] = 255
    out = assign_unclaimed_seams(
        [body],
        source,
        load_taxonomy(),
        max_dist=24,
        domain=domain,
    )
    extra = (out[0].visible > 0) & (domain == 0)
    assert int(extra.sum()) == 0


def test_mutex_face_wins_over_hair_back():
    face = _mask("face", "face", 4, 4, h=12, w=12)
    hair = _mask("hair_back", "hair", 4, 4, h=20, w=20)
    out = refine_masks([hair, face], load_taxonomy())
    by_role = {layer.role: layer for layer in out}
    assert int((by_role["face"].visible > 0).sum()) == 12 * 12
    assert int((by_role["hair_back"].visible > 0).sum()) == 20 * 20 - 12 * 12


def test_mutex_hair_front_keeps_pixels_covering_face():
    """Front hair is a subset of the face; it must not be emptied by face claiming first."""
    face = _mask("face", "face", 4, 4, h=16, w=16)
    bangs = _mask("hair_front", "hair", 4, 4, h=8, w=16)
    hair = _mask("hair_back", "hair", 4, 4, h=24, w=20)
    out = refine_masks([hair, face, bangs], load_taxonomy())
    by_role = {layer.role: layer for layer in out}
    assert "hair_front" in by_role
    assert int((by_role["hair_front"].visible > 0).sum()) == 8 * 16
    assert int((by_role["face"].visible[4:12, 4:20] > 0).sum()) == 0
    assert int((by_role["face"].visible > 0).sum()) == 16 * 16 - 8 * 16


def _character(h: int = 96, w: int = 80) -> np.ndarray:
    domain = np.zeros((h, w), dtype=np.uint8)
    domain[4:92, 8:72] = 255
    return domain


def _hole_frac(layers, domain, source=None) -> float:
    hole = unclaimed_in_domain(layers, domain, load_taxonomy(), source)
    return float(hole.sum() / max(1, int((domain > 0).sum())))


def test_unclaimed_legs_become_body():
    """a1639e70: body kept a waist band; the legs below were in no layer."""
    domain = _character()
    source = np.full((96, 80, 3), 40, dtype=np.uint8)
    clothes = LayerMask(role="clothes", label="clothes", visible=np.zeros_like(domain), source="sam")
    clothes.visible[4:50, 8:72] = 255
    body = LayerMask(role="body", label="body", visible=np.zeros_like(domain), source="sam")
    body.visible[50:58, 8:72] = 255
    assert _hole_frac([clothes, body], domain) > 0.3
    layers, report = fill_unclaimed_domain([clothes, body], domain, load_taxonomy(), source, max_dist=8)
    assert _hole_frac(layers, domain) == 0.0
    assert report["hole_px_after"] == 0
    assert report["body_px"] > 0
    assert report["blobs"]
    by_role = {layer.role: layer for layer in layers}
    assert int(by_role["body"].visible[80, 40]) == 255
    assert int(by_role["clothes"].visible[80, 40]) == 0
    assert "folded into body" in by_role["body"].notes


def test_unclaimed_thin_gap_joins_nearest_layer_not_body():
    domain = _character()
    source = np.full((96, 80, 3), 40, dtype=np.uint8)
    hair = LayerMask(role="hair_front", label="hair", visible=np.zeros_like(domain), source="sam")
    hair.visible[4:30, 8:72] = 255
    body = LayerMask(role="body", label="body", visible=np.zeros_like(domain), source="sam")
    body.visible[34:92, 8:72] = 255
    layers, report = fill_unclaimed_domain([hair, body], domain, load_taxonomy(), source, max_dist=8)
    assert report["hole_px_after"] == 0
    assert report["body_px"] == 0
    assert report["seam_px"] == 4 * 64
    assert _hole_frac(layers, domain) == 0.0
    assert len(layers) == 2
    assert "seam filled" in hair.notes or "seam filled" in body.notes


def test_unclaimed_blob_creates_body_when_none_was_cut():
    """64010c19: hair the detectors missed must not vanish from the stack."""
    domain = _character()
    source = np.full((96, 80, 3), 40, dtype=np.uint8)
    clothes = LayerMask(role="clothes", label="clothes", visible=np.zeros_like(domain), source="sam")
    clothes.visible[40:92, 8:72] = 255
    face = LayerMask(role="face", label="face", visible=np.zeros_like(domain), source="sam")
    face.visible[16:40, 24:56] = 255
    layers, report = fill_unclaimed_domain([clothes, face], domain, load_taxonomy(), source, max_dist=8)
    assert _hole_frac(layers, domain) == 0.0
    roles = [layer.role for layer in layers]
    assert roles == ["body", "clothes", "face"]
    body = layers[0]
    assert body.source == "silhouette"
    assert int(body.visible[8, 40]) == 255
    assert not np.any((body.visible > 0) & (domain == 0))


def test_acc_overlay_is_not_folded_into_body():
    domain = _character()
    source = np.full((96, 80, 3), 40, dtype=np.uint8)
    body = LayerMask(role="body", label="body", visible=domain.copy(), source="sam")
    body.visible[40:60, 20:60] = 0
    acc = LayerMask(role="acc", label="acc", visible=np.zeros_like(domain), source="sam")
    acc.visible[40:60, 20:60] = 255
    layers, report = fill_unclaimed_domain([body, acc], domain, load_taxonomy(), source, max_dist=4)
    body_out = next(layer for layer in layers if layer.role == "body")
    assert int(body_out.visible[50, 40]) == 0
    assert int(((body_out.visible > 0) & (acc.visible > 0)).sum()) == 0
    assert report["hole_px_after"] == int((acc.visible > 0).sum())


def test_fill_skips_eye_overlay_holes():
    domain = _character()
    source = np.full((96, 80, 3), 40, dtype=np.uint8)
    face = LayerMask(role="face", label="face", visible=domain.copy(), source="sam")
    face.visible[40:52, 28:52] = 0
    eye = LayerMask(role="eye_l", label="eye_l", visible=np.zeros_like(domain), source="sam")
    eye.visible[40:52, 28:52] = 255
    layers, report = fill_unclaimed_domain(
        [face, eye],
        domain,
        load_taxonomy(),
        source,
        max_dist=4,
        skip_roles=("eye_l", "eye_r", "mouth"),
    )
    face_out = next(layer for layer in layers if layer.role == "face")
    assert int(face_out.visible[46, 40]) == 0
    assert report["hole_px_after"] > 0


def test_fill_never_leaves_the_character_domain():
    domain = _character()
    source = np.full((96, 80, 3), 40, dtype=np.uint8)
    body = LayerMask(role="body", label="body", visible=np.zeros_like(domain), source="sam")
    body.visible[40:60, 20:60] = 255
    layers, _ = fill_unclaimed_domain([body], domain, load_taxonomy(), source, max_dist=8)
    for layer in layers:
        assert not np.any((layer.visible > 0) & (domain == 0))
    assert _hole_frac(layers, domain) == 0.0


def test_morph_open_drops_one_pixel_rim():
    hair = _mask("hair_back", "hair", 4, 4, h=12, w=16)
    body = _mask("body", "body", 16, 4, h=12, w=16)
    body.visible[2, 4:20] = 255
    out = refine_masks([hair, body], load_taxonomy(), morph_open_px=2)
    by_role = {layer.role: layer for layer in out}
    assert int(by_role["body"].visible[2, 10]) == 0
    assert int(by_role["body"].visible[20, 10]) == 255

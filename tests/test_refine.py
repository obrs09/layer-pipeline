from __future__ import annotations

import numpy as np

from layerforge.backends.segment.base import LayerMask
from layerforge.ops.refine import assign_residual_to_body, assign_unclaimed_seams, refine_masks
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


def test_large_unclaimed_fg_not_dumped_on_body():
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


def test_morph_open_drops_one_pixel_rim():
    hair = _mask("hair_back", "hair", 4, 4, h=12, w=16)
    body = _mask("body", "body", 16, 4, h=12, w=16)
    body.visible[2, 4:20] = 255
    out = refine_masks([hair, body], load_taxonomy(), morph_open_px=2)
    by_role = {layer.role: layer for layer in out}
    assert int(by_role["body"].visible[2, 10]) == 0
    assert int(by_role["body"].visible[20, 10]) == 255

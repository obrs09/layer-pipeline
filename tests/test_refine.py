from __future__ import annotations

import numpy as np

from layerforge.backends.segment.base import LayerMask
from layerforge.ops.refine import refine_masks
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

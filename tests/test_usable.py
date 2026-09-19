from __future__ import annotations

import numpy as np

from layerforge.backends.segment.base import LayerMask
from layerforge.ops.usable import drop_crumbs, usable
from layerforge.taxonomy import load_taxonomy


def _spec(role: str):
    return load_taxonomy().spec(role)


def test_eye_too_large_is_unusable():
    spec = _spec("eye_l")
    character = np.zeros((64, 64), dtype=np.uint8)
    character[8:56, 8:56] = 255
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:50, 8:50] = 255
    result = usable(mask, spec, character, [])
    assert not result.ok
    assert any(r.startswith("too_large") for r in result.reasons)


def test_eye_overlap_face_too_much_fails():
    spec = _spec("eye_l")
    character = np.ones((40, 40), dtype=np.uint8) * 255
    eye = np.zeros((40, 40), dtype=np.uint8)
    eye[5:30, 5:30] = 255
    face = LayerMask(
        role="face",
        label="face",
        visible=np.ones((40, 40), dtype=np.uint8) * 255,
        source="sam",
    )
    result = usable(eye, spec, character, [face])
    assert not result.ok
    assert any("overlap_face" in r for r in result.reasons)


def test_hair_drops_crumbs_keeps_main():
    spec = _spec("hair_back")
    character = np.ones((64, 64), dtype=np.uint8) * 255
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[4:30, 4:40] = 255
    mask[60, 60] = 255
    cleaned = drop_crumbs(mask, spec, int(character.sum()))
    assert int(cleaned[60, 60]) == 0
    assert int((cleaned > 0).sum()) > 100

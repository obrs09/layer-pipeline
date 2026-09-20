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


def test_box_cover_counts_only_the_box_inside_the_character():
    """A held phone box hangs over the table; the mask can only fill the character part."""
    spec = _spec("acc")
    character = np.zeros((100, 100), dtype=np.uint8)
    character[:, 50:] = 255
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 50:70] = 255  # fills the in-character half of the box
    box = [30.0, 40.0, 70.0, 60.0]
    result = usable(mask, spec, character, [], box)
    assert not any(r.startswith("box_cover") for r in result.reasons)
    thin = np.zeros((100, 100), dtype=np.uint8)
    thin[40:60, 50:54] = 255
    result = usable(thin, spec, character, [], box)
    assert any(r.startswith("box_cover") for r in result.reasons)


def test_face_box_cover_still_uses_the_whole_box():
    """Body parts sit inside the silhouette; a loose face box must not pass on the normalized cover."""
    spec = _spec("face")
    assert not spec.cover_in_character
    character = np.zeros((100, 100), dtype=np.uint8)
    character[:, 50:] = 255
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 50:60] = 255  # 25% of the box, 50% of the in-character half
    result = usable(mask, spec, character, [], [30.0, 40.0, 70.0, 60.0])
    assert any(r.startswith("box_cover") for r in result.reasons)


def test_placeholder_face_does_not_fail_hair_overlap():
    spec = _spec("hair_back")
    character = np.ones((40, 40), dtype=np.uint8) * 255
    hair = np.zeros((40, 40), dtype=np.uint8)
    hair[2:18, 8:32] = 255
    face = LayerMask(
        role="face",
        label="face",
        visible=np.zeros((40, 40), dtype=np.uint8),
        source="placeholder",
    )
    face.visible[8:28, 10:30] = 255
    result = usable(hair, spec, character, [face], [6.0, 2.0, 34.0, 20.0])
    assert not any("overlap_face" in r for r in result.reasons)


def test_hair_box_cover_uses_box_minus_face_and_clothes():
    spec = _spec("hair_back")
    assert spec.cover_minus_exclude
    character = np.ones((80, 80), dtype=np.uint8) * 255
    clothes = LayerMask(
        role="clothes",
        label="clothes",
        visible=np.zeros((80, 80), dtype=np.uint8),
        source="sam",
    )
    clothes.visible[40:80, :] = 255
    face = LayerMask(
        role="face",
        label="face",
        visible=np.zeros((80, 80), dtype=np.uint8),
        source="placeholder",
    )
    face.visible[8:32, 20:60] = 255
    hair = np.zeros((80, 80), dtype=np.uint8)
    hair[0:8, :] = 255
    hair[8:20, 0:20] = 255
    hair[8:20, 60:80] = 255
    box = [0.0, 0.0, 80.0, 80.0]
    result = usable(hair, spec, character, [clothes, face], box)
    assert not any(r.startswith("box_cover") for r in result.reasons)
    without = usable(hair, spec, character, [], box)
    assert any(r.startswith("box_cover") for r in without.reasons)


def test_hair_drops_crumbs_keeps_main():
    spec = _spec("hair_back")
    character = np.ones((64, 64), dtype=np.uint8) * 255
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[4:30, 4:40] = 255
    mask[60, 60] = 255
    cleaned = drop_crumbs(mask, spec, int(character.sum()))
    assert int(cleaned[60, 60]) == 0
    assert int((cleaned > 0).sum()) > 100

from __future__ import annotations

import numpy as np

from layerforge.backends.segment.base import LayerMask
from layerforge.ops.hair_hint import fill_box, hair_positive, seed_region


def test_seed_region_punches_face_box():
    character = np.ones((40, 40), dtype=np.uint8) * 255
    region = seed_region(character, [4, 2, 36, 36], face_box=[10, 12, 30, 32], face_dilate_px=0)
    assert int(region[20, 20]) == 0
    assert int(region[6, 20]) == 255


def test_hair_positive_is_outside_face():
    image = np.full((40, 40, 3), 80, dtype=np.uint8)
    image[2:10, 8:32] = (50, 50, 190)
    image[14:32, 12:28] = (200, 150, 140)
    character = np.ones((40, 40), dtype=np.uint8) * 255
    face = LayerMask(
        role="face",
        label="face",
        visible=fill_box((40, 40), [12.0, 14.0, 28.0, 32.0]),
        source="placeholder",
    )
    point = hair_positive(image, character, [8.0, 2.0, 32.0, 34.0], [face])
    assert point is not None
    assert point[1] < 14.0
    assert not (12.0 <= point[0] <= 28.0 and 14.0 <= point[1] <= 32.0)

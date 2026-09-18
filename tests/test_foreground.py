from __future__ import annotations

import numpy as np

from layerforge.ops.normalize import foreground_mask


def test_interior_white_is_not_background():
    img = np.full((32, 32, 3), 255, dtype=np.uint8)
    img[6:26, 6:26] = (20, 20, 20)
    img[8:24, 8:24] = 255
    img[12:20, 12:20] = (10, 10, 200)
    mask = foreground_mask(img, background_luma=250)
    assert mask[0, 0] == 0
    assert mask[16, 16] == 255
    assert mask[10, 10] == 255

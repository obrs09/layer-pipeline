from __future__ import annotations

import numpy as np

from layerforge.ops.reproject import reproject


def test_reproject_keeps_visible_source_pixels():
    source = np.zeros((8, 8, 4), dtype=np.uint8)
    source[..., :3] = 10
    source[..., 3] = 255
    inpainted = np.zeros_like(source)
    inpainted[..., :3] = 200
    inpainted[..., 3] = 255
    visible = np.zeros((8, 8), dtype=np.uint8)
    visible[:4, :4] = 255
    occluded = np.zeros((8, 8), dtype=np.uint8)
    occluded[4:6, 4:6] = 255
    out = reproject(inpainted, source, visible, occluded)
    assert np.all(out[:4, :4, :3] == 10)
    assert np.all(out[4:6, 4:6, :3] == 200)
    assert np.all(out[6:, 6:, 3] == 0)

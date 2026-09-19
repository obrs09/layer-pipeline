from __future__ import annotations

import numpy as np

from layerforge.backends.detect.wdtagger import _preprocess


def test_wd_preprocess_nhwc_white_pad():
    rgb = np.full((10, 20, 3), 80, dtype=np.uint8)
    blob = _preprocess(rgb, 448)
    assert blob.shape == (1, 448, 448, 3)
    assert blob.dtype == np.float32
    assert blob[0, 0, 0, 0] == 255.0

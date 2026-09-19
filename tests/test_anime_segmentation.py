from __future__ import annotations

import numpy as np

from layerforge.backends.detect.anime_segmentation import (
    AniSegCut,
    AnimeSegmentationCut,
    pack_square,
    unpack_square,
)
from layerforge.backends.registry import _build_character


def test_pack_square_center_pads_wide_image():
    rgb = np.ones((10, 40, 3), dtype=np.uint8) * 255
    blob, meta = pack_square(rgb, 32)
    _h0, _w0, nh, nw, top, left = meta
    assert blob.shape == (1, 3, 32, 32)
    assert (nh, nw, top, left) == (8, 32, 12, 0)
    assert float(blob[0, 0, 0, 0]) == 0.0
    assert float(blob[0, 0, 12, 0]) == 1.0
    assert float(blob[0, 0, 19, 31]) == 1.0
    assert float(blob[0, 0, 20, 0]) == 0.0


def test_pack_square_center_pads_tall_image():
    rgb = np.ones((40, 10, 3), dtype=np.uint8) * 255
    blob, meta = pack_square(rgb, 32)
    _h0, _w0, nh, nw, top, left = meta
    assert (nh, nw, top, left) == (32, 8, 0, 12)
    assert float(blob[0, 0, 0, 0]) == 0.0
    assert float(blob[0, 0, 0, 12]) == 1.0


def test_unpack_square_restores_shape():
    rgb = np.zeros((50, 80, 3), dtype=np.uint8)
    rgb[10:40, 20:60] = 200
    blob, meta = pack_square(rgb, 64)
    out = unpack_square(blob[0, 0], meta)
    assert out.shape == (50, 80)
    assert out[25, 40] > 0.5
    assert out[0, 0] < 0.1


def test_character_backend_from_config():
    cut = _build_character({"cascade": {"character": "anime_segmentation"}}, allow_luma=True)
    assert isinstance(cut, AnimeSegmentationCut)
    assert cut.name == "anime_segmentation"
    alias = _build_character({"cascade": {"character": "aniseg"}}, allow_luma=True)
    assert isinstance(alias, AniSegCut)
    assert alias.name == "aniseg"


def test_unknown_character_backend():
    try:
        _build_character({"cascade": {"character": "rmbg"}}, allow_luma=True)
    except ValueError as exc:
        assert "rmbg" in str(exc)
    else:
        raise AssertionError("expected ValueError")

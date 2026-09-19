from __future__ import annotations

import numpy as np

from layerforge.backends.inpaint.router import RoutedInpaint
from layerforge.ops.inpaint_region import crop_to_mask, paste_crop


class _Fake:
    def __init__(self, name: str, value: int) -> None:
        self.name = name
        self.value = value
        self.calls = 0

    def inpaint(self, image, mask, prompt):
        self.calls += 1
        out = image.copy()
        out[mask > 0, :3] = self.value
        return out


def test_crop_to_mask_pads_hole():
    image = np.zeros((20, 20, 4), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[8:12, 8:12] = 255
    crop, crop_mask, box = crop_to_mask(image, mask, pad=2)
    y0, y1, x0, x1 = box
    assert (y0, x0) == (6, 6)
    assert crop.shape[0] == y1 - y0
    assert int((crop_mask > 0).sum()) == 16
    pasted = paste_crop(image, crop, box)
    assert pasted.shape == image.shape


def test_router_small_hole_uses_lama_fake():
    small = _Fake("lama", 10)
    large = _Fake("sd15.anime", 200)
    router = RoutedInpaint({"inpaint": {"small_hole_max_px": 50}}, small=small, large=large)
    image = np.zeros((16, 16, 4), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:7, 4:7] = 255
    out = router.inpaint(image, mask, "anime body")
    assert small.calls == 1
    assert large.calls == 0
    assert router.last_engine == "lama"
    assert np.all(out[mask > 0, 0] == 10)


def test_router_large_hole_uses_sd_fake():
    small = _Fake("lama", 10)
    large = _Fake("sd15.anime", 200)
    router = RoutedInpaint({"inpaint": {"small_hole_max_px": 8}}, small=small, large=large)
    image = np.zeros((16, 16, 4), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:12, 2:12] = 255
    out = router.inpaint(image, mask, "anime hair")
    assert large.calls == 1
    assert small.calls == 0
    assert router.last_engine == "sd15.anime"
    assert np.all(out[mask > 0, 0] == 200)


def test_router_falls_back_to_telea_when_lama_raises():
    class Boom:
        name = "lama"

        def inpaint(self, image, mask, prompt):
            raise RuntimeError("no lama")

    telea = _Fake("opencv.telea", 33)
    router = RoutedInpaint(
        {"inpaint": {"small_hole_max_px": 500}},
        small=Boom(),
        large=_Fake("sd15.anime", 1),
        fallback=telea,
    )
    image = np.zeros((8, 8, 4), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 2:4] = 255
    out = router.inpaint(image, mask, "x")
    assert telea.calls == 1
    assert router.last_engine == "opencv.telea"
    assert np.all(out[mask > 0, 0] == 33)

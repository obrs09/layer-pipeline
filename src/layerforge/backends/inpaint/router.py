from __future__ import annotations

import numpy as np

from layerforge.backends.inpaint.lama import LamaInpaint
from layerforge.backends.inpaint.opencv_telea import OpencvTeleaInpaint
from layerforge.backends.inpaint.sd15 import Sd15AnimeInpaint
from layerforge.ops.inpaint_region import crop_to_mask, paste_crop


class RoutedInpaint:
    """Small holes → LaMa (Telea fallback). Large holes → SD1.5. Crop around the hole."""

    name = "auto"

    def __init__(
        self,
        cfg: dict,
        small=None,
        large=None,
        fallback=None,
    ) -> None:
        self.cfg = cfg
        inpaint_cfg = cfg.get("inpaint") or {}
        self.max_px = int(inpaint_cfg.get("small_hole_max_px", 4096))
        self.small_name = str(inpaint_cfg.get("small_hole_backend", "lama"))
        self.pad = int(inpaint_cfg.get("crop_pad_px", 32))
        self._small = small
        self._large = large
        self._fallback = fallback or OpencvTeleaInpaint()
        self.last_engine = "auto"

    def _small_backend(self):
        if self._small is None:
            if self.small_name in {"opencv.telea", "telea"}:
                self._small = self._fallback
            else:
                self._small = LamaInpaint(self.cfg)
        return self._small

    def _large_backend(self):
        if self._large is None:
            self._large = Sd15AnimeInpaint(self.cfg)
        return self._large

    def inpaint(self, image: np.ndarray, mask: np.ndarray, prompt: str) -> np.ndarray:
        area = int((mask > 0).sum())
        if area == 0:
            self.last_engine = "skip"
            return image.copy()
        crop_img, crop_mask, box = crop_to_mask(image, mask, pad=self.pad)
        if area <= self.max_px:
            filled_crop, engine = self._run_small(crop_img, crop_mask, prompt)
        else:
            filled_crop = self._large_backend().inpaint(crop_img, crop_mask, prompt)
            engine = self._large_backend().name
        self.last_engine = engine
        return paste_crop(image, filled_crop, box)

    def _run_small(self, image: np.ndarray, mask: np.ndarray, prompt: str):
        try:
            backend = self._small_backend()
            return backend.inpaint(image, mask, prompt), backend.name
        except Exception:
            return self._fallback.inpaint(image, mask, prompt), self._fallback.name

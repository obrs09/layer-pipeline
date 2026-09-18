from __future__ import annotations

import numpy as np


class OpencvTeleaInpaint:
    name = "opencv.telea"

    def inpaint(self, image: np.ndarray, mask: np.ndarray, prompt: str) -> np.ndarray:
        import cv2

        rgb = image[:, :, :3] if image.ndim == 3 and image.shape[2] >= 3 else image
        hole = (mask > 0).astype(np.uint8)
        if hole.sum() == 0:
            return image.copy()
        filled = cv2.inpaint(rgb, hole, 3, cv2.INPAINT_TELEA)
        out = image.copy()
        out[:, :, :3] = filled
        if out.shape[2] == 4:
            alpha = out[:, :, 3]
            alpha[hole > 0] = 255
            out[:, :, 3] = alpha
        return out

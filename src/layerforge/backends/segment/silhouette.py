from __future__ import annotations

import numpy as np

from layerforge.backends.segment.base import LayerMask, SegmentHints
from layerforge.ops.normalize import foreground_mask


class SilhouetteSegment:
    """CPU dry-run / fallback: one body mask from non-background pixels."""

    name = "silhouette"

    def __init__(self, background_luma: int = 250) -> None:
        self.background_luma = background_luma

    def segment(self, image: np.ndarray, hints: SegmentHints) -> list[LayerMask]:
        if hints.parts:
            return list(hints.parts)
        visible = foreground_mask(image, self.background_luma)
        return [
            LayerMask(
                role="body",
                label="silhouette",
                visible=visible,
                source="silhouette",
                rgba=None,
                notes="threshold silhouette; replace with sam2.hinted for real cuts",
            )
        ]

from __future__ import annotations

from typing import Protocol

import numpy as np

from layerforge.backends.segment.base import LayerMask, SegmentHints


class SegmentBackend(Protocol):
    name: str

    def segment(self, image: np.ndarray, hints: SegmentHints) -> list[LayerMask]:
        ...

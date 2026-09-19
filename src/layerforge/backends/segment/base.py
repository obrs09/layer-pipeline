from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SegmentHints:
    positive_points: list[tuple[int, int]] = field(default_factory=list)
    negative_points: list[tuple[int, int]] = field(default_factory=list)
    boxes: list[list[int]] = field(default_factory=list)
    parts: list["LayerMask"] = field(default_factory=list)


@dataclass
class LayerMask:
    role: str
    label: str
    visible: np.ndarray
    source: str
    rgba: np.ndarray | None = None
    notes: str = ""
    score: float = 1.0
    needs_click: bool = False

    @property
    def bbox(self) -> list[int]:
        from layerforge.image_io import bbox_from_mask

        return bbox_from_mask(self.visible)

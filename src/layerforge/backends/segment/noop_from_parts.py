from __future__ import annotations

from layerforge.backends.segment.base import LayerMask, SegmentHints


class NoopFromParts:
    name = "noop_from_parts"

    def segment(self, image, hints: SegmentHints) -> list[LayerMask]:
        if not hints.parts:
            raise ValueError("noop_from_parts needs Imagine-placed parts in hints.parts")
        return list(hints.parts)

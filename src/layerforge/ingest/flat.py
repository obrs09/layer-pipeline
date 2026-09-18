from __future__ import annotations

from pathlib import Path

import numpy as np

from layerforge.backends.segment.base import SegmentHints
from layerforge.image_io import open_rgba
from layerforge.ingest.imagine_parts import IngestResult


def ingest_flat(path: Path) -> IngestResult:
    if path.is_dir():
        raise ValueError(f"{path} is a directory; pass a single image for flat ingest")
    source = open_rgba(path)
    return IngestResult(
        kind="flat",
        source_path=path,
        source_rgba=source,
        hints=SegmentHints(),
    )

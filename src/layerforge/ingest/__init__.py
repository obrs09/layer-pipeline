from __future__ import annotations

from pathlib import Path

from layerforge.ingest.flat import ingest_flat
from layerforge.ingest.imagine_parts import ingest_imagine, is_imagine_dir
from layerforge.taxonomy import Taxonomy


def ingest_input(path: Path, taxonomy: Taxonomy):
    path = Path(path)
    if path.is_dir() and is_imagine_dir(path):
        return ingest_imagine(path, taxonomy)
    if path.is_file():
        return ingest_flat(path)
    if path.is_dir():
        raise ValueError(
            f"{path} has no segments/ or parts/ folder. Pass a single image, or an Imagine job directory."
        )
    raise FileNotFoundError(path)

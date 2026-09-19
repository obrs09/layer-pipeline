from __future__ import annotations

from layerforge.backends.inpaint.router import RoutedInpaint
from layerforge.backends.inpaint.identity import IdentityInpaint
from layerforge.backends.inpaint.lama import LamaInpaint
from layerforge.backends.inpaint.opencv_telea import OpencvTeleaInpaint
from layerforge.backends.inpaint.sd15 import Sd15AnimeInpaint
from layerforge.backends.segment.noop_from_parts import NoopFromParts
from layerforge.backends.segment.sam_hinted import Sam3HintedSegment, SamHintedSegment
from layerforge.backends.segment.silhouette import SilhouetteSegment


def build_segment(name: str, cfg: dict, kind: str, dry_run: bool):
    resolved = name
    if name == "auto":
        if kind == "imagine":
            resolved = "noop_from_parts"
        elif dry_run:
            resolved = "silhouette"
        else:
            resolved = "sam2.hinted"
    mapping = {
        "noop_from_parts": lambda: NoopFromParts(),
        "silhouette": lambda: SilhouetteSegment(
            background_luma=int((cfg.get("occlusion") or {}).get("background_luma", 250))
        ),
        "sam2.hinted": lambda: SamHintedSegment(cfg),
        "sam3.hinted": lambda: Sam3HintedSegment(),
    }
    if resolved not in mapping:
        raise ValueError(f"unknown segment backend: {resolved}")
    backend = mapping[resolved]()
    backend.name = resolved
    return backend


def build_inpaint(name: str, cfg: dict, dry_run: bool):
    resolved = "identity" if dry_run and name in {"auto", "sd15.anime", "lama"} else name
    if resolved == "auto":
        backend = RoutedInpaint(cfg)
        backend.name = "auto"
        return backend
    mapping = {
        "identity": lambda: IdentityInpaint(),
        "lama": lambda: LamaInpaint(cfg),
        "sd15.anime": lambda: Sd15AnimeInpaint(cfg),
        "opencv.telea": lambda: OpencvTeleaInpaint(),
    }
    if resolved not in mapping:
        raise ValueError(f"unknown inpaint backend: {resolved}")
    backend = mapping[resolved]()
    backend.name = resolved
    return backend

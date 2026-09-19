from __future__ import annotations

from layerforge.backends.detect.anime_segmentation import AniSegCut, AnimeSegmentationCut
from layerforge.backends.detect.dwpose import DwPoseEstimate
from layerforge.backends.detect.grounding_dino import GroundingDinoBoxes
from layerforge.backends.detect.wdtagger import WdTagger
from layerforge.backends.inpaint.router import RoutedInpaint
from layerforge.backends.inpaint.identity import IdentityInpaint
from layerforge.backends.inpaint.lama import LamaInpaint
from layerforge.backends.inpaint.opencv_telea import OpencvTeleaInpaint
from layerforge.backends.inpaint.sd15 import Sd15AnimeInpaint
from layerforge.backends.segment.cascade import CascadeSegment
from layerforge.backends.segment.noop_from_parts import NoopFromParts
from layerforge.backends.segment.sam_hinted import Sam3HintedSegment, SamHintedSegment
from layerforge.backends.segment.sam3_text import Sam3TextMasker
from layerforge.backends.segment.silhouette import SilhouetteSegment
from layerforge.taxonomy import load_taxonomy


def build_segment(name: str, cfg: dict, kind: str, dry_run: bool):
    resolved = name
    if name == "auto":
        if kind == "imagine":
            resolved = "cascade"
        elif dry_run:
            resolved = "silhouette"
        else:
            resolved = "cascade"
    mapping = {
        "noop_from_parts": lambda: NoopFromParts(),
        "silhouette": lambda: SilhouetteSegment(
            background_luma=int((cfg.get("occlusion") or {}).get("background_luma", 250))
        ),
        "sam2.hinted": lambda: SamHintedSegment(cfg),
        "sam3.hinted": lambda: Sam3HintedSegment(),
        "cascade": lambda: _build_cascade(cfg, dry_run),
    }
    if resolved not in mapping:
        raise ValueError(f"unknown segment backend: {resolved}")
    backend = mapping[resolved]()
    backend.name = resolved
    return backend


def _build_cascade(cfg: dict, dry_run: bool) -> CascadeSegment:
    taxonomy = load_taxonomy()
    return CascadeSegment(
        cfg,
        taxonomy,
        character=_build_character(cfg, allow_luma=dry_run),
        tagger=None if dry_run else WdTagger(cfg),
        boxes=None if dry_run else GroundingDinoBoxes(cfg),
        sam3=None if dry_run else Sam3TextMasker(cfg),
        sam2=None if dry_run else SamHintedSegment(cfg),
        pose=None if dry_run else _build_pose(cfg),
        dry_run=dry_run,
    )


def _build_character(cfg: dict, *, allow_luma: bool):
    name = (cfg.get("cascade") or {}).get("character", "anime_segmentation")
    mapping = {
        "anime_segmentation": lambda: AnimeSegmentationCut(cfg, allow_luma=allow_luma),
        "aniseg": lambda: AniSegCut(cfg, allow_luma=allow_luma),
    }
    if name not in mapping:
        raise ValueError(f"unknown character backend: {name}")
    return mapping[name]()


def _build_pose(cfg: dict):
    name = (cfg.get("cascade") or {}).get("pose", "dwpose")
    if not name or str(name).lower() in {"none", "off", "false"}:
        return None
    if name == "dwpose":
        return DwPoseEstimate(cfg)
    raise ValueError(f"unknown pose backend: {name}")


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

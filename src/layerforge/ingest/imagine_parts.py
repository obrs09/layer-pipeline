from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from layerforge.backends.segment.base import LayerMask, SegmentHints
from layerforge.image_io import IMAGE_SUFFIXES, bbox_from_mask, is_image, open_rgba
from layerforge.ingest.place import place_crop
from layerforge.taxonomy import Taxonomy

SOURCE_NAMES = {"source.png", "source.jpg", "source.jpeg", "source.webp"}


@dataclass
class IngestResult:
    kind: str  # flat | imagine
    source_path: Path
    source_rgba: np.ndarray
    hints: SegmentHints = field(default_factory=SegmentHints)


def is_imagine_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "segments").is_dir() or (path / "parts").is_dir()


def find_source(path: Path) -> Path:
    for name in SOURCE_NAMES:
        candidate = path / name
        if candidate.exists():
            return candidate
    grok = sorted(path.glob("grok-image-*.jpg")) + sorted(path.glob("grok-image-*.png"))
    grok += sorted(path.glob("grok-image-*.jpeg")) + sorted(path.glob("grok-image-*.webp"))
    if grok:
        return grok[0]
    skip = {"segments", "parts"}
    rasters = [
        p
        for p in path.iterdir()
        if p.is_file() and is_image(p) and p.parent.name not in skip
    ]
    if len(rasters) == 1:
        return rasters[0]
    raise FileNotFoundError(f"No source image in {path}")


def iter_part_files(job_dir: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for folder_name in ("segments", "parts"):
        root = job_dir / folder_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            files.append((_label_for(path, root), path))
    return files


def _label_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    stem = path.stem
    if rel.parent != Path("."):
        folder = rel.parts[0]
        prefix = folder + "-"
        if stem.startswith(prefix) or stem == folder:
            return folder
        return folder
    return stem


def ingest_imagine(job_dir: Path, taxonomy: Taxonomy) -> IngestResult:
    source_path = find_source(job_dir)
    source = open_rgba(source_path)
    parts: list[LayerMask] = []
    unknown_i = 0
    pending_pairs: dict[str, list[LayerMask]] = {}
    search_by_label: dict[str, np.ndarray] = {}
    loaded: list[tuple[str, Path, np.ndarray]] = []
    for label, part_path in iter_part_files(job_dir):
        if len(label) <= 1:
            continue
        loaded.append((label, part_path, open_rgba(part_path)))
    loaded.sort(key=lambda item: int((item[2][:, :, 3] > 0).sum()), reverse=True)
    sh, sw = source.shape[:2]
    source_area = sh * sw
    char_roi: list[int] | None = None
    for label, part_path, crop in loaded:
        search = search_by_label.setdefault(label, source.copy())
        crop_area = int((crop[:, :, 3] > 0).sum())
        roi = char_roi if crop_area < 0.2 * source_area else None
        placed, score, notes = place_crop(search, crop, roi=roi)
        used = placed[:, :, 3] > 0
        search[used] = 255
        visible = placed[:, :, 3]
        if int((visible > 0).sum()) == 0:
            continue
        if char_roi is None and int(used.sum()) > 0.12 * source_area:
            x, y, bw, bh = bbox_from_mask(visible)
            pad = 48
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(sw, x + bw + pad)
            y1 = min(sh, y + bh + pad)
            char_roi = [x0, y0, x1 - x0, y1 - y0]
        family = taxonomy.pair_family(label)
        role = taxonomy.resolve_label(label)
        mask = LayerMask(
            role=role or label,
            label=label,
            visible=visible,
            source="imagine_part",
            rgba=placed,
            notes=notes,
            score=score,
        )
        if family:
            pending_pairs.setdefault(family, []).append(mask)
            continue
        if role is None:
            mask.role = f"unknown_{unknown_i}"
            mask.notes = f"unmapped label={label}; {notes}".strip("; ")
            unknown_i += 1
        parts.append(mask)
    for family, masks in pending_pairs.items():
        assigned = _assign_pairs(family, masks, taxonomy)
        parts.extend(assigned)
    hints = SegmentHints(
        parts=parts,
        boxes=[
            [b[0], b[1], b[0] + b[2], b[1] + b[3]]
            for p in parts
            for b in [p.bbox]
            if b[2] > 0 and b[3] > 0
        ],
    )
    return IngestResult(
        kind="imagine",
        source_path=source_path,
        source_rgba=source,
        hints=hints,
    )


def _assign_pairs(family: str, masks: list[LayerMask], taxonomy: Taxonomy) -> list[LayerMask]:
    left_role, right_role = taxonomy.pair_roles[family]
    split: list[LayerMask] = []
    for mask in masks:
        pieces = _split_components(mask)
        split.extend(pieces if len(pieces) >= 2 else [mask])
    if len(split) > 2:
        split = _pick_pair(split)
    split = sorted(split, key=lambda m: m.bbox[0] + m.bbox[2] / 2)
    if len(split) == 1:
        split[0].role = left_role
        split[0].notes += f"; single {family} instance, default {left_role}"
        return split
    split[0].role = left_role
    split[1].role = right_role
    return split[:2]


def _center(mask: LayerMask) -> tuple[float, float]:
    x, y, w, h = mask.bbox
    return x + w / 2, y + h / 2


def _pick_pair(masks: list[LayerMask]) -> list[LayerMask]:
    import itertools

    best: tuple[float, LayerMask, LayerMask] | None = None
    for a, b in itertools.combinations(masks, 2):
        ax, ay = _center(a)
        bx, by = _center(b)
        area_a = max(1, int((a.visible > 0).sum()))
        area_b = max(1, int((b.visible > 0).sum()))
        ratio = max(area_a, area_b) / min(area_a, area_b)
        dx = abs(ax - bx)
        dy = abs(ay - by)
        if dx < 4:
            continue
        # Real L/R pairs share a row and a similar area; skip a blob stacked far below.
        cost = dy + 40.0 * (ratio - 1.0) - 0.15 * dx
        if best is None or cost < best[0]:
            best = (cost, a, b)
    if best is None:
        return sorted(masks, key=lambda m: int((m.visible > 0).sum()), reverse=True)[:2]
    return [best[1], best[2]]


def _split_components(mask: LayerMask, min_area: int = 24) -> list[LayerMask]:
    try:
        import cv2
    except ImportError:
        return [mask]
    vis = (mask.visible > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(vis, connectivity=8)
    pieces: list[LayerMask] = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cc = labels == i
        visible = cc.astype(np.uint8) * 255
        rgba = None
        if mask.rgba is not None:
            rgba = mask.rgba.copy()
            rgba[~cc, 3] = 0
        pieces.append(
            LayerMask(
                role=mask.role,
                label=mask.label,
                visible=visible,
                source=mask.source,
                rgba=rgba,
                notes=mask.notes,
                score=mask.score,
            )
        )
    return pieces if len(pieces) >= 2 else [mask]

from __future__ import annotations

import shutil
from pathlib import Path

from layerforge.contracts import Manifest
from layerforge.image_io import save_png
from layerforge.ops.qa import write_preview

_PACK_DIRS = ("layers", "masks", "preview")


def _reset_pack_dirs(out_dir: Path) -> None:
    for name in _PACK_DIRS:
        dest = out_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)


def export_png_pack(
    out_dir: Path,
    manifest: Manifest,
    source,
    layers_rgba: dict[str, object],
    masks_visible: dict[str, object],
    masks_occluded: dict[str, object],
    composite,
    visible_union,
) -> None:
    _reset_pack_dirs(out_dir)
    save_png(out_dir / "source.png", source)
    for rec in manifest.layers:
        save_png(out_dir / rec.file, layers_rgba[rec.id])
        save_png(out_dir / rec.mask_visible, masks_visible[rec.id])
        if rec.mask_occluded:
            save_png(out_dir / rec.mask_occluded, masks_occluded[rec.id])
    write_preview(out_dir / "preview", source, composite, visible_union)

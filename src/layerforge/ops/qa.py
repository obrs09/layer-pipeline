from __future__ import annotations

from pathlib import Path

import numpy as np

from layerforge.image_io import alpha_over, save_png


def write_preview(
    preview_dir: Path,
    source: np.ndarray,
    composite: np.ndarray,
    visible_union: np.ndarray | None = None,
) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    src_rgb = source[:, :, :3]
    comp_rgb = composite[:, :, :3]
    stack = np.concatenate([src_rgb, comp_rgb], axis=1)
    save_png(preview_dir / "stack.png", stack)
    diff = np.abs(comp_rgb.astype(np.int16) - src_rgb.astype(np.int16)).astype(np.uint8)
    if visible_union is not None:
        leak = diff.copy()
        leak[visible_union == 0] = 0
        save_png(preview_dir / "diff.png", leak)
    else:
        save_png(preview_dir / "diff.png", diff)

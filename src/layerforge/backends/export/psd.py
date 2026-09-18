from __future__ import annotations

from pathlib import Path


def export_psd(out_dir: Path, layer_files: list[Path]) -> Path | None:
    """Optional PSD. v0 writes only if pytoshop is installed."""
    try:
        import pytoshop
        from pytoshop.user import nested_layers
        from PIL import Image
        import numpy as np
    except ImportError:
        return None
    layers = []
    for path in layer_files:
        rgba = np.array(Image.open(path).convert("RGBA"))
        channels = {
            0: rgba[:, :, 0],
            1: rgba[:, :, 1],
            2: rgba[:, :, 2],
            -1: rgba[:, :, 3],
        }
        layers.append(
            nested_layers.Image(
                name=path.stem,
                channels=channels,
                opacity=255,
            )
        )
    psd = nested_layers.nested_layers_to_psd(layers, color_mode=pytoshop.enums.ColorMode.rgb)
    dest = out_dir / "layers.psd"
    with dest.open("wb") as fh:
        psd.write(fh)
    return dest

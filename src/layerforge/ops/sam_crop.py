from __future__ import annotations


def expand_box(box: list[float], pad_frac: float, width: int, height: int) -> list[float]:
    """Grow xyxy by pad_frac of its size and clip to the image."""
    x0, y0, x1, y1 = [float(v) for v in box]
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    pad_x = bw * float(pad_frac)
    pad_y = bh * float(pad_frac)
    return [
        max(0.0, x0 - pad_x),
        max(0.0, y0 - pad_y),
        min(float(width), x1 + pad_x),
        min(float(height), y1 + pad_y),
    ]


def crop_origin(box: list[float], pad_frac: float, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(round(v)) for v in expand_box(box, pad_frac, width, height)]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    return x0, y0, x1, y1


def upscale_hw(height: int, width: int, min_side: int) -> tuple[int, int, float]:
    """Scale so the short side is at least min_side. Never downscale."""
    short = max(1, min(height, width))
    scale = max(1.0, float(min_side) / float(short))
    if scale <= 1.0:
        return height, width, 1.0
    return max(8, int(round(height * scale))), max(8, int(round(width * scale))), scale


def to_crop_xy(x: float, y: float, origin: tuple[int, int], scale: float) -> tuple[float, float]:
    return ((x - origin[0]) * scale, (y - origin[1]) * scale)


def box_to_crop(box: list[float], origin: tuple[int, int], scale: float) -> list[float]:
    x0, y0 = to_crop_xy(box[0], box[1], origin, scale)
    x1, y1 = to_crop_xy(box[2], box[3], origin, scale)
    return [x0, y0, x1, y1]


def points_to_crop(
    points: list[tuple[float, float]],
    origin: tuple[int, int],
    scale: float,
    crop_w: int,
    crop_h: int,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for x, y in points:
        cx, cy = to_crop_xy(x, y, origin, scale)
        if 0.0 <= cx < float(crop_w) and 0.0 <= cy < float(crop_h):
            out.append((cx, cy))
    return out

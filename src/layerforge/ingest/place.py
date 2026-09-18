from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    cv2 = None
    _CV2_ERR = exc
else:
    _CV2_ERR = None


def place_crop(
    source_rgba: np.ndarray,
    crop_rgba: np.ndarray,
    roi: list[int] | None = None,
) -> tuple[np.ndarray, float, str]:
    """Template-match a cropped Imagine sprite onto the full source canvas."""
    if cv2 is None:
        raise RuntimeError("opencv-python-headless is required to place Imagine crops") from _CV2_ERR
    sh, sw = source_rgba.shape[:2]
    ch, cw = crop_rgba.shape[:2]
    canvas = np.zeros((sh, sw, 4), dtype=np.uint8)
    if ch > sh or cw > sw:
        return canvas, 0.0, "crop larger than source; skipped"
    alpha = crop_rgba[:, :, 3]
    if int((alpha > 0).sum()) < 8:
        return canvas, 0.0, "crop has no opaque pixels"
    src_full = source_rgba[:, :, :3]
    ox, oy = 0, 0
    src = src_full
    if roi is not None and roi[2] >= cw and roi[3] >= ch:
        x0, y0, rw, rh = roi
        x0 = min(max(0, int(x0)), sw - 1)
        y0 = min(max(0, int(y0)), sh - 1)
        x1 = min(sw, x0 + max(int(rw), cw))
        y1 = min(sh, y0 + max(int(rh), ch))
        src = src_full[y0:y1, x0:x1]
        ox, oy = x0, y0
        if src.shape[0] < ch or src.shape[1] < cw:
            src = src_full
            ox, oy = 0, 0
    templ = crop_rgba[:, :, :3]
    mask = (alpha > 0).astype(np.uint8) * 255
    rh, rw = src.shape[:2]
    scale = 1.0
    if max(ch, cw) > 256 or max(rh, rw) > 1024:
        scale = 0.5 if max(rh, rw) <= 2048 else 0.25
    if scale != 1.0:
        src_s = cv2.resize(src, (int(rw * scale), int(rh * scale)), interpolation=cv2.INTER_AREA)
        tw, th = max(1, int(cw * scale)), max(1, int(ch * scale))
        templ_s = cv2.resize(templ, (tw, th), interpolation=cv2.INTER_AREA)
        mask_s = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
        if templ_s.shape[0] >= src_s.shape[0] or templ_s.shape[1] >= src_s.shape[1]:
            scale = 1.0
            src_s, templ_s, mask_s = src, templ, mask
    else:
        src_s, templ_s, mask_s = src, templ, mask
    result = cv2.matchTemplate(src_s, templ_s, cv2.TM_CCORR_NORMED, mask=mask_s)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    x = ox + int(round(max_loc[0] / scale))
    y = oy + int(round(max_loc[1] / scale))
    x = min(max(0, x), sw - cw)
    y = min(max(0, y), sh - ch)
    if scale != 1.0:
        x, y, max_val = _refine(src_full, templ, mask, x, y)
    canvas[y : y + ch, x : x + cw] = crop_rgba
    notes = f"placed@({x},{y}) score={max_val:.3f}"
    if max_val < 0.75:
        notes += "; low match"
    return canvas, float(max_val), notes


def _refine(src, templ, mask, x, y) -> tuple[int, int, float]:
    sh, sw = src.shape[:2]
    ch, cw = templ.shape[:2]
    pad = 8
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(sw, x + cw + pad)
    y1 = min(sh, y + ch + pad)
    window = src[y0:y1, x0:x1]
    if window.shape[0] < ch or window.shape[1] < cw:
        return x, y, 0.0
    result = cv2.matchTemplate(window, templ, cv2.TM_CCORR_NORMED, mask=mask)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return x0 + max_loc[0], y0 + max_loc[1], float(max_val)

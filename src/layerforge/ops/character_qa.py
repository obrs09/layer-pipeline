from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class CharacterQA:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def dice_coef(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = int(np.logical_and(aa, bb).sum())
    denom = int(aa.sum()) + int(bb.sum())
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def shannon_entropy(prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def connected_component_metrics(
    mask: np.ndarray,
    *,
    min_component_px: int = 50,
) -> dict:
    import cv2

    binary = (mask > 0).astype(np.uint8)
    fg = int(binary.sum())
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    kept = [area for area in areas if area >= min_component_px]
    largest = max(areas, default=0)
    return {
        "fg_px": fg,
        "n_raw": max(0, n - 1),
        "n_kept": len(kept),
        "largest_px": largest,
        "largest_frac": (largest / fg) if fg else 0.0,
    }


def evaluate_character_qa(
    mask: np.ndarray,
    prob: np.ndarray | None,
    peers: dict[str, np.ndarray | None],
    cfg: dict | None = None,
) -> CharacterQA:
    cfg = cfg or {}
    min_largest = float(cfg.get("min_largest_frac", 0.60))
    max_components = int(cfg.get("max_components", 15))
    min_island = int(cfg.get("min_component_px", 50))
    gray_lo = float(cfg.get("gray_lo", 0.15))
    gray_hi = float(cfg.get("gray_hi", 0.85))
    max_gray = float(cfg.get("max_gray_frac", 0.25))
    erode_px = int(cfg.get("erode_px", 15))
    entropy_high = float(cfg.get("entropy_high", 0.5))
    max_interior = float(cfg.get("max_interior_entropy_frac", 0.05))
    min_dice = float(cfg.get("min_dice", 0.85))

    reasons: list[str] = []
    metrics: dict = {}

    cc = connected_component_metrics(mask, min_component_px=min_island)
    metrics.update(cc)
    if cc["fg_px"] < 64:
        reasons.append("empty_mask")
    if cc["largest_frac"] < min_largest:
        reasons.append(
            f"largest_frac={cc['largest_frac']:.3f}<{min_largest:.2f}"
        )
    if cc["n_kept"] > max_components:
        reasons.append(f"components={cc['n_kept']}>{max_components}")

    if prob is not None:
        p = np.asarray(prob, dtype=np.float32)
        if p.shape != mask.shape[:2]:
            raise ValueError("prob shape must match mask")
        fg = p > gray_lo
        gray = (p >= gray_lo) & (p <= gray_hi)
        gray_frac = float(gray.sum() / max(1, int(fg.sum())))
        metrics["gray_frac"] = gray_frac
        if gray_frac > max_gray:
            reasons.append(f"gray_frac={gray_frac:.3f}>{max_gray:.2f}")

        entropy = shannon_entropy(p)
        interior = _erode_bool(mask > 0, erode_px)
        interior_n = int(interior.sum())
        high = interior & (entropy > entropy_high)
        interior_frac = float(high.sum() / max(1, interior_n))
        metrics["interior_px"] = interior_n
        metrics["interior_entropy_frac"] = interior_frac
        metrics["entropy_mean"] = float(entropy.mean())
        if interior_n >= 64 and interior_frac > max_interior:
            reasons.append(
                f"interior_entropy={interior_frac:.3f}>{max_interior:.2f}"
            )
    else:
        metrics["gray_frac"] = None
        metrics["interior_entropy_frac"] = None

    dice_scores: dict[str, float | None] = {}
    for name, other in (peers or {}).items():
        if other is None:
            dice_scores[name] = None
            continue
        score = float(dice_coef(mask, other))
        dice_scores[name] = score
        if score < min_dice:
            reasons.append(f"dice_{name}={score:.3f}<{min_dice:.2f}")
    metrics["dice"] = dice_scores

    return CharacterQA(ok=not reasons, reasons=reasons, metrics=_json_safe(metrics))


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _erode_bool(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    import cv2

    k = 2 * int(px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode((mask > 0).astype(np.uint8), kernel)
    return eroded > 0

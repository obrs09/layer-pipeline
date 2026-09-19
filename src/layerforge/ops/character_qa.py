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


def containment(inner: np.ndarray, outer: np.ndarray) -> float:
    a = inner > 0
    b = outer > 0
    denom = int(a.sum())
    if denom == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / denom)


def extra_frac(outer: np.ndarray, inner: np.ndarray) -> float:
    b = outer > 0
    denom = int(b.sum())
    if denom == 0:
        return 0.0
    return float(np.logical_and(b, inner <= 0).sum() / denom)


def structural_score(mask: np.ndarray, cfg: dict | None = None) -> float:
    report = evaluate_character_qa(mask, None, {}, cfg)
    if not report.ok or int(report.metrics.get("fg_px") or 0) < 64:
        return -1.0
    n_kept = int(report.metrics.get("n_kept") or 0)
    largest = float(report.metrics.get("largest_frac") or 0.0)
    return largest - 0.01 * max(0, n_kept - 2)


def choose_peer_character(
    isnet: np.ndarray,
    peers: dict[str, np.ndarray | None],
    cfg: dict | None = None,
) -> dict | None:
    """If ToonOut/MODNet agree more than isnet, return a replacement mask.

    Two agreeing peers are AND-ed. Otherwise the structurally better peer
    is used, as long as it still covers enough of the isnet silhouette.
    """
    cfg = cfg or {}
    if not bool(cfg.get("prefer_peers", True)):
        return None
    min_dice = float(cfg.get("min_dice", 0.85))
    min_agree = float(cfg.get("min_peer_agree", min_dice))
    margin = float(cfg.get("peer_better_margin", 0.02))
    contain_min = float(cfg.get("overinclude_contain", 0.90))
    extra_min = float(cfg.get("overinclude_extra", 0.10))
    keep_min = float(cfg.get("min_keep_of_isnet", 0.50))
    isnet_fg = int((isnet > 0).sum())
    if isnet_fg < 64:
        return None

    available = {
        name: ((mask > 0).astype(np.uint8) * 255)
        for name, mask in (peers or {}).items()
        if mask is not None and int((mask > 0).sum()) >= 64
    }
    if not available:
        return None

    names = list(available)
    isnet_dices = {name: float(dice_coef(isnet, mask)) for name, mask in available.items()}
    contains = {name: containment(mask, isnet) for name, mask in available.items()}
    extras = {name: extra_frac(isnet, mask) for name, mask in available.items()}
    peer_dice = None
    if len(available) >= 2:
        peer_dice = float(dice_coef(available[names[0]], available[names[1]]))

    max_isnet_dice = max(isnet_dices.values())
    mean_extra = float(sum(extras.values()) / len(extras))
    peers_agree = peer_dice is not None and peer_dice >= min_agree
    peers_closer = peers_agree and peer_dice > max_isnet_dice + margin
    isnet_outlier = peers_agree and max_isnet_dice < min_dice
    overinclude = (
        len(available) >= 1
        and all(value >= contain_min for value in contains.values())
        and mean_extra >= extra_min
        and (peers_agree or len(available) == 1)
    )
    if not (peers_closer or isnet_outlier or overinclude):
        return None

    chosen = None
    source = None
    if peers_agree:
        combo = np.logical_and(available[names[0]] > 0, available[names[1]] > 0)
        combo_u8 = combo.astype(np.uint8) * 255
        if (
            int(combo.sum()) >= max(64, int(keep_min * isnet_fg))
            and structural_score(combo_u8, cfg) >= 0
        ):
            chosen = combo_u8
            source = "peer_and"
    if chosen is None:
        ranked = sorted(
            available.items(),
            key=lambda item: (structural_score(item[1], cfg), int((item[1] > 0).sum())),
            reverse=True,
        )
        name, mask = ranked[0]
        if structural_score(mask, cfg) < 0:
            return None
        if int((mask > 0).sum()) < max(64, int(keep_min * isnet_fg)):
            return None
        chosen = mask
        source = name

    others = {name: mask for name, mask in available.items() if name != source}
    report = evaluate_character_qa(chosen, (chosen > 0).astype(np.float32), others, cfg)
    if not report.ok and source == "peer_and":
        # AND is allowed to fail peer-dice vs a looser peer; structural still required.
        structural = evaluate_character_qa(chosen, None, {}, cfg)
        if not structural.ok:
            return None
        report = structural

    return {
        "mask": chosen,
        "source": source,
        "reasons": [
            key
            for key, flag in (
                ("peers_closer", peers_closer),
                ("isnet_outlier", isnet_outlier),
                ("overinclude", overinclude),
            )
            if flag
        ],
        "metrics": _json_safe(
            {
                "peer_dice": peer_dice,
                "isnet_dice": isnet_dices,
                "contain_in_isnet": contains,
                "isnet_extra": extras,
                "chosen_fg": int((chosen > 0).sum()),
                "isnet_fg": isnet_fg,
            }
        ),
        "qa": report.to_dict(),
    }


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

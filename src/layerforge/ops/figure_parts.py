from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LIMB_ROLES = ("arm_l", "arm_r", "leg_l", "leg_r")
PART_ROLES = LIMB_ROLES + ("torso",)


@dataclass
class FigureParts:
    skins: dict[str, np.ndarray]
    clothes_parts: dict[str, np.ndarray]
    clothes: np.ndarray
    remainder: np.ndarray
    notes: dict[str, int] = field(default_factory=dict)


def split_figure(
    body: np.ndarray,
    parts: dict[str, np.ndarray],
    clothes: np.ndarray | None,
) -> FigureParts:
    """Split a whole figure into pose parts, clothes on those parts, and leftover.

    Skin is the part of each pose piece that is not clothes. Clothes stays the
    garment mask on the figure. Pixels in neither a pose part nor clothes are
    the remainder (accessories until something else claims them).
    """
    body_b = body > 0
    clothes_b = np.zeros(body_b.shape, dtype=bool) if clothes is None else (clothes > 0) & body_b
    claimed = np.zeros(body_b.shape, dtype=bool)
    skins: dict[str, np.ndarray] = {}
    clothes_parts: dict[str, np.ndarray] = {}
    owned: dict[str, np.ndarray] = {}
    for role in LIMB_ROLES:
        part = (parts.get(role, np.zeros(body_b.shape, dtype=np.uint8)) > 0) & body_b & ~claimed
        claimed |= part
        owned[role] = part
    torso = (parts.get("torso", np.zeros(body_b.shape, dtype=np.uint8)) > 0) & body_b & ~claimed
    claimed |= torso
    owned["torso"] = torso
    for role in PART_ROLES:
        part = owned[role]
        inter = part & clothes_b
        skin = part & ~clothes_b
        clothes_parts[role] = inter.astype(np.uint8) * 255
        skins[role] = skin.astype(np.uint8) * 255
    remainder = body_b & ~claimed & ~clothes_b
    return FigureParts(
        skins=skins,
        clothes_parts=clothes_parts,
        clothes=clothes_b.astype(np.uint8) * 255,
        remainder=remainder.astype(np.uint8) * 255,
        notes={
            "body_px": int(body_b.sum()),
            "clothes_px": int(clothes_b.sum()),
            "remainder_px": int(remainder.sum()),
            **{f"{role}_skin": int((skins[role] > 0).sum()) for role in PART_ROLES},
            **{f"{role}_clothes": int((clothes_parts[role] > 0).sum()) for role in PART_ROLES},
        },
    )

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from layerforge.image_io import save_png
from layerforge.ops.normalize import to_rgba

ROLE_COLORS: dict[str, tuple[int, int, int]] = {
    "body": (80, 200, 80),
    "clothes": (50, 120, 255),
    "face": (255, 170, 70),
    "hair_front": (220, 80, 200),
    "hair_back": (160, 50, 160),
    "eye_l": (255, 80, 80),
    "eye_r": (255, 40, 40),
    "mouth": (255, 120, 160),
    "arm_l": (80, 220, 220),
    "arm_r": (40, 180, 180),
    "acc": (255, 220, 60),
}


def masked_rgba(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgba = to_rgba(image).copy()
    rgba[mask <= 0, 3] = 0
    return rgba


def tint_overlay(
    image: np.ndarray,
    items: list[tuple[np.ndarray, tuple[int, int, int]]],
    base_dim: float = 0.4,
    tint: float = 0.55,
) -> np.ndarray:
    out = to_rgba(image)[:, :, :3].astype(np.float32) * base_dim
    for mask, color in items:
        m = (mask > 0).astype(np.float32)[..., None]
        if float(m.sum()) == 0:
            continue
        arr = np.array(color, dtype=np.float32)
        out = out * (1.0 - tint * m) + arr * (tint * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def role_color(role: str) -> tuple[int, int, int]:
    if role in ROLE_COLORS:
        return ROLE_COLORS[role]
    h = abs(hash(role)) % 180
    return (40 + h, 80, 220 - h // 2)


class StepDump:
    """Numbered per-stage previews. Does not change the PNG pack contract."""

    def __init__(self, out_dir: Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.root = Path(out_dir) / "steps"

    def reset(self) -> None:
        if not self.enabled:
            return
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_png(self, rel: str, array: np.ndarray) -> None:
        if not self.enabled:
            return
        save_png(self.root / rel, array)

    def write_json(self, rel: str, payload: object) -> None:
        if not self.enabled:
            return
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def write_character(self, image: np.ndarray, mask: np.ndarray, *, flagged: bool = False) -> None:
        if not self.enabled:
            return
        self.write_png("01_character/mask.png", mask)
        self.write_png("01_character/rgba.png", masked_rgba(image, mask))
        color = (255, 70, 40) if flagged else (0, 210, 255)
        self.write_png(
            "01_character/overlay.png",
            tint_overlay(image, [(mask, color)]),
        )

    def write_character_raw(self, image: np.ndarray, mask: np.ndarray) -> None:
        if not self.enabled:
            return
        self.write_png("01_character/raw_mask.png", mask)
        self.write_png("01_character/raw.png", masked_rgba(image, mask))

    def write_pose(self, image: np.ndarray, estimate) -> None:
        if not self.enabled or estimate is None:
            return
        person = estimate.person_mask
        self.write_png("01_pose/person.png", person)
        self.write_png("01_pose/rgba.png", masked_rgba(image, person))
        overlay = getattr(estimate, "overlay", None)
        if overlay is not None:
            self.write_png("01_pose/overlay.png", overlay)
        self.write_json(
            "01_pose/keypoints.json",
            {
                "person_box": list(getattr(estimate, "person_box", []) or []),
                "boxes": dict(getattr(estimate, "boxes", {}) or {}),
                "keypoints": list(getattr(estimate, "keypoints", []) or []),
                "notes": str(getattr(estimate, "notes", "") or ""),
            },
        )

    def write_boxes(self, image: np.ndarray, boxes: list[dict]) -> None:
        if not self.enabled:
            return
        numbered = []
        for idx, item in enumerate(boxes):
            row = dict(item)
            row["index"] = int(item.get("index", idx))
            numbered.append(row)
        self.write_json("03_boxes/boxes.json", numbered)
        rgb = to_rgba(image)[:, :, :3].copy()
        canvas = Image.fromarray(rgb)
        draw = ImageDraw.Draw(canvas)
        for item in numbered:
            xyxy = item.get("xyxy") or []
            if len(xyxy) != 4:
                continue
            x0, y0, x1, y1 = [int(round(float(v))) for v in xyxy]
            role = str(item.get("role") or "acc")
            color = role_color(role)
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            query = str(item.get("query") or role)
            label = f"{int(item['index']):02d} {query}"
            draw.text((x0 + 2, max(0, y0 - 12)), label[:40], fill=color)
            one = Image.fromarray(rgb.copy())
            one_draw = ImageDraw.Draw(one)
            one_draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            one_draw.text((x0 + 2, max(0, y0 - 12)), label[:40], fill=color)
            self.write_png(f"03_boxes/{int(item['index']):02d}_{role}.png", np.array(one))
        self.write_png("03_boxes/overlay.png", np.array(canvas))

    def write_layers(
        self,
        step: str,
        image: np.ndarray,
        layers,
        ids: list[str],
        occluded: dict[int, np.ndarray] | None = None,
    ) -> None:
        if not self.enabled:
            return
        tints: list[tuple[np.ndarray, tuple[int, int, int]]] = []
        for idx, (layer_id, layer) in enumerate(zip(ids, layers)):
            mask = layer.visible
            self.write_png(f"{step}/{layer_id}.mask.png", mask)
            self.write_png(f"{step}/{layer_id}.png", masked_rgba(image, mask))
            tints.append((mask, role_color(layer.role)))
            if occluded is not None:
                hole = occluded.get(idx)
                if hole is not None and int((hole > 0).sum()) > 0:
                    self.write_png(f"{step}/{layer_id}.occluded.png", hole)
        self.write_png(f"{step}/overlay.png", tint_overlay(image, tints))

    def write_inpaint_hole(self, layer_id: str, filled: np.ndarray, occluded: np.ndarray) -> None:
        if not self.enabled or int((occluded > 0).sum()) == 0:
            return
        self.write_png(f"07_inpaint/{layer_id}.png", masked_rgba(filled, occluded))

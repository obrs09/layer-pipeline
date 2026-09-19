from __future__ import annotations

from dataclasses import replace

import numpy as np

from layerforge.backends.segment.base import LayerMask, SegmentHints
from layerforge.ops.inventory import build_inventory
from layerforge.ops.usable import usable
from layerforge.taxonomy import Taxonomy, load_taxonomy


class CascadeSegment:
    """AniSeg → WDTagger inventory → DINO box → SAM3/SAM2 mask → usable retry.

    Closed role set. Do not assume pose. Never average SAM. Max 4 attempts / role.
    Imagine parts skip detect 1–4 and only check / recut failures.
    """

    name = "cascade"

    def __init__(
        self,
        cfg: dict,
        taxonomy: Taxonomy | None = None,
        *,
        character=None,
        tagger=None,
        boxes=None,
        sam3=None,
        sam2=None,
        dry_run: bool = False,
    ) -> None:
        self.cfg = cfg
        self.taxonomy = taxonomy or load_taxonomy()
        self.character = character
        self.tagger = tagger
        self.boxes = boxes
        self.sam3 = sam3
        self.sam2 = sam2
        self.dry_run = dry_run
        cascade_cfg = cfg.get("cascade") or {}
        self.max_attempts = int(cascade_cfg.get("max_attempts", 4))
        self.dino_threshold = float(cascade_cfg.get("dino_threshold", 0.25))
        self.tag_threshold = float(cascade_cfg.get("tag_threshold", 0.35))
        self.missing: list[str] = []
        self.needs_click: list[str] = []
        self.inventory: list[str] = []

    def segment(self, image: np.ndarray, hints: SegmentHints) -> list[LayerMask]:
        self.missing = []
        self.needs_click = []
        self.inventory = []
        if hints.parts:
            return self._from_parts(image, hints.parts)
        return self._from_flat(image)

    def _from_flat(self, image: np.ndarray) -> list[LayerMask]:
        character = self._cut_character(image)
        tags = self._tag(image)
        inventory = build_inventory(tags, self.taxonomy, self.tag_threshold)
        self.inventory = list(inventory)
        self._prepare_sam2(image)
        kept: list[LayerMask] = []
        skip_pair: set[str] = set()
        ordered = sorted(
            [role for role in inventory if role != "body"],
            key=lambda role: -self.taxonomy.spec(role).cut_priority,
        )
        for role in ordered:
            family = self.taxonomy.family_of_role(role)
            if family:
                if family in skip_pair:
                    continue
                skip_pair.add(family)
                left, right = self.taxonomy.pair_roles[family]
                if left in inventory:
                    layer = self._cut_one(image, character, left, kept)
                    if layer is not None:
                        kept.append(layer)
                    elif self.taxonomy.spec(left).required:
                        self._mark_missing(left)
                if right in inventory:
                    layer = self._cut_one(image, character, right, kept)
                    if layer is not None:
                        kept.append(layer)
                    elif self.taxonomy.spec(right).required:
                        self._mark_missing(right)
                continue
            if role == "acc":
                kept.extend(self._cut_acc(image, character, kept))
                continue
            layer = self._cut_one(image, character, role, kept)
            if layer is not None:
                kept.append(layer)
            elif self.taxonomy.spec(role).required:
                self._mark_missing(role)
        if "body" in inventory:
            body = self._body_from_residual(character, kept, image)
            if body is not None:
                kept.append(body)
            else:
                self._mark_missing("body")
        for spec in self.taxonomy.roles.values():
            if spec.required and spec.name not in {layer.role for layer in kept}:
                self._mark_missing(spec.name)
        return kept

    def _from_parts(self, image: np.ndarray, parts: list[LayerMask]) -> list[LayerMask]:
        character = np.zeros(image.shape[:2], dtype=np.uint8)
        for part in parts:
            character = np.maximum(character, (part.visible > 0).astype(np.uint8) * 255)
        if int((character > 0).sum()) < 32:
            character[:] = 255
        recut = not self.dry_run and self.sam2 is not None
        self.inventory = [part.role for part in parts]
        if recut:
            self._prepare_sam2(image)
        kept: list[LayerMask] = []
        for part in parts:
            spec = self.taxonomy.spec(part.role)
            box = _xyxy_from_wh(part.bbox)
            result = usable(part.visible, spec, character, kept, box)
            if result.ok:
                layer = replace(part, visible=result.mask, score=result.score)
                kept.append(layer)
                continue
            if recut:
                replacement = self._cut_one(image, character, part.role, kept)
                if replacement is not None:
                    replacement.rgba = part.rgba
                    replacement.source = "sam"
                    kept.append(replacement)
                    continue
            part.notes = _join_notes(part.notes, f"unusable:{','.join(result.reasons)}; needs_click")
            part.needs_click = True
            part.visible = result.mask if int((result.mask > 0).sum()) else part.visible
            self.needs_click.append(part.role)
            kept.append(part)
        required = [name for name, spec in self.taxonomy.roles.items() if spec.required]
        have = {layer.role for layer in kept}
        for role in required:
            if role not in have:
                if recut:
                    layer = self._cut_one(image, character, role, kept)
                    if layer is not None:
                        kept.append(layer)
                        continue
                self._mark_missing(role)
        return kept

    def _cut_character(self, image: np.ndarray) -> np.ndarray:
        if self.character is None:
            raise RuntimeError("cascade needs a CharacterCut backend (AniSeg).")
        mask = self.character.cut(image)
        if int((mask > 0).sum()) < 64:
            raise RuntimeError("AniSeg produced an empty character mask.")
        return mask

    def _tag(self, image: np.ndarray) -> dict[str, float]:
        if self.tagger is None:
            raise RuntimeError("cascade needs a Tagger backend (WDTagger).")
        return self.tagger.tag(image)

    def _prepare_sam2(self, image: np.ndarray) -> None:
        if self.sam2 is not None and hasattr(self.sam2, "prepare"):
            self.sam2.prepare(image[:, :, :3])

    def _cut_one(
        self,
        image: np.ndarray,
        character: np.ndarray,
        role: str,
        others: list[LayerMask],
    ) -> LayerMask | None:
        spec = self.taxonomy.spec(role)
        queries = list(spec.queries) or [role.replace("_", " ")]
        attempts = 0
        last_reasons: list[str] = []
        box: list[float] | None = None
        threshold = self.dino_threshold

        if self.sam3 is not None:
            for query in queries[:2]:
                if attempts >= self.max_attempts:
                    break
                attempts += 1
                mask = self.sam3.predict_text(image, query, character)
                if mask is None:
                    last_reasons.append(f"sam3_empty:{query}")
                    continue
                result = usable(mask, spec, character, others, None)
                last_reasons = result.reasons
                if result.ok:
                    return _layer(role, result.mask, result.score, f"sam3.text query={query}")
                last_reasons.append(f"sam3_unusable:{query}")

        if self.boxes is None or self.sam2 is None:
            return None
        for query in queries:
            if attempts >= self.max_attempts:
                break
            attempts += 1
            detections = self.boxes.detect(image, [query], threshold)
            if not detections:
                threshold = max(0.08, threshold * 0.7)
                detections = self.boxes.detect(image, [query], threshold)
            if not detections:
                last_reasons.append(f"no_box:{query}")
                continue
            box = _pick_box(detections, character, role)
            pos = [_box_center(box)]
            neg = _neighbor_negatives(others, spec.exclude_roles)
            mask, sam_score = self.sam2.predict_box_points(box, pos, neg)
            if mask is None:
                last_reasons.append(f"sam2_empty:{query}")
                continue
            mask = ((mask > 0) & (character > 0)).astype(np.uint8) * 255
            result = usable(mask, spec, character, others, box)
            last_reasons = result.reasons
            if result.ok:
                return _layer(
                    role,
                    result.mask,
                    min(result.score, float(sam_score)),
                    f"sam2.box query={query}; score={sam_score:.3f}",
                )
            last_reasons.append(f"sam2_unusable:{query}")
        self.needs_click.append(role)
        _ = last_reasons
        return None

    def _cut_acc(
        self,
        image: np.ndarray,
        character: np.ndarray,
        others: list[LayerMask],
    ) -> list[LayerMask]:
        spec = self.taxonomy.spec("acc")
        if self.boxes is None or self.sam2 is None:
            return []
        detections = self.boxes.detect(image, list(spec.queries), self.dino_threshold)
        layers: list[LayerMask] = []
        for _query, xyxy, _score in detections[:8]:
            pos = [_box_center(xyxy)]
            neg = _neighbor_negatives(others + layers, spec.exclude_roles)
            mask, sam_score = self.sam2.predict_box_points(xyxy, pos, neg)
            if mask is None:
                continue
            mask = ((mask > 0) & (character > 0)).astype(np.uint8) * 255
            result = usable(mask, spec, character, others + layers, xyxy)
            if not result.ok:
                continue
            layers.append(
                _layer("acc", result.mask, min(result.score, float(sam_score)), "sam2.box acc")
            )
        return layers

    def _body_from_residual(
        self,
        character: np.ndarray,
        others: list[LayerMask],
        image: np.ndarray,
    ) -> LayerMask | None:
        claimed = np.zeros(character.shape, dtype=bool)
        for layer in others:
            if self.taxonomy.spec(layer.role).overlay:
                continue
            claimed |= layer.visible > 0
        residual = (character > 0) & ~claimed
        mask = residual.astype(np.uint8) * 255
        spec = self.taxonomy.spec("body")
        result = usable(mask, spec, character, others, None)
        if result.ok:
            return _layer("body", result.mask, result.score, "character residual")
        if self.boxes is None or self.sam2 is None:
            if int(residual.sum()) >= 64:
                return _layer("body", mask, 0.4, "character residual weak")
            return None
        layer = self._cut_one(image, character, "body", others)
        return layer

    def _mark_missing(self, role: str) -> None:
        if role not in self.missing:
            self.missing.append(role)
        if role not in self.needs_click:
            self.needs_click.append(role)


def _layer(role: str, mask: np.ndarray, score: float, notes: str) -> LayerMask:
    return LayerMask(
        role=role,
        label=role,
        visible=mask,
        source="sam",
        notes=notes,
        score=float(score),
    )


def _join_notes(existing: str, extra: str) -> str:
    if existing:
        return f"{existing}; {extra}"
    return extra


def _xyxy_from_wh(bbox: list[int]) -> list[float]:
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


def _box_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _pick_box(detections, character: np.ndarray, role: str) -> list[float]:
    boxes = [item[1] for item in detections]
    if role.endswith("_l") and len(boxes) > 1:
        boxes = sorted(boxes, key=lambda b: (b[0] + b[2]) / 2.0)
        return boxes[0]
    if role.endswith("_r") and len(boxes) > 1:
        boxes = sorted(boxes, key=lambda b: (b[0] + b[2]) / 2.0)
        return boxes[-1]
    best = boxes[0]
    char = character > 0
    best_hit = -1.0
    for box in boxes:
        x0, y0, x1, y1 = [int(v) for v in box]
        h, w = char.shape
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        hit = float(char[y0:y1, x0:x1].mean())
        if hit > best_hit:
            best_hit = hit
            best = box
    return best


def _neighbor_negatives(others: list[LayerMask], exclude_roles: tuple[str, ...]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for layer in others:
        if layer.role not in exclude_roles:
            continue
        ys, xs = np.where(layer.visible > 0)
        if xs.size == 0:
            continue
        points.append((float(xs.mean()), float(ys.mean())))
    return points[:6]

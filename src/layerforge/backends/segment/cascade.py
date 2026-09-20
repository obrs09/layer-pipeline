from __future__ import annotations

from dataclasses import replace

import numpy as np

from layerforge.backends.segment.base import LayerMask, SegmentHints
from layerforge.ops.character_qa import (
    choose_peer_character,
    evaluate_character_qa,
    peers_agree_isnet_outlier,
)
from layerforge.ops.face_split import split_face_colors
from layerforge.ops.inventory import build_inventory
from layerforge.ops.morph import dilate_mask, morph_open
from layerforge.ops.usable import usable
from layerforge.taxonomy import Taxonomy, _norm_tag, load_taxonomy


class CascadeSegment:
    """anime-segmentation → WDTagger → DINO → SAM3/SAM2. DWPose is hints only.

    Character cut is isnet. Pose never overwrites that mask. Body is character
    residual after clothes/hair, not leftover rims. Never average SAM.
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
        pose=None,
        character_peers=None,
        dry_run: bool = False,
    ) -> None:
        self.cfg = cfg
        self.taxonomy = taxonomy or load_taxonomy()
        self.character = character
        self.tagger = tagger
        self.boxes = boxes
        self.sam3 = sam3
        self.sam2 = sam2
        self.pose = pose
        self.character_peers = list(character_peers or [])
        self.dry_run = dry_run
        cascade_cfg = cfg.get("cascade") or {}
        self.max_attempts = int(cascade_cfg.get("max_attempts", 4))
        self.dino_threshold = float(cascade_cfg.get("dino_threshold", 0.25))
        self.tag_threshold = float(cascade_cfg.get("tag_threshold", 0.35))
        refine_cfg = cfg.get("refine") or {}
        self.morph_open_px = int(refine_cfg.get("morph_open_px", 2))
        self.hair_reach = float(cascade_cfg.get("hair_reach", 0.6))
        self.hair_punch_pose_roles = tuple(
            cascade_cfg.get("hair_punch_pose_roles") or ("arm_l", "arm_r")
        )
        self.hair_punch_pose_pad = int(cascade_cfg.get("hair_punch_pose_pad", 48))
        self.missing: list[str] = []
        self.needs_click: list[str] = []
        self.inventory: list[str] = []
        self.tags: dict[str, float] = {}
        self.dump = None
        self.debug_boxes: list[dict] = []
        self.cut_plan: list[str] = []
        self.cut_failures: list[dict] = []
        self.character_mask = None
        self.character_qa = None
        self.pose_person = None
        self.pose_boxes: dict[str, list[float]] = {}
        self.pose_points: dict[str, list[tuple[float, float]]] = {}
        self._last_isnet_mask = None
        self._last_peer_masks = {}

    def bind_dump(self, dump) -> None:
        self.dump = dump

    def segment(self, image: np.ndarray, hints: SegmentHints) -> list[LayerMask]:
        self.missing = []
        self.needs_click = []
        self.inventory = []
        self.tags = {}
        self.debug_boxes = []
        self.cut_plan = []
        self.cut_failures = []
        self.character_mask = None
        self.character_qa = None
        self.pose_person = None
        self.pose_boxes = {}
        self.pose_points = {}
        self._last_isnet_mask = None
        self._last_peer_masks = {}
        if hints.parts:
            return self._from_parts(image, hints.parts)
        return self._from_flat(image)

    def _from_flat(self, image: np.ndarray) -> list[LayerMask]:
        character = self._cut_character(image)
        self._dump_character(image, character)
        self._apply_pose_hints(image)
        tags = self._tag(image)
        self.tags = dict(tags)
        inventory = build_inventory(tags, self.taxonomy, self.tag_threshold)
        self.inventory = list(inventory)
        self._prepare_sam2(image)
        kept: list[LayerMask] = []
        skip_pair: set[str] = set()
        ordered = self._cut_order(inventory)
        self.cut_plan = list(ordered)
        for role in ordered:
            if role == "hair_front":
                continue
            family = self.taxonomy.family_of_role(role)
            if family:
                if family in skip_pair:
                    continue
                skip_pair.add(family)
                left, right = self.taxonomy.pair_roles[family]
                pair_layers = self._cut_pair(image, character, family, kept)
                have = {layer.role for layer in pair_layers}
                for member in (left, right):
                    if member not in inventory or member in have:
                        continue
                    fallback = self._cut_one(image, character, member, kept + pair_layers)
                    if fallback is not None:
                        pair_layers.append(fallback)
                        have.add(member)
                kept.extend(pair_layers)
                for member in (left, right):
                    if member not in inventory:
                        continue
                    if member in have:
                        continue
                    if self.taxonomy.spec(member).required:
                        self._mark_missing(member)
                    elif member not in self.needs_click:
                        self.needs_click.append(member)
                continue
            if role == "acc":
                kept.extend(self._cut_acc(image, character, kept))
                continue
            spec = self.taxonomy.spec(role)
            layer = self._cut_one(image, character, role, kept)
            if layer is not None:
                kept.append(layer)
                if role == "face":
                    self._punch_face_from_hair(kept)
                    self._split_hair_front(kept)
            elif spec.required or spec.required_if_tags:
                self._mark_missing(role)
        hair = self._hair_from_residual(character, kept)
        if hair is not None:
            kept.append(hair)
            for bucket in (self.missing, self.needs_click):
                if "hair_back" in bucket:
                    bucket.remove("hair_back")
        self._punch_face_from_hair(kept)
        self._split_hair_front(kept)
        if "hair_front" in inventory and not any(layer.role == "hair_front" for layer in kept):
            self._mark_missing("hair_front")
        if "body" in inventory:
            body = self._body_from_residual(character, kept, image)
            if body is not None:
                kept.append(body)
            else:
                self._mark_missing("body")
        for spec in self.taxonomy.roles.values():
            if spec.required and spec.name not in {layer.role for layer in kept}:
                self._mark_missing(spec.name)
        kept = self._apply_face_split(image, kept)
        self._dump_boxes(image)
        self._dump_failures()
        return kept

    def _cut_order(self, inventory: list[str]) -> list[str]:
        """Clothes, then whole hair, then face. Front hair is split later, not SAM-cut."""
        overlay: list[str] = []
        mid: list[str] = []
        for role in inventory:
            if role == "body" or role == "hair_front":
                continue
            spec = self.taxonomy.spec(role)
            if not spec.queries and not spec.overlay:
                continue
            if spec.overlay:
                overlay.append(role)
            else:
                mid.append(role)

        def mid_key(role: str):
            spec = self.taxonomy.spec(role)
            if role == "hair_back":
                return (1, spec.order)
            if role == "face":
                return (2, spec.order)
            return (0, spec.order)

        mid.sort(key=mid_key)
        overlay.sort(key=lambda role: -self.taxonomy.spec(role).cut_priority)
        return mid + overlay

    def _from_parts(self, image: np.ndarray, parts: list[LayerMask]) -> list[LayerMask]:
        character = np.zeros(image.shape[:2], dtype=np.uint8)
        for part in parts:
            character = np.maximum(character, (part.visible > 0).astype(np.uint8) * 255)
        if int((character > 0).sum()) < 32:
            character[:] = 255
        self.character_mask = character
        self._dump_character(image, character)
        recut = not self.dry_run and self.sam2 is not None
        self.inventory = [part.role for part in parts]
        self.cut_plan = list(self.inventory)
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
        self._punch_face_from_hair(kept)
        self._split_hair_front(kept)
        kept = self._apply_face_split(image, kept)
        self._dump_boxes(image)
        self._dump_failures()
        return kept

    def _cut_character(self, image: np.ndarray) -> np.ndarray:
        if self.character is None:
            raise RuntimeError("cascade needs a CharacterCut backend (anime-segmentation).")
        qa_cfg = self.cfg.get("character_qa") or {}
        max_attempts = int(qa_cfg.get("max_attempts", 3))
        attempts: list[dict] = []
        last_mask = None
        last_prob = None
        peers = self._peer_masks(image)
        flagged = False
        for seed in range(max(1, max_attempts)):
            if hasattr(self.character, "predict"):
                mask, prob = self.character.predict(image, seed=seed)
            else:
                mask = self.character.cut(image)
                prob = (mask > 0).astype(np.float32)
            report = evaluate_character_qa(mask, prob, peers, qa_cfg)
            attempts.append(
                {
                    "seed": seed,
                    "ok": report.ok,
                    "reasons": list(report.reasons),
                    "metrics": report.metrics,
                }
            )
            last_mask, last_prob = mask, prob
            if report.ok:
                flagged = False
                break
            flagged = True
            if peers_agree_isnet_outlier(mask, peers, qa_cfg):
                # Another threshold/noise variant cannot close a structural gap.
                attempts[-1]["skipped_retries"] = "peers agree with each other; isnet is the outlier"
                break
        if last_mask is None or int((last_mask > 0).sum()) < 64:
            raise RuntimeError("anime-segmentation produced an empty character mask.")
        source = "anime_segmentation"
        peer_replace = None
        isnet_mask = last_mask
        choice = choose_peer_character(isnet_mask, peers, qa_cfg, prob=last_prob)
        if choice is not None:
            last_mask = choice["mask"]
            last_prob = (last_mask > 0).astype(np.float32)
            source = str(choice["source"])
            peer_replace = {key: value for key, value in choice.items() if key != "mask"}
            flagged = not bool((choice.get("qa") or {}).get("ok", True))
            if not flagged and "character" in self.needs_click:
                self.needs_click.remove("character")
        self.character_mask = last_mask
        self.character_qa = {
            "flagged": flagged,
            "chosen_seed": attempts[-1]["seed"] if attempts else 0,
            "chosen_source": source,
            "attempts": attempts,
            "peer_replace": peer_replace,
        }
        if flagged and "character" not in self.needs_click:
            self.needs_click.append("character")
        self._last_character_prob = last_prob
        self._last_isnet_mask = isnet_mask
        self._last_peer_masks = peers
        return last_mask

    def _peer_masks(self, image: np.ndarray) -> dict[str, np.ndarray | None]:
        out: dict[str, np.ndarray | None] = {}
        for peer in self.character_peers:
            try:
                out[peer.name] = peer.cut(image)
            except Exception:
                out[peer.name] = None
        return out

    def _apply_pose_hints(self, image: np.ndarray) -> None:
        """Skeleton boxes/points for SAM. Does not change the isnet character mask."""
        if self.pose is None:
            return
        estimate = self.pose.estimate(image)
        if estimate is None:
            return
        if self.dump is not None:
            self.dump.write_pose(image, estimate)
        self.pose_person = estimate.person_mask
        self.pose_boxes = dict(estimate.boxes or {})
        self.pose_points = dict(estimate.points or {})

    def _tag(self, image: np.ndarray) -> dict[str, float]:
        if self.tagger is None:
            raise RuntimeError("cascade needs a Tagger backend (WDTagger).")
        return self.tagger.tag(image)

    def _prepare_sam2(self, image: np.ndarray) -> None:
        if self.sam2 is not None and hasattr(self.sam2, "prepare"):
            self.sam2.prepare(image[:, :, :3])

    def _sam_predict(
        self,
        role: str,
        box: list[float],
        pos: list[tuple[float, float]],
        neg: list[tuple[float, float]],
    ) -> tuple[np.ndarray | None, float]:
        crop_cfg = (self.cfg.get("cascade") or {}).get("sam_crop") or {}
        roles = set(crop_cfg.get("roles") or ("face",))
        use_crop = bool(crop_cfg.get("enabled", True)) and role in roles
        if use_crop and hasattr(self.sam2, "predict_on_crop"):
            return self.sam2.predict_on_crop(
                box,
                pos,
                neg,
                pad_frac=float(crop_cfg.get("pad_frac", 0.18)),
                min_side=int(crop_cfg.get("min_side", 1024)),
            )
        return self.sam2.predict_box_points(box, pos, neg)

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
                if not spec.overlay:
                    mask = morph_open(mask, self.morph_open_px)
                result = usable(mask, spec, character, others, None)
                last_reasons = result.reasons
                if result.ok:
                    return _layer(role, result.mask, result.score, f"sam3.text query={query}")
                last_reasons.append(f"sam3_unusable:{query}")

        if self.sam2 is None:
            self._record_failure(role, last_reasons)
            return None
        for query in queries:
            if attempts >= self.max_attempts:
                break
            attempts += 1
            detections: list = []
            if self.boxes is not None:
                detections = self.boxes.detect(image, [query], threshold) or []
                if not detections:
                    threshold = max(0.08, threshold * 0.7)
                    detections = self.boxes.detect(image, [query], threshold) or []
            if detections:
                if role.endswith("_l") or role.endswith("_r"):
                    ranked = [_pick_box(detections, character, role)]
                else:
                    ranked = _rank_boxes(detections, character, spec)
            else:
                ranked = []
                last_reasons.append(f"no_box:{query}")
            ranked = self._inject_pose_box(role, ranked)
            if not ranked:
                continue
            for box in ranked[:2]:
                self._trace_box(role, query, box, detections)
                pos = [_box_center(box)]
                pos.extend(self.pose_points.get(role) or [])
                avoid = box if role == "face" else None
                neg = _neighbor_negatives(others, spec.exclude_roles, avoid_box=avoid)
                mask, sam_score = self._sam_predict(role, box, pos, neg)
                if mask is None:
                    last_reasons.append(f"sam2_empty:{query}")
                    continue
                mask = _constrain_mask(mask, character, box)
                if not spec.overlay:
                    mask = morph_open(mask, self.morph_open_px)
                result = usable(mask, spec, character, others, box)
                last_reasons = list(result.reasons)
                if result.ok:
                    return _layer(
                        role,
                        result.mask,
                        min(result.score, float(sam_score)),
                        f"sam2.box query={query}; score={sam_score:.3f}",
                    )
                last_reasons.append(f"sam2_unusable:{query}")
        self.needs_click.append(role)
        self._record_failure(role, last_reasons)
        return None

    def _cut_pair(
        self,
        image: np.ndarray,
        character: np.ndarray,
        family: str,
        others: list[LayerMask],
    ) -> list[LayerMask]:
        left, right = self.taxonomy.pair_roles[family]
        spec_l = self.taxonomy.spec(left)
        spec_r = self.taxonomy.spec(right)
        detections: list[tuple[str, list[float], float]] = []
        threshold = self.dino_threshold
        query_groups = [
            list(self.taxonomy.pair_queries.get(family, (family,))),
            list(spec_l.queries),
            list(spec_r.queries),
        ]
        if self.boxes is not None:
            for group in query_groups:
                if not group:
                    continue
                detections.extend(self.boxes.detect(image, group, threshold) or [])
            face_box = self._face_anchor(image, character, others)
            if face_box is not None:
                detections.extend(
                    self._detect_in_box(
                        image,
                        face_box,
                        list(self.taxonomy.pair_queries.get(family, (family,))),
                        threshold,
                    )
                )
            boxes = _filter_boxes_for_spec(
                _spatial_pair(_boxes_on_character(_nms_boxes(detections), character)),
                character,
                spec_l,
            )
            if len(boxes) < 2:
                threshold = max(0.08, threshold * 0.7)
                extra: list[tuple[str, list[float], float]] = []
                for group in query_groups:
                    if not group:
                        continue
                    extra.extend(self.boxes.detect(image, group, threshold) or [])
                if face_box is not None:
                    extra.extend(
                        self._detect_in_box(
                            image,
                            face_box,
                            list(self.taxonomy.pair_queries.get(family, (family,))),
                            threshold,
                        )
                    )
                detections = detections + extra
        else:
            face_box = self._face_anchor(image, character, others)
        boxes = _filter_boxes_for_spec(
            _spatial_pair(_boxes_on_character(_nms_boxes(detections), character)),
            character,
            spec_l,
        )
        if face_box is None:
            face_box = self._face_anchor(image, character, others)
        if len(boxes) == 1:
            cx = _box_center(face_box)[0] if face_box else _mask_center_x(character)
            if cx is not None:
                mirrored = _mirror_box(boxes[0], cx, character.shape)
                if face_box is not None:
                    mirrored = _clip_box(mirrored, face_box)
                if _box_iou(mirrored, boxes[0]) < 0.45 and _box_area(mirrored) > 4:
                    boxes = _spatial_pair(boxes + [mirrored])
        assigned: dict[str, list[float]] = {}
        if len(boxes) >= 2:
            ordered = sorted(boxes, key=lambda box: (box[0] + box[2]) / 2.0)
            assigned[left] = ordered[0]
            assigned[right] = ordered[-1]
        elif len(boxes) == 1:
            cx = _box_center(face_box)[0] if face_box else (_mask_center_x(character) or 0.0)
            box = boxes[0]
            assigned[left if _box_center(box)[0] <= cx else right] = box
        if self.sam2 is None:
            return []
        inventory = set(self.inventory)
        layers: list[LayerMask] = []
        for role, box in assigned.items():
            if role not in inventory:
                continue
            self._trace_box(role, family, box, detections)
            spec = self.taxonomy.spec(role)
            pos = [_box_center(box)]
            neg = _neighbor_negatives(others + layers, spec.exclude_roles)
            for other_role, other_box in assigned.items():
                if other_role != role:
                    neg.append(_box_center(other_box))
            mask, sam_score = self._sam_predict(role, box, pos, neg)
            if mask is None:
                if role not in self.needs_click:
                    self.needs_click.append(role)
                continue
            mask = _constrain_mask(mask, character, box)
            result = usable(mask, spec, character, others + layers, box)
            if result.ok:
                layers.append(
                    _layer(
                        role,
                        result.mask,
                        min(result.score, float(sam_score)),
                        f"sam2.pair family={family}; score={sam_score:.3f}",
                    )
                )
            else:
                self._record_failure(role, result.reasons)
                if role not in self.needs_click:
                    self.needs_click.append(role)
        return layers

    def _face_anchor(
        self,
        image: np.ndarray,
        character: np.ndarray,
        others: list[LayerMask],
    ) -> list[float] | None:
        for layer in others:
            if layer.role != "face":
                continue
            if int((layer.visible > 0).sum()) == 0:
                continue
            return _xyxy_from_wh(layer.bbox)
        if self.boxes is None:
            return None
        detections = self.boxes.detect(image, ["anime face", "face"], self.dino_threshold) or []
        if not detections:
            return None
        return _pick_box(detections, character, "face")

    def _detect_in_box(
        self,
        image: np.ndarray,
        box: list[float],
        queries: list[str],
        threshold: float,
    ) -> list[tuple[str, list[float], float]]:
        if self.boxes is None or not queries:
            return []
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        h, w = image.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return []
        crop = image[y0:y1, x0:x1]
        detections = self.boxes.detect(crop, queries, threshold) or []
        crop_h, crop_w = crop.shape[:2]
        remapped: list[tuple[str, list[float], float]] = []
        for label, xyxy, score in detections:
            if xyxy[2] <= crop_w + 1.5 and xyxy[3] <= crop_h + 1.5:
                remapped.append(
                    (
                        label,
                        [xyxy[0] + x0, xyxy[1] + y0, xyxy[2] + x0, xyxy[3] + y0],
                        float(score),
                    )
                )
            else:
                remapped.append((label, [float(v) for v in xyxy], float(score)))
        return remapped

    def _cut_acc(
        self,
        image: np.ndarray,
        character: np.ndarray,
        others: list[LayerMask],
    ) -> list[LayerMask]:
        spec = self.taxonomy.spec("acc")
        if self.boxes is None or self.sam2 is None:
            return []
        thresh = spec.tag_threshold if spec.tag_threshold > 0 else self.tag_threshold
        scores = {_norm_tag(name): float(score) for name, score in self.tags.items()}
        base_gate = spec.required_if_tags or spec.tag_names
        queries: list[str] = []
        if not base_gate or any(scores.get(tag, 0.0) >= thresh for tag in base_gate):
            queries.extend(spec.queries)
        queries.extend(q for q in spec.fired_queries(self.tags, thresh) if q not in queries)
        if not queries:
            return []
        detections = self.boxes.detect(image, queries, self.dino_threshold)
        layers: list[LayerMask] = []
        for query, xyxy, _score in detections[:8]:
            self._trace_box("acc", query, xyxy, detections)
            pos = [_box_center(xyxy)]
            neg = _neighbor_negatives(others + layers, spec.exclude_roles)
            mask, sam_score = self._sam_predict("acc", xyxy, pos, neg)
            if mask is None:
                continue
            mask = _constrain_mask(mask, character, xyxy)
            result = usable(mask, spec, character, others + layers, xyxy)
            if not result.ok:
                self._record_failure("acc", [f"query={query}", *result.reasons])
                continue
            layers.append(
                _layer("acc", result.mask, min(result.score, float(sam_score)), f"sam2.box acc query={query}")
            )
        return layers

    def _hair_from_residual(
        self,
        character: np.ndarray,
        others: list[LayerMask],
    ) -> LayerMask | None:
        """Hair the detectors wanted but SAM could not cut: leftover character pixels around the head.

        Only runs when a hair role is in the inventory and no hair layer exists. Keeps
        residual components that touch the head zone (face grown by hair_reach). Hair
        DINO boxes only clip the search region so long strands can be considered;
        sitting inside a box is not enough (a staff centroid in a hair box is not hair).
        Pose arm boxes punch held props out of the leftover blob before components run.
        """
        have = {layer.role for layer in others}
        wanted = [
            role
            for role in ("hair_back", "hair_front")
            if role in self.inventory and role not in have
        ]
        if not wanted:
            return None
        faces = [layer for layer in others if layer.role == "face" and int((layer.visible > 0).sum()) > 0]
        spec = self.taxonomy.spec("hair_back")
        hair_boxes = [
            item["xyxy"]
            for item in self.debug_boxes
            if item.get("role") in ("hair_back", "hair_front")
            and item.get("xyxy")
            and _box_character_frac(item["xyxy"], character) <= spec.max_area_frac
        ]
        if not faces and not hair_boxes:
            return None
        claimed = np.zeros(character.shape, dtype=bool)
        for layer in others:
            if self.taxonomy.spec(layer.role).overlay:
                continue
            claimed |= layer.visible > 0
        residual = morph_open(((character > 0) & ~claimed).astype(np.uint8) * 255, self.morph_open_px) > 0
        residual &= ~self._pose_hair_punch(character.shape)
        if not residual.any():
            return None
        head_zone = np.zeros(character.shape, dtype=bool)
        for face in faces:
            _x, _y, _w, face_h = face.bbox
            reach = max(8, int(round(face_h * self.hair_reach)))
            head_zone |= dilate_mask(face.visible, reach) > 0
        # Clip to the head halo plus traced hair boxes so a hem rim cannot chain in.
        # Boxes do not grant membership: a staff inside a hair box still needs to touch the head.
        region = head_zone.copy()
        h, w = character.shape
        for x0, y0, x1, y1 in hair_boxes:
            x0, y0 = max(0, int(x0) - 12), max(0, int(y0) - 12)
            x1, y1 = min(w, int(x1) + 12), min(h, int(y1) + 12)
            if x1 > x0 and y1 > y0:
                region[y0:y1, x0:x1] = True
        candidate = residual & region
        import cv2

        n, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8), connectivity=8
        )
        hair = np.zeros(character.shape, dtype=bool)
        char_area = max(1, int((character > 0).sum()))
        for k in range(1, n):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < max(64, int(0.002 * char_area)):
                continue
            comp = labels == k
            if (comp & head_zone).any():
                hair |= comp
        if int(hair.sum()) < 64:
            return None
        result = usable(hair.astype(np.uint8) * 255, spec, character, others, None)
        if not result.ok:
            self._record_failure("hair_back", ["residual", *result.reasons])
            return None
        return _layer("hair_back", result.mask, 0.5, "character residual around head; SAM hair cut failed")

    def _pose_hair_punch(self, shape: tuple[int, int]) -> np.ndarray:
        """Wrist discs only. Full arm boxes swallow long hair (640 arm_l covers the right stream)."""
        h, w = shape
        punch = np.zeros((h, w), dtype=bool)
        r = max(8, self.hair_punch_pose_pad)
        for role in self.hair_punch_pose_roles:
            pts = list(self.pose_points.get(role) or [])
            if not pts:
                continue
            x, y = pts[-1]
            cx, cy = int(round(x)), int(round(y))
            x0, y0 = max(0, cx - r), max(0, cy - r)
            x1, y1 = min(w, cx + r + 1), min(h, cy + r + 1)
            if x1 > x0 and y1 > y0:
                punch[y0:y1, x0:x1] = True
        return punch

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
        if self.morph_open_px > 0:
            claimed = dilate_mask(claimed.astype(np.uint8) * 255, self.morph_open_px) > 0
        residual = (character > 0) & ~claimed
        mask = residual.astype(np.uint8) * 255
        mask = morph_open(mask, self.morph_open_px)
        spec = self.taxonomy.spec("body")
        result = usable(mask, spec, character, others, self.pose_boxes.get("body"))
        notes = "character residual"
        if result.ok:
            return _layer("body", result.mask, result.score, notes)
        if self.sam2 is not None:
            layer = self._cut_one(image, character, "body", others)
            if layer is not None:
                punched = (layer.visible > 0) & ~claimed
                layer.visible = morph_open(punched.astype(np.uint8) * 255, self.morph_open_px)
                if int((layer.visible > 0).sum()) >= 64:
                    layer.notes = f"{layer.notes}; {notes}"
                    return layer
        if int((mask > 0).sum()) >= 64:
            return _layer("body", mask, 0.4, f"{notes} weak")
        return None

    def _inject_pose_box(self, role: str, ranked: list[list[float]]) -> list[list[float]]:
        box = self.pose_boxes.get(role)
        if not box:
            return ranked
        key = tuple(round(float(v), 1) for v in box)
        rest = [item for item in ranked if tuple(round(float(v), 1) for v in item) != key]
        return [[float(v) for v in box], *rest]

    def _mark_missing(self, role: str) -> None:
        if role not in self.missing:
            self.missing.append(role)
        if role not in self.needs_click:
            self.needs_click.append(role)

    def _punch_face_from_hair(self, kept: list[LayerMask]) -> None:
        face = next((layer for layer in kept if layer.role == "face" and int((layer.visible > 0).sum()) > 0), None)
        if face is None:
            return
        face_vis = face.visible > 0
        for layer in kept:
            if layer.role != "hair_back":
                continue
            remain = (layer.visible > 0) & ~face_vis
            if int(remain.sum()) < 16:
                continue
            layer.visible = remain.astype(np.uint8) * 255
            layer.notes = _join_notes(layer.notes, "punched face")

    def _split_hair_front(self, kept: list[LayerMask]) -> None:
        """Peel bangs from whole hair by overlap with the face, not by color."""
        split_cfg = (self.cfg.get("cascade") or {}).get("hair_split") or {}
        if not bool(split_cfg.get("enabled", True)):
            return
        if "hair_front" not in set(self.inventory):
            return
        if any(layer.role == "hair_front" and int((layer.visible > 0).sum()) > 0 for layer in kept):
            return
        hair = next((layer for layer in kept if layer.role == "hair_back" and int((layer.visible > 0).sum()) > 0), None)
        face = next((layer for layer in kept if layer.role == "face" and int((layer.visible > 0).sum()) > 0), None)
        if hair is None or face is None:
            return
        dilate_px = int(split_cfg.get("face_dilate_px", 8))
        up_frac = float(split_cfg.get("up_frac", 0.45))
        height_frac = float(split_cfg.get("face_height_frac", 0.55))
        min_front = int(split_cfg.get("min_front_px", 32))
        min_back = int(split_cfg.get("min_back_px", 64))
        zone = dilate_mask(face.visible, dilate_px) > 0
        x, y, bw, bh = face.bbox
        h, w = hair.visible.shape
        y0 = max(0, int(round(y - bh * up_frac)))
        y1 = min(h, int(round(y + bh * height_frac)))
        x0 = max(0, int(round(x - bw * 0.08)))
        x1 = min(w, int(round(x + bw * 1.08)))
        if x1 > x0 and y1 > y0:
            zone[y0:y1, x0:x1] = True
        hair_vis = hair.visible > 0
        front = hair_vis & zone
        back = hair_vis & ~zone
        if int(front.sum()) < min_front:
            return
        if int(back.sum()) >= min_back:
            hair.visible = back.astype(np.uint8) * 255
            hair.notes = _join_notes(hair.notes, "hair_split remainder")
        kept.append(
            _layer(
                "hair_front",
                front.astype(np.uint8) * 255,
                hair.score,
                "hair_split whole hair ∩ face zone",
            )
        )
        for bucket in (self.missing, self.needs_click):
            if "hair_front" in bucket:
                bucket.remove("hair_front")
        if self.dump is not None:
            self.dump.write_json(
                "04_segment/hair_split.json",
                {
                    "applied": True,
                    "front_px": int(front.sum()),
                    "back_px": int(back.sum()),
                    "peeled": int(back.sum()) >= min_back,
                },
            )

    def _apply_face_split(self, image: np.ndarray, kept: list[LayerMask]) -> list[LayerMask]:
        split_cfg = (self.cfg.get("cascade") or {}).get("face_split") or {}
        kept, report = split_face_colors(image, kept, self.taxonomy, split_cfg)
        dest = report.get("hair_role")
        if dest and int(report.get("hair_px") or 0) > 0:
            for bucket in (self.missing, self.needs_click):
                if dest in bucket:
                    bucket.remove(dest)
        if self.dump is not None and report.get("applied"):
            self.dump.write_json("04_segment/face_split.json", report)
        return kept

    def _dump_character(self, image: np.ndarray, mask: np.ndarray) -> None:
        if self.dump is None:
            return
        flagged = bool((self.character_qa or {}).get("flagged"))
        self.dump.write_character(image, mask, flagged=flagged)
        isnet_mask = getattr(self, "_last_isnet_mask", None)
        source = (self.character_qa or {}).get("chosen_source") or "anime_segmentation"
        if isnet_mask is not None and source != "anime_segmentation":
            self.dump.write_character_raw(image, isnet_mask)
            self.dump.write_png("01_character/isnet.png", isnet_mask)
        if self.character_qa is not None:
            self.dump.write_json("01_character/qa.json", self.character_qa)
        if flagged:
            self.dump.write_json("01_character/FLAGGED.json", self.character_qa)
        peers = getattr(self, "_last_peer_masks", None) or {}
        for name, peer_mask in peers.items():
            if peer_mask is None:
                continue
            self.dump.write_png(f"01_character/peer_{name}.png", peer_mask)

    def _dump_boxes(self, image: np.ndarray) -> None:
        if self.dump is None:
            return
        self.dump.write_boxes(image, self.debug_boxes)

    def _trace_box(self, role: str, query: str, box: list[float], detections) -> None:
        score = None
        for item in detections or []:
            if len(item) < 2:
                continue
            if item[1] == box:
                score = float(item[2]) if len(item) > 2 else None
                break
        self.debug_boxes.append(
            {
                "index": len(self.debug_boxes),
                "role": role,
                "query": query,
                "xyxy": [round(float(v), 1) for v in box],
                "score": None if score is None else round(float(score), 4),
            }
        )

    def _record_failure(self, role: str, reasons: list[str]) -> None:
        self.cut_failures.append({"role": role, "reasons": list(reasons or [])})

    def _dump_failures(self) -> None:
        if self.dump is None or not self.cut_failures:
            return
        self.dump.write_json("04_segment/failures.json", self.cut_failures)


def _box_character_frac(box: list[float], character: np.ndarray) -> float:
    """Share of the character mask that falls inside the box."""
    char = character > 0
    h, w = char.shape
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(char[y0:y1, x0:x1].sum() / max(1, int(char.sum())))


def _constrain_mask(mask: np.ndarray, character: np.ndarray, box: list[float] | None, pad: int = 12) -> np.ndarray:
    vis = (mask > 0) & (character > 0)
    if box is None:
        return vis.astype(np.uint8) * 255
    h, w = vis.shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    clipped = np.zeros_like(vis)
    if x1 > x0 and y1 > y0:
        clipped[y0:y1, x0:x1] = vis[y0:y1, x0:x1]
    return clipped.astype(np.uint8) * 255


def _rank_boxes(detections, character: np.ndarray, spec) -> list[list[float]]:
    char = character > 0
    char_area = max(1, int(char.sum()))
    h, w = char.shape
    scored: list[tuple] = []
    seen: set[tuple] = set()
    for item in detections:
        box = item[1]
        score = float(item[2]) if len(item) > 2 else 0.0
        key = tuple(round(float(v), 1) for v in box)
        if key in seen:
            continue
        seen.add(key)
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        inter = int(char[y0:y1, x0:x1].sum())
        box_area = max(1, (x1 - x0) * (y1 - y0))
        hit = inter / box_area
        char_frac = inter / char_area
        too_big = char_frac > max(spec.max_area_frac * 1.5, 0.12)
        scored.append((too_big, -hit, -score, [float(v) for v in box]))
    scored.sort()
    return [item[-1] for item in scored]


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


def _neighbor_negatives(
    others: list[LayerMask],
    exclude_roles: tuple[str, ...],
    avoid_box: list[float] | None = None,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for layer in others:
        if layer.role not in exclude_roles:
            continue
        vis = layer.visible > 0
        if avoid_box is not None:
            h, w = vis.shape
            x0, y0, x1, y1 = [int(round(v)) for v in avoid_box]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 > x0 and y1 > y0 and vis.any():
                hole = vis.copy()
                hole[y0:y1, x0:x1] = False
                if hole.any():
                    vis = hole
        ys, xs = np.where(vis)
        if xs.size == 0:
            continue
        points.append((float(xs.mean()), float(ys.mean())))
    return points[:6]


def _box_iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(1e-6, area_a + area_b - inter)


def _nms_boxes(
    detections: list[tuple[str, list[float], float]],
    iou_thresh: float = 0.45,
) -> list[tuple[str, list[float], float]]:
    kept: list[tuple[str, list[float], float]] = []
    for item in sorted(detections, key=lambda det: float(det[2]), reverse=True):
        if all(_box_iou(item[1], prev[1]) < iou_thresh for prev in kept):
            kept.append(item)
    return kept


def _boxes_on_character(
    detections: list[tuple[str, list[float], float]],
    character: np.ndarray,
    min_hit: float = 0.15,
) -> list[list[float]]:
    char = character > 0
    h, w = char.shape
    out: list[list[float]] = []
    for _label, box, _score in detections:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        if float(char[y0:y1, x0:x1].mean()) >= min_hit:
            out.append([float(v) for v in box])
    return out


def _spatial_pair(boxes: list[list[float]]) -> list[list[float]]:
    if len(boxes) < 2:
        return list(boxes)
    ordered = sorted(boxes, key=lambda box: (box[0] + box[2]) / 2.0)
    left, right = ordered[0], ordered[-1]
    gap = _box_center(right)[0] - _box_center(left)[0]
    width = max(right[2] - right[0], left[2] - left[0], 1.0)
    if gap < 0.6 * width:
        return [left]
    return [left, right]


def _box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _clip_box(box: list[float], limit: list[float]) -> list[float]:
    return [
        max(box[0], limit[0]),
        max(box[1], limit[1]),
        min(box[2], limit[2]),
        min(box[3], limit[3]),
    ]


def _filter_boxes_for_spec(
    boxes: list[list[float]],
    character: np.ndarray,
    spec,
) -> list[list[float]]:
    char_area = max(1, int((character > 0).sum()))
    cap = max(spec.max_area_frac * 2.5, 0.08)
    out: list[list[float]] = []
    for box in boxes:
        if _box_area(box) / char_area > cap:
            continue
        if _box_area(box) < 4:
            continue
        out.append(box)
    return out


def _mirror_box(box: list[float], cx: float, shape: tuple[int, ...]) -> list[float]:
    x0, y0, x1, y1 = box
    width = x1 - x0
    old_cx = (x0 + x1) / 2.0
    new_cx = 2.0 * float(cx) - old_cx
    h, w = int(shape[0]), int(shape[1])
    nx0 = min(max(0.0, new_cx - width / 2.0), float(max(0, w - 1)))
    nx1 = min(max(nx0 + 1.0, new_cx + width / 2.0), float(w))
    ny0 = min(max(0.0, y0), float(max(0, h - 1)))
    ny1 = min(max(ny0 + 1.0, y1), float(h))
    return [nx0, ny0, nx1, ny1]


def _mask_center_x(mask: np.ndarray) -> float | None:
    _ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return float(xs.mean())

from __future__ import annotations

from pathlib import Path

import numpy as np

from layerforge.backends.segment.base import LayerMask, SegmentHints
from layerforge.config import resolve_path


class SamHintedSegment:
    name = "sam2.hinted"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._predictor = None
        self._generator = None
        self._backend: dict = {}

    def _cuda_or_die(self) -> str:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("sam2.hinted needs torch. Install the [gpu] extra.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("sam2.hinted needs CUDA. Refusing to run SAM on CPU.")
        return "cuda"

    def _checkpoint(self) -> Path:
        paths = self.cfg.get("paths") or {}
        explicit = paths.get("sam2_checkpoint")
        search_dir = resolve_path(paths.get("sam2_dir", "model/sam2"))
        if explicit:
            path = resolve_path(explicit)
            if path.exists():
                return path
        found = sorted(search_dir.glob("*.pt")) + sorted(search_dir.glob("*.pth"))
        if not found:
            raise RuntimeError(
                f"No SAM2 checkpoint under {search_dir}. Put a .pt file in model/sam2/."
            )
        return found[0]

    def _load(self):
        if self._predictor is not None:
            return
        device = self._cuda_or_die()
        ckpt = self._checkpoint()
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError("sam2 package is not installed.") from exc
        backend = {}
        backend_path = resolve_path("configs/backends/segment_sam2.yaml")
        if backend_path.exists():
            import yaml

            backend = yaml.safe_load(backend_path.read_text(encoding="utf-8")) or {}
        self._backend = backend
        model_cfg = backend.get("model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml")
        last_err = None
        sam = None
        for cfg_name in (
            model_cfg,
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            "sam2.1/sam2.1_hiera_l.yaml",
            "configs/sam2/sam2_hiera_l.yaml",
        ):
            try:
                sam = build_sam2(cfg_name, str(ckpt), device=device)
                break
            except Exception as exc:
                last_err = exc
        if sam is None:
            raise RuntimeError(f"Could not build SAM2 from {ckpt}: {last_err}") from last_err
        self._predictor = SAM2ImagePredictor(sam)
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            self._generator = SAM2AutomaticMaskGenerator(
                sam,
                points_per_side=int(backend.get("points_per_side", 16)),
            )
        except Exception:
            self._generator = None

    def prepare(self, rgb: np.ndarray) -> None:
        self._load()
        self._predictor.set_image(rgb[:, :, :3] if rgb.ndim == 3 and rgb.shape[2] >= 3 else rgb)

    def predict_box_points(
        self,
        box: list[float] | None,
        positive: list[tuple[float, float]] | None = None,
        negative: list[tuple[float, float]] | None = None,
    ) -> tuple[np.ndarray | None, float]:
        """One multimask pick. Never average. Same prompts are not re-run here."""
        self._load()
        kw: dict = {"multimask_output": True}
        if box is not None:
            kw["box"] = np.array(box, dtype=np.float32)
        pos = list(positive or [])
        neg = list(negative or [])
        if pos or neg:
            pts = pos + neg
            labs = [1] * len(pos) + [0] * len(neg)
            kw["point_coords"] = np.array(pts, dtype=np.float32)
            kw["point_labels"] = np.array(labs, dtype=np.int32)
        masks, scores, _ = self._predictor.predict(**kw)
        chosen = self._pick(masks, scores)
        return (chosen.astype(np.uint8) * 255), float(np.max(scores))

    def segment(self, image: np.ndarray, hints: SegmentHints) -> list[LayerMask]:
        self.prepare(image)
        rgb = image[:, :, :3]
        if hints.parts:
            return self._from_parts(hints.parts)
        if hints.boxes or hints.positive_points:
            return self._from_prompts(rgb, hints)
        return self._automatic(rgb)

    def _pick(self, masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
        # Never average. One mask only.
        order = np.argsort(-scores)
        for idx in order:
            mask = masks[idx].astype(bool)
            if mask.sum() >= 32:
                return mask
        return masks[int(order[0])].astype(bool)

    def _from_parts(self, parts: list[LayerMask]) -> list[LayerMask]:
        out: list[LayerMask] = []
        for part in parts:
            box = part.bbox
            if box[2] <= 1 or box[3] <= 1:
                out.append(part)
                continue
            xyxy = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
            masks, scores, _ = self._predictor.predict(
                box=xyxy,
                multimask_output=True,
            )
            chosen = self._pick(masks, scores)
            if part.visible is not None:
                chosen = chosen & (part.visible > 0)
            visible = (chosen.astype(np.uint8) * 255)
            # Prefer SAM cut, keep Imagine RGBA for reproject of that region.
            out.append(
                LayerMask(
                    role=part.role,
                    label=part.label,
                    visible=visible,
                    source="sam",
                    rgba=part.rgba,
                    notes=f"{part.notes}; sam2 box refine".strip("; "),
                    score=float(np.max(scores)),
                )
            )
        return out

    def _from_prompts(self, rgb: np.ndarray, hints: SegmentHints) -> list[LayerMask]:
        layers: list[LayerMask] = []
        point_coords = None
        point_labels = None
        if hints.positive_points or hints.negative_points:
            pts = list(hints.positive_points) + list(hints.negative_points)
            labs = [1] * len(hints.positive_points) + [0] * len(hints.negative_points)
            point_coords = np.array(pts, dtype=np.float32)
            point_labels = np.array(labs, dtype=np.int32)
        boxes = hints.boxes or [None]
        for i, box in enumerate(boxes):
            kw = {"multimask_output": True}
            if box is not None:
                kw["box"] = np.array(box, dtype=np.float32)
            if point_coords is not None:
                kw["point_coords"] = point_coords
                kw["point_labels"] = point_labels
            masks, scores, _ = self._predictor.predict(**kw)
            chosen = self._pick(masks, scores)
            layers.append(
                LayerMask(
                    role=f"unknown_{i}",
                    label=f"sam_{i}",
                    visible=(chosen.astype(np.uint8) * 255),
                    source="sam",
                    notes="sam2 prompt",
                    score=float(np.max(scores)),
                )
            )
        return layers

    def _automatic(self, rgb: np.ndarray) -> list[LayerMask]:
        from layerforge.image_io import bbox_from_mask
        from layerforge.ops.normalize import foreground_mask

        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
        fg = foreground_mask(rgba) > 0
        if int(fg.sum()) < 64:
            raise RuntimeError("No foreground to prompt SAM2.")
        x, y, bw, bh = bbox_from_mask((fg.astype(np.uint8) * 255))
        box = np.array([x, y, x + bw, y + bh], dtype=np.float32)
        masks, scores, _ = self._predictor.predict(box=box, multimask_output=True)
        body = self._pick(masks, scores)
        body = body & fg
        layers = [
            LayerMask(
                role="body",
                label="sam_fg_box",
                visible=(body.astype(np.uint8) * 255),
                source="sam",
                notes="sam2 box from foreground bbox; multimask pick-one",
                score=float(np.max(scores)),
            )
        ]
        if self._generator is None:
            return layers
        extras = self._generator.generate(rgb)
        extras = sorted(extras, key=lambda m: float(m.get("predicted_iou", 0)), reverse=True)
        unknown_i = 1
        for item in extras:
            seg = np.asarray(item["segmentation"], dtype=bool)
            if int(seg.sum()) < 64:
                continue
            if int((seg & fg).sum()) / max(1, int(seg.sum())) < 0.5:
                continue
            inter = int((seg & body).sum())
            union = int((seg | body).sum())
            if union and inter / union > 0.85:
                continue
            layers.append(
                LayerMask(
                    role=f"unknown_{unknown_i}",
                    label=f"sam_auto_{unknown_i}",
                    visible=(seg.astype(np.uint8) * 255),
                    source="sam",
                    notes="sam2 automatic sub-part",
                    score=float(item.get("predicted_iou", 0)),
                )
            )
            unknown_i += 1
        return layers


class Sam3HintedSegment:
    name = "sam3.hinted"

    def segment(self, image, hints: SegmentHints) -> list[LayerMask]:
        raise RuntimeError("sam3.hinted is not the cascade text adapter. Use --segment cascade.")

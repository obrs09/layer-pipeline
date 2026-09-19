from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from layerforge.config import resolve_path


class GroundingDinoBoxes:
    """Open-vocab boxes for one role's queries. Not a full-image segmenter."""

    name = "grounding_dino"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._model = None
        self._processor = None

    def _model_dir(self) -> Path:
        paths = self.cfg.get("paths") or {}
        explicit = paths.get("dino_dir", "model/grounding_dino")
        path = resolve_path(explicit)
        if not path.exists() or not path.is_dir():
            raise RuntimeError(
                f"Grounding DINO missing under {path}. "
                "Run scripts/download_models.py or snapshot IDEA-Research/grounding-dino-tiny."
            )
        has_weight = (
            (path / "config.json").exists()
            and (
                (path / "model.safetensors").exists()
                or any(path.glob("*.bin"))
                or any(path.glob("*.safetensors"))
            )
        )
        if not has_weight:
            raise RuntimeError(
                f"Grounding DINO missing under {path}. "
                "Run scripts/download_models.py."
            )
        return path

    def _cuda_or_die(self) -> str:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("grounding_dino needs CUDA. Refusing CPU.")
        return "cuda"

    def _load(self) -> None:
        if self._model is not None:
            return
        device = self._cuda_or_die()
        path = self._model_dir()
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        self._processor = AutoProcessor.from_pretrained(str(path))
        self._model = GroundingDinoForObjectDetection.from_pretrained(str(path)).to(device)
        self._model.eval()

    def detect(
        self,
        image: np.ndarray,
        queries: list[str],
        threshold: float,
    ) -> list[tuple[str, list[float], float]]:
        if not queries:
            return []
        self._load()
        import torch

        rgb = Image.fromarray(image[:, :, :3])
        text = " . ".join(queries) + " ."
        inputs = self._processor(images=rgb, text=text, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        h, w = image.shape[:2]
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=float(threshold),
            text_threshold=float(threshold),
            target_sizes=[(h, w)],
        )[0]
        boxes = []
        labels = results.get("labels") or results.get("text_labels") or []
        scores = results.get("scores")
        xyxy = results.get("boxes")
        if xyxy is None:
            return []
        for i, box in enumerate(xyxy):
            score = float(scores[i]) if scores is not None else 0.0
            label = str(labels[i]) if i < len(labels) else queries[0]
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            boxes.append((label, [x0, y0, x1, y1], score))
        boxes.sort(key=lambda item: item[2], reverse=True)
        return boxes

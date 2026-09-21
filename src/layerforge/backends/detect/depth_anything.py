from __future__ import annotations

from pathlib import Path

import numpy as np

from layerforge.config import resolve_path


class DepthAnythingMap:
    """Monocular relative depth. Weights from YAML paths, never a hardcoded checkpoint."""

    name = "depth_anything"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._processor = None
        self._model = None
        self._device = None

    def _model_dir(self) -> Path:
        paths = self.cfg.get("paths") or {}
        return resolve_path(paths.get("depth_anything_dir", "model/depth_anything"))

    def _hub_id(self) -> str:
        paths = self.cfg.get("paths") or {}
        return str(paths.get("depth_anything_model") or "depth-anything/Depth-Anything-V2-Small-hf")

    def available(self) -> bool:
        path = self._model_dir()
        return path.exists() and any(path.glob("*.json"))

    def _cuda_or_die(self) -> str:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("depth_anything needs CUDA.")
        return "cuda"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        device = self._cuda_or_die()
        local = self._model_dir()
        allow_hub = bool((((self.cfg.get("cascade") or {}).get("hair_split") or {}).get("depth") or {}).get("allow_hub", False))
        if not self.available() and not allow_hub:
            raise RuntimeError("no local depth_anything weights")
        source = str(local) if self.available() else self._hub_id()
        kwargs: dict = {}
        local.mkdir(parents=True, exist_ok=True)
        kwargs["cache_dir"] = str(local)
        if self.available():
            kwargs["local_files_only"] = True
        self._processor = AutoImageProcessor.from_pretrained(source, **kwargs)
        self._model = AutoModelForDepthEstimation.from_pretrained(source, **kwargs).to(device)
        self._model.eval()
        self._device = device
        self._torch = torch

    def predict(self, image: np.ndarray) -> np.ndarray:
        """HxW float32 relative depth. Larger values' meaning is calibrated by the split."""
        self._load()
        from PIL import Image

        rgb = image[:, :, :3] if image.ndim == 3 else image
        pil = Image.fromarray(rgb.astype(np.uint8))
        inputs = self._processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            depth = self._model(**inputs).predicted_depth
        depth = self._torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        )[0, 0]
        return depth.detach().float().cpu().numpy()

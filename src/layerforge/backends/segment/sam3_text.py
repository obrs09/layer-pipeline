from __future__ import annotations

from pathlib import Path

import numpy as np

from layerforge.config import resolve_path


class Sam3TextMasker:
    """SAM 3 text mask. Optional cascade step; missing package → skip to SAM2."""

    name = "sam3.text"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._predictor = None

    def _checkpoint(self) -> Path:
        paths = self.cfg.get("paths") or {}
        explicit = paths.get("sam3_checkpoint")
        search = resolve_path(paths.get("sam3_dir", "model/sam3"))
        if explicit:
            path = resolve_path(explicit)
            if path.exists():
                return path
        found = sorted(search.glob("*.pt")) + sorted(search.glob("*.pth"))
        if not found:
            raise RuntimeError(f"No SAM3 checkpoint under {search}.")
        return found[0]

    def _load(self) -> None:
        if self._predictor is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("sam3.text needs torch.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("sam3.text needs CUDA. Refusing CPU.")
        ckpt = self._checkpoint()
        last_err = None
        for builder in (_build_sam3_official,):
            try:
                self._predictor = builder(ckpt)
                return
            except Exception as exc:
                last_err = exc
        raise RuntimeError(f"Could not load SAM3 from {ckpt}: {last_err}") from last_err

    def predict_text(
        self,
        image: np.ndarray,
        query: str,
        character: np.ndarray,
    ) -> np.ndarray | None:
        try:
            self._load()
        except RuntimeError:
            return None
        rgb = image[:, :, :3]
        mask = self._predictor(rgb, query)
        if mask is None:
            return None
        chosen = np.asarray(mask).astype(bool)
        if character is not None:
            chosen = chosen & (character > 0)
        if int(chosen.sum()) < 16:
            return None
        return chosen.astype(np.uint8) * 255


def _build_sam3_official(ckpt: Path):
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(checkpoint_path=str(ckpt))
    model.cuda()
    model.eval()

    def _predict(rgb: np.ndarray, query: str):
        import torch

        from sam3.model.sam3_image_processor import Sam3Processor

        processor = Sam3Processor(model)
        state = processor.set_image(rgb)
        with torch.no_grad():
            out = processor.set_text_prompt(prompt=query, state=state)
        masks = out.get("masks") if isinstance(out, dict) else None
        if masks is None:
            return None
        arr = masks
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 4:
            arr = arr[0, 0]
        elif arr.ndim == 3:
            arr = arr[0]
        return arr > 0.5

    return _predict

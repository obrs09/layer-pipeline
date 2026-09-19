from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from layerforge.config import resolve_path


class Sd15AnimeInpaint:
    name = "sd15.anime"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._pipe = None
        self._prompt = (
            (cfg.get("inpaint") or {}).get("prompt")
            or "anime illustration, continue existing material and lineart, no new pattern"
        )
        self._negative = (cfg.get("inpaint") or {}).get(
            "negative_prompt", "photorealistic, watermark, extra limbs, text"
        )

    def _cuda_or_die(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("sd15.anime needs torch. Install the [gpu] extra.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("sd15.anime needs CUDA. Refusing to run SD on CPU.")

    def _model_path(self) -> Path:
        paths = self.cfg.get("paths") or {}
        explicit = paths.get("sd15_model")
        search = resolve_path(explicit or paths.get("sd15_dir", "model/sd15"))
        if not search.exists():
            raise RuntimeError(f"SD1.5 model path missing: {search}. Put weights in model/sd15/.")
        has_weight = (
            any(search.glob("*.safetensors"))
            or any(search.glob("*.ckpt"))
            or (search / "model_index.json").exists()
        )
        if not has_weight and search.is_dir() and not any(search.iterdir()):
            raise RuntimeError(f"model/sd15 is empty. Place a Diffusers folder or .safetensors there.")
        return search

    def _load(self):
        if self._pipe is not None:
            return
        self._cuda_or_die()
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        model = self._model_path()
        backend = {}
        import yaml

        p = resolve_path("configs/backends/inpaint_sd15.yaml")
        if p.exists():
            backend = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        self._prompt = backend.get("prompt", self._prompt)
        self._negative = backend.get("negative_prompt", self._negative)
        self._steps = int(backend.get("steps", 28))
        self._guidance = float(backend.get("guidance", 7.0))
        kwargs = {"torch_dtype": torch.float16, "safety_checker": None}
        weights = (
            sorted(model.glob("*inpainting*.safetensors"))
            + sorted(model.glob("*inpainting*.ckpt"))
            + sorted(model.glob("*.safetensors"))
            + sorted(model.glob("*.ckpt"))
        )
        last_err = None
        if weights:
            try:
                self._pipe = StableDiffusionInpaintPipeline.from_single_file(
                    str(weights[0]), **kwargs
                )
            except Exception as exc:
                last_err = exc
                self._pipe = None
        if self._pipe is None and model.is_dir() and (model / "model_index.json").exists():
            try:
                self._pipe = StableDiffusionInpaintPipeline.from_pretrained(str(model), **kwargs)
            except Exception as exc:
                last_err = exc
                self._pipe = None
        if self._pipe is None:
            raise RuntimeError(
                f"Could not load SD1.5 inpaint from {model}: {last_err}"
            ) from last_err
        self._pipe = self._pipe.to("cuda")
        self._pipe.enable_attention_slicing()

    def inpaint(self, image: np.ndarray, mask: np.ndarray, prompt: str) -> np.ndarray:
        self._load()
        if (mask > 0).sum() == 0:
            return image.copy()
        rgb = Image.fromarray(image[:, :, :3])
        hole = Image.fromarray(((mask > 0) * 255).astype(np.uint8))
        used_prompt = prompt or self._prompt
        result = self._pipe(
            prompt=used_prompt,
            negative_prompt=self._negative,
            image=rgb,
            mask_image=hole,
            num_inference_steps=self._steps,
            guidance_scale=self._guidance,
        ).images[0]
        filled = np.array(result.convert("RGB"))
        out = image.copy()
        h, w = out.shape[:2]
        if filled.shape[0] != h or filled.shape[1] != w:
            filled = np.array(Image.fromarray(filled).resize((w, h), Image.Resampling.LANCZOS))
        out[:, :, :3] = filled
        if out.shape[2] == 4:
            alpha = out[:, :, 3]
            alpha[mask > 0] = 255
            out[:, :, 3] = alpha
        return out

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from layerforge.config import resolve_path


class Sam3TextMasker:
    """SAM 3 text mask. Optional cascade step; missing package → skip to SAM2."""

    name = "sam3.text"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._predictor = None
        self.load_error: str | None = None
        self.infer_error: str | None = None

    @property
    def error(self) -> str | None:
        return self.load_error or self.infer_error

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
        if self.load_error:
            raise RuntimeError(self.load_error)
        try:
            import torch
        except ImportError as exc:
            self.load_error = "sam3.text needs torch."
            raise RuntimeError(self.load_error) from exc
        if not torch.cuda.is_available():
            self.load_error = "sam3.text needs CUDA. Refusing CPU."
            raise RuntimeError(self.load_error)
        try:
            ckpt = self._checkpoint()
            self._predictor = _build_sam3_official(ckpt)
        except Exception as exc:
            self.load_error = str(exc) if str(exc) else repr(exc)
            if "Could not load SAM3" not in self.load_error:
                self.load_error = f"Could not load SAM3: {self.load_error}"
            raise RuntimeError(self.load_error) from exc

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
        try:
            mask = self._predictor(rgb, query)
        except Exception as exc:
            # One bad query must not disable SAM3 for the rest of the process.
            self.infer_error = f"SAM3 infer failed ({query}): {exc}"
            return None
        self.infer_error = None
        if mask is None:
            return None
        chosen = np.asarray(mask).astype(bool)
        if character is not None:
            chosen = chosen & (character > 0)
        if int(chosen.sum()) < 16:
            return None
        return chosen.astype(np.uint8) * 255


def _install_optional_import_stubs() -> None:
    """SAM3 image text does not need tracker EDT or COCO loaders; Windows often lacks both."""
    try:
        import triton  # noqa: F401
    except ImportError:
        _stub_edt()
    _stub_pycocotools()


def _stub_pycocotools() -> None:
    try:
        from pycocotools import mask as _mask  # noqa: F401
        return
    except ImportError:
        pass
    pkg = types.ModuleType("pycocotools")
    pkg.__path__ = []  # type: ignore[attr-defined]
    mask = types.ModuleType("pycocotools.mask")
    sys.modules["pycocotools"] = pkg
    sys.modules["pycocotools.mask"] = mask
    pkg.mask = mask


def _stub_edt() -> None:
    if "sam3.model.edt" in sys.modules:
        return
    edt = types.ModuleType("sam3.model.edt")

    def edt_triton(*_args, **_kwargs):
        raise RuntimeError("SAM3 tracker EDT needs triton (not shipped on Windows).")

    edt.edt_triton = edt_triton
    sys.modules["sam3.model.edt"] = edt


def _install_windows_edt_stub() -> None:
    _install_optional_import_stubs()


def _build_sam3_official(ckpt: Path):
    _install_optional_import_stubs()
    import sam3.model_builder as mb
    from sam3.model.sam3_image_processor import Sam3Processor
    from PIL import Image

    bpe = Path(mb.__file__).resolve().parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe.is_file():
        raise RuntimeError(f"SAM3 BPE vocab missing: {bpe}")
    model = mb.build_sam3_image_model(
        checkpoint_path=str(ckpt),
        bpe_path=str(bpe),
        load_from_HF=False,
        device="cuda",
        eval_mode=True,
        enable_inst_interactivity=False,
    )
    processor = Sam3Processor(model)
    # Holding `ref` keeps the source buffer alive, so a later image cannot land on the
    # same address and hit a stale embedding.
    cache: dict = {"key": None, "state": None, "ref": None}

    def _predict(rgb: np.ndarray, query: str):
        import torch

        key = (rgb.shape, int(rgb.dtype.itemsize), int(rgb.ctypes.data))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if cache["key"] != key:
                pil = Image.fromarray(rgb.astype(np.uint8))
                cache["state"] = processor.set_image(pil)
                cache["key"] = key
                cache["ref"] = rgb
            out = processor.set_text_prompt(prompt=query, state=cache["state"])
        masks = out.get("masks") if isinstance(out, dict) else None
        if masks is None:
            return None
        arr = masks
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        if arr.size == 0:
            return None
        if arr.ndim == 4:
            arr = arr.any(axis=(0, 1))
        elif arr.ndim == 3:
            arr = arr.any(axis=0)
        elif arr.ndim != 2:
            return None
        return arr > 0.5

    return _predict

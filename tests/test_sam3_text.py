from __future__ import annotations

import numpy as np
import pytest

from layerforge.backends.segment.sam3_text import Sam3TextMasker, _install_optional_import_stubs


def test_sam3_missing_checkpoint_returns_none_and_records_error(tmp_path):
    masker = Sam3TextMasker({"paths": {"sam3_dir": str(tmp_path / "empty")}})
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    character = np.full((8, 8), 255, dtype=np.uint8)
    assert masker.predict_text(image, "hair", character) is None
    assert masker.load_error
    assert "checkpoint" in masker.load_error.lower() or "cuda" in masker.load_error.lower() or "torch" in masker.load_error.lower()


def test_sam3_image_builder_imports_without_triton():
    pytest.importorskip("sam3")
    _install_optional_import_stubs()
    from sam3.model_builder import build_sam3_image_model

    assert callable(build_sam3_image_model)

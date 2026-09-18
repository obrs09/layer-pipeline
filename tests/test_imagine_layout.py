from __future__ import annotations

from pathlib import Path

from layerforge.ingest.imagine_parts import is_imagine_dir


def test_detects_segments_layout():
    root = Path(__file__).resolve().parents[1] / "test_input" / "image_partial_sag" / "image0"
    assert is_imagine_dir(root)


def test_flat_folder_is_not_imagine():
    root = Path(__file__).resolve().parents[1] / "test_input" / "image_no_sag"
    assert not is_imagine_dir(root)

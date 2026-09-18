from __future__ import annotations

from pathlib import Path

import pytest

from layerforge.ingest.imagine_parts import is_imagine_dir

TEST_INPUT = Path(__file__).resolve().parents[1] / "test_input"
IMAGINE_JOB = TEST_INPUT / "image_partial_sag" / "image0"
FLAT_DIR = TEST_INPUT / "image_no_sag"

pytestmark = pytest.mark.skipif(
    not IMAGINE_JOB.exists() or not FLAT_DIR.exists(),
    reason="local test_input/ is not present",
)


def test_detects_segments_layout():
    assert is_imagine_dir(IMAGINE_JOB)


def test_flat_folder_is_not_imagine():
    assert not is_imagine_dir(FLAT_DIR)

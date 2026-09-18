from __future__ import annotations

from layerforge.ingest.imagine_parts import is_imagine_dir


def test_detects_segments_layout(tmp_path):
    job = tmp_path / "job"
    (job / "segments").mkdir(parents=True)
    assert is_imagine_dir(job)


def test_detects_parts_layout(tmp_path):
    job = tmp_path / "job"
    (job / "parts").mkdir(parents=True)
    assert is_imagine_dir(job)


def test_flat_folder_is_not_imagine(tmp_path):
    job = tmp_path / "flat"
    job.mkdir()
    assert not is_imagine_dir(job)

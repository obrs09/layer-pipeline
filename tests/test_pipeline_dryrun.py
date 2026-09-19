from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from layerforge.config import load_config
from layerforge.ingest.imagine_parts import ingest_imagine
from layerforge.pipeline import run_pipeline
from layerforge.taxonomy import load_taxonomy


def _make_imagine_job(root: Path) -> Path:
    job = root / "imagine_job"
    seg = job / "segments" / "pink-eyes"
    seg.mkdir(parents=True)
    source = np.zeros((48, 64, 3), dtype=np.uint8)
    source[:] = 250
    source[10:22, 8:20] = (30, 80, 200)
    source[10:22, 40:52] = (30, 80, 200)
    Image.fromarray(source, "RGB").save(job / "source.png")
    crop = np.zeros((12, 12, 4), dtype=np.uint8)
    crop[:, :, :3] = (30, 80, 200)
    crop[:, :, 3] = 255
    Image.fromarray(crop, "RGBA").save(seg / "pink-eyes-1.png")
    Image.fromarray(crop, "RGBA").save(seg / "pink-eyes-2.png")
    return job


def test_imagine_places_crops(tmp_path: Path):
    job = _make_imagine_job(tmp_path)
    result = ingest_imagine(job, load_taxonomy())
    assert result.kind == "imagine"
    roles = {part.role for part in result.hints.parts}
    assert roles == {"eye_l", "eye_r"}
    xs = sorted(part.bbox[0] for part in result.hints.parts)
    assert xs[0] == 8
    assert xs[1] == 40


def test_pipeline_dryrun_imagine(tmp_path: Path):
    job = _make_imagine_job(tmp_path)
    out = run_pipeline(
        job,
        tmp_path / "runs",
        load_config(),
        job_id="dry",
        dry_run=True,
        segment_name="noop_from_parts",
        inpaint_name="identity",
    )
    manifest = (out / "manifest.json").read_text(encoding="utf-8")
    assert "layerforge.manifest.v2" in manifest
    assert (out / "preview" / "stack.png").exists()
    assert (out / "preview" / "diff.png").exists()
    assert list((out / "layers").glob("*.png"))


def test_imagine_visible_pixels_match_source(tmp_path: Path):
    job = tmp_path / "imagine_job"
    (job / "segments").mkdir(parents=True)
    source = np.zeros((32, 40, 3), dtype=np.uint8)
    source[:] = 250
    source[8:20, 8:20] = (10, 20, 200)
    Image.fromarray(source, "RGB").save(job / "source.png")
    crop = np.zeros((12, 12, 4), dtype=np.uint8)
    crop[:, :, :3] = (255, 0, 0)
    crop[:, :, 3] = 255
    Image.fromarray(crop, "RGBA").save(job / "segments" / "pink-eyes.png")
    out = run_pipeline(
        job,
        tmp_path / "runs",
        load_config(),
        job_id="reproject",
        dry_run=True,
        segment_name="noop_from_parts",
        inpaint_name="identity",
    )
    layer = np.array(Image.open(out / "layers" / "60_eye_l.png").convert("RGBA"))
    vis = layer[:, :, 3] > 0
    src = np.array(Image.open(out / "source.png").convert("RGBA"))
    assert vis.any()
    assert np.all(layer[vis, :3] == src[vis, :3])


def test_export_clears_stale_layer_files(tmp_path: Path):
    job = _make_imagine_job(tmp_path)
    stale_root = tmp_path / "runs" / "dry"
    stale = stale_root / "layers"
    stale.mkdir(parents=True)
    (stale / "90_acc_33.png").write_bytes(b"not-a-real-png")
    (stale / "50_unknown_0.png").write_bytes(b"not-a-real-png")
    out = run_pipeline(
        job,
        tmp_path / "runs",
        load_config(),
        job_id="dry",
        dry_run=True,
        segment_name="noop_from_parts",
        inpaint_name="identity",
    )
    names = {p.name for p in (out / "layers").glob("*.png")}
    assert "90_acc_33.png" not in names
    assert "50_unknown_0.png" not in names
    import json

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    expected = {Path(layer["file"]).name for layer in manifest["layers"]}
    assert names == expected


def test_export_resets_run_log(tmp_path: Path):
    job = _make_imagine_job(tmp_path)
    log_path = tmp_path / "runs" / "dry" / "logs" / "run.jsonl"
    log_path.parent.mkdir(parents=True)
    log_path.write_text('{"event":"old-run"}\n', encoding="utf-8")
    out = run_pipeline(
        job,
        tmp_path / "runs",
        load_config(),
        job_id="dry",
        dry_run=True,
        segment_name="noop_from_parts",
        inpaint_name="identity",
    )
    text = (out / "logs" / "run.jsonl").read_text(encoding="utf-8")
    assert "old-run" not in text
    assert '"event": "start"' in text
    assert text.count('"event": "start"') == 1

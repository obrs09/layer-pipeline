from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from layerforge.backends.segment.cascade import CascadeSegment
from layerforge.cli import expand_run_inputs
from layerforge.config import load_config
from layerforge.ops.trace import StepDump, masked_rgba
from layerforge.pipeline import run_pipeline, short_job_id
from layerforge.taxonomy import load_taxonomy


def test_short_job_id_from_grok_filename():
    path = Path("test_input/image_no_sag/grok-image-64010c19-6759-4ab7-92fa-b0bede4bb0d7.jpg")
    assert short_job_id(path) == "64010c19"


def test_expand_run_inputs_lists_images(tmp_path: Path):
    folder = tmp_path / "flats"
    folder.mkdir()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(folder / "a.png")
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(folder / "b.png")
    (folder / "notes.txt").write_text("no", encoding="utf-8")
    names = [p.name for p in expand_run_inputs(folder)]
    assert names == ["a.png", "b.png"]


def test_step_dump_character(tmp_path: Path):
    dump = StepDump(tmp_path, enabled=True)
    dump.reset()
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[:] = (10, 20, 30)
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:6, 3:8] = 255
    dump.write_character(image, mask)
    rgba = np.array(Image.open(tmp_path / "steps" / "01_character" / "rgba.png"))
    assert rgba.shape[2] == 4
    assert int(rgba[0, 0, 3]) == 0
    assert int(rgba[3, 4, 3]) == 255
    assert tuple(rgba[3, 4, :3]) == (10, 20, 30)


def test_cascade_writes_character_step(tmp_path: Path):
    class _Cut:
        name = "anime_segmentation"

        def cut(self, image):
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
            return mask

    dump = StepDump(tmp_path, enabled=True)
    dump.reset()
    seg = CascadeSegment({}, load_taxonomy(), character=_Cut(), dry_run=True)
    seg.bind_dump(dump)
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[1:7, 2:8] = (40, 80, 200)
    mask = seg._cut_character(image)
    seg._dump_character(image, mask)
    assert (tmp_path / "steps" / "01_character" / "mask.png").exists()
    assert (tmp_path / "steps" / "01_character" / "rgba.png").exists()
    assert (tmp_path / "steps" / "01_character" / "overlay.png").exists()


def test_pipeline_dryrun_writes_steps(tmp_path: Path):
    job = tmp_path / "imagine_job"
    seg = job / "segments" / "pink-eyes"
    seg.mkdir(parents=True)
    source = np.zeros((32, 40, 3), dtype=np.uint8)
    source[:] = 250
    source[8:20, 8:20] = (10, 20, 200)
    Image.fromarray(source, "RGB").save(job / "source.png")
    crop = np.zeros((12, 12, 4), dtype=np.uint8)
    crop[:, :, :3] = (10, 20, 200)
    crop[:, :, 3] = 255
    Image.fromarray(crop, "RGBA").save(seg / "pink-eyes.png")
    out = run_pipeline(
        job,
        tmp_path / "runs",
        load_config(),
        job_id="steps",
        dry_run=True,
        segment_name="noop_from_parts",
        inpaint_name="identity",
    )
    assert (out / "steps" / "04_segment" / "overlay.png").exists()
    assert (out / "steps" / "05_refine" / "overlay.png").exists()
    assert (out / "steps" / "06_occlusion" / "overlay.png").exists()
    assert list((out / "layers").glob("*.png"))


def test_masked_rgba_keeps_visible_rgb():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = (9, 8, 7)
    mask = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    out = masked_rgba(image, mask)
    assert tuple(out[0, 0, :3]) == (9, 8, 7)
    assert int(out[0, 1, 3]) == 0

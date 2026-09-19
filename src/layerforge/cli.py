from __future__ import annotations

import argparse
import sys
from pathlib import Path

from layerforge.config import load_config
from layerforge.image_io import is_image
from layerforge.ingest.imagine_parts import is_imagine_dir
from layerforge.pipeline import run_pipeline


def expand_run_inputs(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir() and is_imagine_dir(path):
        return [path]
    if path.is_dir():
        images = sorted(p for p in path.iterdir() if is_image(p))
        if images:
            return images
        raise ValueError(
            f"{path} has no images and no segments/ folder. "
            "Pass a single image, a folder of images, or an Imagine job directory."
        )
    raise FileNotFoundError(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="layerforge")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="ingest → layers pack")
    run.add_argument("--input", required=True, help="image file, folder of images, or Imagine job directory")
    run.add_argument("--out", default="runs")
    run.add_argument("--config", default=None)
    run.add_argument("--segment", default=None)
    run.add_argument("--inpaint", default=None)
    run.add_argument("--job-id", default=None)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        cfg = load_config(args.config)
        inputs = expand_run_inputs(Path(args.input))
        for item in inputs:
            job_id = args.job_id if len(inputs) == 1 else None
            out = run_pipeline(
                input_path=item,
                out_root=args.out,
                cfg=cfg,
                segment_name=args.segment,
                inpaint_name=args.inpaint,
                job_id=job_id,
                dry_run=args.dry_run,
            )
            print(out)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

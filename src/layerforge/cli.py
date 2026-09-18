from __future__ import annotations

import argparse
import sys
from pathlib import Path

from layerforge.config import load_config
from layerforge.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="layerforge")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="ingest → layers pack")
    run.add_argument("--input", required=True, help="image file or Imagine job directory")
    run.add_argument("--out", default="runs")
    run.add_argument("--config", default=None)
    run.add_argument("--segment", default=None)
    run.add_argument("--inpaint", default=None)
    run.add_argument("--job-id", default=None)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        cfg = load_config(args.config)
        out = run_pipeline(
            input_path=args.input,
            out_root=args.out,
            cfg=cfg,
            segment_name=args.segment,
            inpaint_name=args.inpaint,
            job_id=args.job_id,
            dry_run=args.dry_run,
        )
        print(out)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

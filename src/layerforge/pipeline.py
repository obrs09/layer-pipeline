from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from layerforge.backends.export.png_pack import export_png_pack
from layerforge.backends.export.psd import export_psd
from layerforge.backends.registry import build_inpaint, build_segment
from layerforge.contracts import SCHEMA, BackendInfo, Canvas, LayerRecord, Manifest, validate_manifest
from layerforge.image_io import alpha_over, bbox_from_mask
from layerforge.ingest import ingest_input
from layerforge.logutil import RunLog
from layerforge.ops.assign_roles import assign_roles
from layerforge.ops.normalize import resize_max_side, to_rgba
from layerforge.ops.occlusion import plan_occlusion
from layerforge.ops.refine import assign_residual_to_body, refine_masks
from layerforge.ops.reproject import reproject
from layerforge.taxonomy import Taxonomy, load_taxonomy


def _layer_ids(layers, taxonomy: Taxonomy) -> list[str]:
    counts: Counter[str] = Counter()
    ids: list[str] = []
    for layer in layers:
        order = taxonomy.spec(layer.role).order
        base = f"{order:02d}_{layer.role}"
        counts[base] += 1
        n = counts[base]
        ids.append(base if n == 1 else f"{base}_{n}")
    return ids


def run_pipeline(
    input_path: str | Path,
    out_root: str | Path,
    cfg: dict,
    taxonomy: Taxonomy | None = None,
    segment_name: str | None = None,
    inpaint_name: str | None = None,
    job_id: str | None = None,
    dry_run: bool = False,
) -> Path:
    input_path = Path(input_path)
    out_root = Path(out_root)
    taxonomy = taxonomy or load_taxonomy()
    ingested = ingest_input(input_path, taxonomy)
    source = to_rgba(ingested.source_rgba)
    max_side = int((cfg.get("normalize") or {}).get("max_side", 2048))
    source = resize_max_side(source, max_side)
    if source.shape != ingested.source_rgba.shape and ingested.hints.parts:
        # Parts were placed on original size; skip resize if we already placed.
        source = to_rgba(ingested.source_rgba)

    job_id = job_id or _default_job_id(input_path)
    out_dir = out_root / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(out_dir / "logs" / "run.jsonl")
    log.write("start", input=str(input_path), kind=ingested.kind, dry_run=dry_run)

    seg_name = segment_name or (cfg.get("segment") or {}).get("name", "auto")
    inp_name = inpaint_name or (cfg.get("inpaint") or {}).get("name", "identity")
    segment = build_segment(seg_name, cfg, ingested.kind, dry_run)
    inpaint = build_inpaint(inp_name, cfg, dry_run)
    log.write("backends", segment=segment.name, inpaint=inpaint.name)

    masks = segment.segment(source, ingested.hints)
    log.write("segment", count=len(masks))
    masks = assign_roles(masks, source, taxonomy)
    log.write("roles", roles=[m.role for m in masks])
    refine_cfg = cfg.get("refine") or {}
    masks = refine_masks(
        masks,
        taxonomy,
        min_area=int(refine_cfg.get("min_area", 64)),
        mutex=bool(refine_cfg.get("mutex", True)),
        overlay_iou=float(refine_cfg.get("overlay_iou", 0.5)),
        overlay_contain=float(refine_cfg.get("overlay_contain", 0.75)),
    )
    masks = assign_residual_to_body(
        masks,
        source,
        min_area=int(refine_cfg.get("min_area", 64)),
        background_luma=int((cfg.get("occlusion") or {}).get("background_luma", 250)),
    )
    log.write("refine", count=len(masks))
    occ_map = plan_occlusion(
        masks,
        source,
        taxonomy,
        background_luma=int((cfg.get("occlusion") or {}).get("background_luma", 250)),
    )
    seam = int((cfg.get("inpaint") or {}).get("seam_dilate_px", 2))

    ids = _layer_ids(masks, taxonomy)
    layers_rgba: dict[str, np.ndarray] = {}
    masks_visible: dict[str, np.ndarray] = {}
    masks_occluded: dict[str, np.ndarray] = {}
    records: list[LayerRecord] = []
    composite = np.zeros_like(source)
    visible_union = np.zeros(source.shape[:2], dtype=np.uint8)

    identity = inpaint.name == "identity"
    prompt_tpl = (
        "anime {role}, continue existing material and lineart, "
        "do not add new patterns, text, or limbs"
    )

    for idx, (layer_id, layer) in enumerate(zip(ids, masks)):
        visible = layer.visible
        occluded = occ_map.get(idx, np.zeros_like(visible))
        if seam and int((occluded > 0).sum()) > 0:
            import cv2

            occluded = cv2.dilate(occluded, np.ones((3, 3), np.uint8), iterations=seam)
            occluded[visible > 0] = 0
        occ_for_fill = occluded if not identity else np.zeros_like(visible)
        if identity or int((occluded > 0).sum()) == 0:
            filled = source
            complete = int((occluded > 0).sum()) == 0
        else:
            prompt = prompt_tpl.format(role=layer.role)
            filled = inpaint.inpaint(source, occluded, prompt)
            complete = True
            engine = getattr(inpaint, "last_engine", inpaint.name)
            if engine and engine not in {"skip", "identity"}:
                layer.notes = f"{layer.notes}; inpaint={engine}".strip("; ")
        layer_rgba = reproject(filled, source, visible, occ_for_fill)
        layers_rgba[layer_id] = layer_rgba
        masks_visible[layer_id] = visible
        masks_occluded[layer_id] = occluded
        visible_union = np.maximum(visible_union, visible)
        composite = alpha_over(composite, layer_rgba)
        occ_path = f"masks/{layer_id}.occluded.png" if int((occluded > 0).sum()) else None
        records.append(
            LayerRecord(
                id=layer_id,
                role=layer.role,
                order=taxonomy.spec(layer.role).order,
                file=f"layers/{layer_id}.png",
                mask_visible=f"masks/{layer_id}.png",
                mask_occluded=occ_path,
                bbox=bbox_from_mask(visible),
                source=layer.source,
                complete=complete,
                notes=layer.notes,
            )
        )
        log.write("layer", id=layer_id, role=layer.role, complete=complete)

    manifest = Manifest(
        schema=SCHEMA,
        job_id=job_id,
        source="source.png",
        canvas=Canvas(w=int(source.shape[1]), h=int(source.shape[0])),
        backend=BackendInfo(
            segment=segment.name,
            inpaint=inpaint.name,
            compose=(cfg.get("compose") or {}).get("name", "reproject_v1"),
        ),
        layers=records,
    )
    validate_manifest(manifest.to_dict())
    export_png_pack(
        out_dir,
        manifest,
        source,
        layers_rgba,
        masks_visible,
        masks_occluded,
        composite,
        visible_union,
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if (cfg.get("export") or {}).get("psd"):
        export_psd(out_dir, [out_dir / rec.file for rec in records])
    log.write("done", out=str(out_dir), layers=len(records))
    return out_dir


def _default_job_id(input_path: Path) -> str:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{input_path.stem}"

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
from layerforge.logutil import RunLog, write_tags_json
from layerforge.ops.assign_roles import assign_roles
from layerforge.ops.normalize import resize_max_side, to_rgba
from layerforge.ops.occlusion import plan_occlusion
from layerforge.ops.occlusion import occluded_over_lower_visible
from layerforge.ops.refine import (
    assign_residual_to_body,
    assign_unclaimed_seams,
    fill_unclaimed_domain,
    refine_masks,
    unclaimed_in_domain,
)
from layerforge.ops.reproject import reproject
from layerforge.ops.trace import StepDump
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


def _cut_ids(layers) -> list[str]:
    """Debug dump names: cut sequence, not taxonomy draw order."""
    return [f"{idx:02d}_{layer.role}" for idx, layer in enumerate(layers)]


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

    job_id = job_id or short_job_id(input_path)
    out_dir = out_root / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dump = StepDump(out_dir, enabled=bool((cfg.get("export") or {}).get("steps", True)))
    dump.reset()
    log = RunLog(out_dir / "logs" / "run.jsonl")
    log.write("start", input=str(input_path), kind=ingested.kind, dry_run=dry_run)

    seg_name = segment_name or (cfg.get("segment") or {}).get("name", "auto")
    inp_name = inpaint_name or (cfg.get("inpaint") or {}).get("name", "identity")
    segment = build_segment(seg_name, cfg, ingested.kind, dry_run)
    inpaint = build_inpaint(inp_name, cfg, dry_run)
    if hasattr(segment, "bind_dump"):
        segment.bind_dump(dump)
    log.write("backends", segment=segment.name, inpaint=inpaint.name)

    masks = segment.segment(source, ingested.hints)
    tags = getattr(segment, "tags", None) or {}
    if tags:
        write_tags_json(out_dir / "steps" / "02_tags.json", tags)
        write_tags_json(out_dir / "logs" / "tags.json", tags)
        top = sorted(tags.items(), key=lambda item: (-float(item[1]), item[0]))[:16]
        log.write(
            "tags",
            count=len(tags),
            top=[{"tag": name, "score": round(float(score), 4)} for name, score in top],
        )
    character_qa = getattr(segment, "character_qa", None)
    if character_qa:
        dump.write_json("01_character/qa.json", character_qa)
        if character_qa.get("flagged"):
            dump.write_json("01_character/FLAGGED.json", character_qa)
        (out_dir / "logs").mkdir(parents=True, exist_ok=True)
        (out_dir / "logs" / "character_qa.json").write_text(
            json.dumps(character_qa, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.write(
            "character_qa",
            flagged=bool(character_qa.get("flagged")),
            chosen_seed=character_qa.get("chosen_seed"),
            attempts=len(character_qa.get("attempts") or []),
            reasons=(character_qa.get("attempts") or [{}])[-1].get("reasons") or [],
        )
    log.write(
        "segment",
        count=len(masks),
        inventory=list(getattr(segment, "inventory", []) or []),
        missing=list(getattr(segment, "missing", []) or []),
        needs_click=list(getattr(segment, "needs_click", []) or []),
    )
    masks = assign_roles(masks, source, taxonomy)
    log.write("roles", roles=[m.role for m in masks])
    cut_ids = _cut_ids(masks)
    dump.write_layers("04_segment", source, masks, cut_ids)
    dump.write_json(
        "04_segment/cut_order.json",
        {
            "planned": list(getattr(segment, "cut_plan", []) or []),
            "kept": [
                {"index": idx, "id": cut_ids[idx], "role": layer.role, "notes": layer.notes}
                for idx, layer in enumerate(masks)
            ],
        },
    )
    domain = getattr(segment, "character_mask", None)
    refine_cfg = cfg.get("refine") or {}
    masks = refine_masks(
        masks,
        taxonomy,
        min_area=int(refine_cfg.get("min_area", 64)),
        mutex=bool(refine_cfg.get("mutex", True)),
        overlay_iou=float(refine_cfg.get("overlay_iou", 0.5)),
        overlay_contain=float(refine_cfg.get("overlay_contain", 0.75)),
        morph_open_px=int(refine_cfg.get("morph_open_px", 2)),
    )
    masks = assign_residual_to_body(
        masks,
        source,
        min_area=int(refine_cfg.get("min_area", 64)),
        background_luma=int((cfg.get("occlusion") or {}).get("background_luma", 250)),
        max_frac=float(refine_cfg.get("residual_max_frac", 0.04)),
        domain=domain,
    )
    masks = assign_unclaimed_seams(
        masks,
        source,
        taxonomy,
        background_luma=int((cfg.get("occlusion") or {}).get("background_luma", 250)),
        max_dist=float(refine_cfg.get("seam_fill_px", 24)),
        domain=domain,
    )
    if domain is not None:
        background_luma = int((cfg.get("occlusion") or {}).get("background_luma", 250))
        before = unclaimed_in_domain(masks, domain, taxonomy, source, background_luma)
        if before.any():
            dump.write_png("05_refine/unclaimed.png", before.astype(np.uint8) * 255)
        masks, coverage = fill_unclaimed_domain(
            masks,
            domain,
            taxonomy,
            source,
            background_luma=background_luma,
            min_area=int(refine_cfg.get("min_area", 64)),
            max_dist=float(refine_cfg.get("seam_fill_px", 24)),
        )
        uncovered = unclaimed_in_domain(masks, domain, taxonomy)
        coverage["character_uncovered_px"] = int(uncovered.sum())
        log.write("coverage", **coverage)
    log.write("refine", count=len(masks))
    ids = _layer_ids(masks, taxonomy)
    dump.write_layers("05_refine", source, masks, ids)
    occ_map = plan_occlusion(
        masks,
        source,
        taxonomy,
        background_luma=int((cfg.get("occlusion") or {}).get("background_luma", 250)),
        seam_dilate_px=int((cfg.get("inpaint") or {}).get("seam_dilate_px", 2)),
    )
    overlap = occluded_over_lower_visible(masks, occ_map, taxonomy)
    if overlap:
        raise RuntimeError(
            f"occlusion plan puts {overlap} hole pixels on a lower layer's visible pixels"
        )
    log.write("occlusion", holes=sum(int((m > 0).sum()) for m in occ_map.values()), over_lower_visible=overlap)
    dump.write_layers("06_occlusion", source, masks, ids, occluded=occ_map)

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
            dump.write_inpaint_hole(layer_id, filled, occluded)
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
                needs_click=bool(getattr(layer, "needs_click", False)),
            )
        )
        log.write("layer", id=layer_id, role=layer.role, complete=complete)

    missing = list(getattr(segment, "missing", []) or [])
    if missing:
        log.write("missing", roles=missing)
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
        missing=missing,
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


def short_job_id(input_path: str | Path) -> str:
    path = Path(input_path)
    if path.is_dir():
        return path.name
    stem = path.stem
    if stem.startswith("grok-image-"):
        parts = stem.split("-")
        if len(parts) >= 3 and parts[2]:
            return parts[2]
    return stem

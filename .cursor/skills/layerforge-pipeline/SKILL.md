---
name: layerforge-pipeline
description: LayerForge v0 pipeline scope, manifest contract, stage order, and bans. Use when running layering, cutting parts, inpainting, exporting packs, or following the project plan.
---

# LayerForge pipeline (v0)

Read `docs/PROJECT_PLAN.md` if a field or stage is unclear. Do not implement `src/layerforge/future/`.

## Scope

In: one flat PNG/WebP/JPEG, or one Imagine job directory.
Out: `out/<job_id>/` with RGBA layers, masks, `manifest.json`, `preview/stack.png`, `preview/diff.png`.

Not in v0: training, video, Spine/Live2D playback, WebUI product, averaging SAM masks.

## Manifest

`schema` must stay `layerforge.manifest.v1`. New meaning = new schema version.
Backends change `manifest.backend.*` names only. Layer fields stay stable.

Roles come from `taxonomy.yaml`, never hardcoded in Python except as a load-time list.

## Stage order

```
ingest → normalize → segment → refine → occlusion → inpaint → reproject → compose/qa → export
```

Each stage writes disk artifacts so a later stage can rerun.

CLI: `python -m layerforge run --input <path> --out runs/`

## Bans

- Do not average SAM multimask outputs. Pick one.
- Do not inpaint visible pixels and keep them. Always reproject original onto `mask_visible`.
- Do not look for occluded pixels with SAM. Occlusion plan + inpaint only.
- GPU backends must error if CUDA is missing. No silent CPU SD.
- Weights live under `model/` (gitignored), paths in `configs/local.yaml`.
- Imagine export shape changes → only `ingest.imagine_parts` + this ingest skill.

## Builder delivery

End code-changing turns with:

```
CHANGE
- 做了：
- 怎么跑：一条命令
- 产物路径：
- 未做 / 已知缺陷：
```

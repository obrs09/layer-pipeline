---
name: imagine-parts-ingest
description: Ingest Grok Imagine segmented exports and flat images into LayerForge. Use when handling Imagine 分块, parts/, segments/, grok-image downloads, or unnamed crops.
---

# Imagine parts ingest

Planner fiction in `docs/PROJECT_PLAN.md` §3.1 is **not** what Grok Imagine actually writes. Prefer the on-disk layout below. Keep the planner layout as a fallback.

## Actual Grok Imagine job (test_input)

```
<job>/
  grok-image-<uuid>.jpg     # full RGB source (JPEG)
  segments/
    <label>.png             # cropped RGBA sprite, real alpha 0/255
    <label>/
      <label>-1.png         # repeated instances of the same label
      <label>-2.png
```

Observed in `test_input/image_partial_sag/image0` and `image1`:

- Source is **not** named `source.png` and is JPEG, not PNG.
- Parts live in `segments/`, not `parts/`.
- Files are **bbox crops**, not full-canvas layers. Must template-match onto source.
- `<label>` is an Imagine caption (`pink-eyes`, `white-and-purple-gown`), not a taxonomy role.
- Some labels are junk (`e.png`). Drop by `refine.min_area`.
- A near-full-figure dump often exists (`white-haired-anime-girl.png`, `anime-girl.png`). Keep it as a back `body` (or alias) so mutex can carve residual skin/hair. Do not skip it.
- Accessory overlays (embroidery, gems) may duplicate garment pixels. Mark those roles `overlay: true` in taxonomy so they do not punch holes in clothes.
- Place **largest sprites first**, then search small crops only inside that character bbox. Same-label repeats blank the matched region before the next match.
- Mutex: smaller non-overlay sprites keep pixels; larger nested dumps lose them. Drop same-role fragments under 25% of the largest sprite (shoulders inside body). Cape vs gown stay separate because they are similar size.

`test_input/image_no_sag/` is **flat** images only — no `segments/`. Use `ingest.flat_image`.

## Fallback planner layout

```
<job>/
  source.png
  parts/part_00.png
  optional_meta.json        # box/label if present
```

If `optional_meta.json` has boxes, use those instead of matching.

## Placement

1. Collect PNGs under `segments/` or `parts/` recursively. Ignore the source image.
2. Label = parent folder if file is `<label>-N.png`, else file stem.
3. Match crop → source with masked NCC (`cv2.matchTemplate` + alpha). Record `bbox = [x, y, w, h]`.
4. Paste onto a full-canvas RGBA. `source` field = `imagine_part`.
5. Map label → role via `taxonomy.yaml` `aliases`. Unmapped → `unknown_N`, original label in `notes`.
6. Same role with L/R in taxonomy (eyes, arms, brows): split instances by centroid x (image-left = `*_l`).
7. Do not invent SAM labels here. Segment backend for Imagine is `noop_from_parts`.

## Detection

A directory is Imagine if it contains `segments/` or `parts/`, plus one source image (`grok-image-*.jpg`, `source.png`, or the sole raster that is not inside those folders).

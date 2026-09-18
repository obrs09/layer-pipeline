---
name: occlusion-inpaint
description: Occlusion masks and inpaint rules for LayerForge. Use when something is covered, extending sleeves/hair, SD1.5, LaMa, or reproject.
---

# Occlusion + inpaint

Visible mask = pixels this layer owns on the source.
Occluded mask = pixels the layer should continue into (covered by a true occluder), **not** decoration overlays.

## Plan

For each completable role (`taxonomy.yaml` `complete: true`):

```
want = dilate(visible, expand_px)
occluders = union of roles in occluded_by (front garments, hair, arms, …)
occluded = want & occluders & ~visible
```

Do not grow into the background. Optionally extend limbs toward the joint listed in taxonomy.

## Inpaint

- Run **only** where `occluded` is set (dilate a few px for seam, still never commit those extra px without reproject).
- Hole area ≤ `inpaint.small_hole_max_px` → LaMa (or OpenCV Telea fallback).
- Larger → SD1.5 anime inpaint.
- No CUDA → hard error for SD. Do not silently CPU-run SD.
- Prompt template: continue existing material and line art; do not add new patterns, text, or extra limbs.

## Reproject (mandatory)

```
out[visible] = source[visible]
```

Inpaint may only remain in `mask_occluded`. If `preview/diff.png` lights up originally-visible pixels, reproject is broken.

## Identity backend

`inpaint.identity` copies the image and leaves holes empty. Valid for dry-run and Imagine path without GPU. Set `complete=false` when occluded area > 0.

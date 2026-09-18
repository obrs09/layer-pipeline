---
name: sam-hint-protocol
description: SAM hint rules for LayerForge. Use when SAM misses parts, hair sticks to face, two pieces merge, or when placing positive/negative points.
---

# SAM hint protocol

Never average masks. Never use SAM to fill occluded regions.

## Multimask

SAM2 may return 3 masks. Keep **one**: highest predicted IoU that also has area in `[min_area, max_area]`. Discard the rest. Do not mean/or them.

## Hints

`SegmentHints`:

- `positive_points`: (x, y) on the piece
- `negative_points`: (x, y) on the neighbor to exclude (hair vs face, sleeve vs torso)
- `boxes`: xyxy, including Imagine-placed bboxes reused as prompts
- `role`: desired taxonomy role if known

If a run merges two pieces: add an exclude point on the wrong piece and recut. Do not blur the two masks together.

## Layered cuts

Cut back-to-front from taxonomy order when possible (body before clothes before hair_front). After each accepted mask, add its interior as **negative** points/mask for the next cut.

## Mutex (post)

Assign each pixel to the front-most non-`overlay` layer that claims it. Overlay accessories do not carve parents.

Drop connected components smaller than `refine.min_area`.

## Occlusion

If the piece is hidden (hair behind head, arm behind torso), **stop SAM**. `ops/occlusion.py` + inpaint own that region.

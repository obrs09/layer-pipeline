---
name: layer-qa-review
description: LayerForge Reviewer checklist for v0 acceptance. Use when reviewing, accepting, checking preview/diff, or validating manifest.
---

# Layer QA / Reviewer

This skill is for the Reviewer Agent. Do not implement features. Do not "fix while reviewing".

## Verdict format

```
VERDICT: PASS | FAIL
CHECKS:
- [x] ...
- [ ] ...
BLOCKERS:
- ...
NITS:
- ...
```

FAIL → Builder only fixes BLOCKERS.

## Checks (must)

1. Still v0. No `future/` implementations, no training, no Spine playback.
2. `manifest.schema` is `layerforge.manifest.v1`.
3. New models go through Protocol + YAML, not hardcoded imports of a specific checkpoint in pipeline code.
4. Visible pixels are original: `preview/diff.png` should be near-black on source-visible regions. Bright visible-area diff = inpaint leak.
5. Tests cover contracts + dry-run.
6. Names are readable (files, roles, functions).
7. Builder `CHANGE` block is honest (known holes listed).

## How to read preview

- `stack.png`: layer composite vs source. Silhouette and color should match on visible parts.
- `diff.png`: abs difference. Ignore true occluded fill (those pixels were not in source). Flag any change where `mask_visible` of some layer was set **and** source already had the character.

## Imagine jobs

Confirm parts were placed (bboxes not all `[0,0,w,h]` unless the crop really is full-canvas). Junk tiny labels should be gone. Whole-figure dump should not sit as a duplicate opaque top layer.

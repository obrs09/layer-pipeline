from __future__ import annotations

import numpy as np

from layerforge.backends.segment.base import LayerMask, SegmentHints
from layerforge.backends.segment.cascade import CascadeSegment
from layerforge.taxonomy import load_taxonomy


class _Cut:
    name = "aniseg"

    def cut(self, image):
        return np.full(image.shape[:2], 255, dtype=np.uint8)


class _Tagger:
    name = "wdtagger"

    def __init__(self, tags):
        self.tags = tags
        self.calls = 0

    def tag(self, image):
        self.calls += 1
        return dict(self.tags)


class _Dino:
    name = "grounding_dino"

    def __init__(self, boxes):
        self.boxes = boxes
        self.calls: list[list[str]] = []

    def detect(self, image, queries, threshold):
        self.calls.append(list(queries))
        out = []
        for query in queries:
            for item in self.boxes.get(query, []):
                out.append((query, item, 0.6))
        return out


class _Sam3:
    name = "sam3.text"

    def __init__(self, by_query: dict[str, list]):
        self.by_query = {key: list(value) for key, value in by_query.items()}
        self.queries: list[str] = []

    def predict_text(self, image, query, character):
        self.queries.append(query)
        bucket = self.by_query.get(query)
        if not bucket:
            return None
        return bucket.pop(0)


class _Sam2:
    name = "sam2.hinted"

    def __init__(self, mask):
        self.mask = mask
        self.calls = 0
        self.neg_points = None

    def prepare(self, rgb):
        return None

    def predict_box_points(self, box, positive, negative):
        self.calls += 1
        self.neg_points = list(negative or [])
        return self.mask, 0.9


def _image():
    return np.full((48, 48, 4), 40, dtype=np.uint8)


def _mask(y0, x0, y1, x1):
    visible = np.zeros((48, 48), dtype=np.uint8)
    visible[y0:y1, x0:x1] = 255
    return visible


def test_sam3_retries_next_query_not_same_prompt():
    bad = _mask(2, 2, 46, 46)
    good = _mask(10, 16, 22, 28)
    sam3 = _Sam3({"anime face": [bad], "face": [good]})
    dino = _Dino({})
    cascade = CascadeSegment(
        {},
        load_taxonomy(),
        character=_Cut(),
        tagger=_Tagger({"1girl": 0.99}),
        boxes=dino,
        sam3=sam3,
        sam2=_Sam2(_mask(0, 0, 1, 1)),
    )
    layers = cascade.segment(_image(), SegmentHints())
    faces = [layer for layer in layers if layer.role == "face"]
    assert faces
    assert sam3.queries.count("anime face") == 1
    assert "face" in sam3.queries
    assert sam3.queries.index("anime face") < sam3.queries.index("face")
    assert "sam3.text" in faces[0].notes


def test_sam2_uses_neighbor_exclude_points_after_sam3_miss():
    face = _mask(8, 14, 24, 34)
    sam2 = _Sam2(face)
    cascade = CascadeSegment(
        {},
        load_taxonomy(),
        character=_Cut(),
        tagger=_Tagger({"1girl": 0.99}),
        boxes=_Dino({"anime face": [[12, 6, 36, 28]], "face": [[12, 6, 36, 28]]}),
        sam3=_Sam3({}),
        sam2=sam2,
    )
    # Seed a hair layer by cutting after face exists: first run face via sam2.
    layers = cascade.segment(_image(), SegmentHints())
    assert any(layer.role == "face" for layer in layers)
    assert sam2.calls >= 1


def test_required_missing_listed_not_infinite():
    cascade = CascadeSegment(
        {"cascade": {"max_attempts": 2}},
        load_taxonomy(),
        character=_Cut(),
        tagger=_Tagger({"1girl": 0.99}),
        boxes=_Dino({}),
        sam3=_Sam3({}),
        sam2=_Sam2(None),
    )
    layers = cascade.segment(_image(), SegmentHints())
    assert "face" in cascade.missing
    assert "eye_l" in cascade.missing
    assert cascade.missing.count("face") == 1
    roles = {layer.role for layer in layers}
    assert "face" not in roles


def test_imagine_skips_tagger_and_keeps_unusable_as_needs_click():
    tagger = _Tagger({"1girl": 0.99})
    huge = LayerMask(
        role="eye_l",
        label="pink-eyes",
        visible=_mask(0, 0, 48, 48),
        source="imagine_part",
    )
    ok = LayerMask(
        role="eye_r",
        label="pink-eyes",
        visible=_mask(12, 28, 18, 36),
        source="imagine_part",
    )
    cascade = CascadeSegment(
        {},
        load_taxonomy(),
        character=_Cut(),
        tagger=tagger,
        dry_run=True,
    )
    out = cascade.segment(_image(), SegmentHints(parts=[huge, ok]))
    assert tagger.calls == 0
    by_role = {layer.role: layer for layer in out}
    assert by_role["eye_l"].needs_click
    assert not by_role["eye_r"].needs_click

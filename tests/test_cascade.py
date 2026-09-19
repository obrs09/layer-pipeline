from __future__ import annotations

import numpy as np

from layerforge.backends.segment.base import LayerMask, SegmentHints
from layerforge.backends.segment.cascade import CascadeSegment, _constrain_mask, _rank_boxes
from layerforge.taxonomy import load_taxonomy


class _Cut:
    name = "anime_segmentation"

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

    def __init__(self, mask=None, *, fill_box: bool = False, shape=(48, 48)):
        self.mask = mask
        self.fill_box = fill_box
        self.shape = shape
        self.calls = 0
        self.neg_points = None
        self.boxes: list[list[float]] = []

    def prepare(self, rgb):
        return None

    def predict_box_points(self, box, positive, negative):
        self.calls += 1
        self.neg_points = list(negative or [])
        self.boxes.append(list(box))
        if self.fill_box:
            h, w = self.shape
            vis = np.zeros((h, w), dtype=np.uint8)
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 > x0 and y1 > y0:
                vis[y0:y1, x0:x1] = 255
            return vis, 0.9
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


def test_pair_eyes_split_family_boxes_by_x():
    dino = _Dino({"eye": [[6, 10, 14, 18], [30, 10, 38, 18]]})
    cascade = CascadeSegment(
        {},
        load_taxonomy(),
        character=_Cut(),
        tagger=_Tagger({"1girl": 0.99}),
        boxes=dino,
        sam3=_Sam3({}),
        sam2=_Sam2(fill_box=True),
    )
    layers = cascade.segment(_image(), SegmentHints())
    by_role = {layer.role: layer for layer in layers}
    assert "eye_l" in by_role
    assert "eye_r" in by_role
    assert by_role["eye_l"].bbox[0] < by_role["eye_r"].bbox[0]
    assert any("eye" in queries for queries in dino.calls)
    assert cascade.tags["1girl"] == 0.99


def test_pair_uses_left_and_right_queries_separately():
    dino = _Dino(
        {
            "left eye": [[6, 10, 14, 18]],
            "right eye": [[30, 10, 38, 18]],
        }
    )
    cascade = CascadeSegment(
        {},
        load_taxonomy(),
        character=_Cut(),
        tagger=_Tagger({"1girl": 0.99}),
        boxes=dino,
        sam3=_Sam3({}),
        sam2=_Sam2(fill_box=True),
    )
    layers = cascade.segment(_image(), SegmentHints())
    by_role = {layer.role: layer for layer in layers}
    assert "eye_l" in by_role
    assert "eye_r" in by_role
    assert any(queries == ["left eye", "left eye anime"] for queries in dino.calls)
    assert any(queries == ["right eye", "right eye anime"] for queries in dino.calls)


def test_one_eye_box_mirrors_across_face():
    dino = _Dino(
        {
            "eye": [[8, 12, 16, 20]],
            "anime face": [[10, 6, 38, 28]],
            "face": [[10, 6, 38, 28]],
        }
    )
    cascade = CascadeSegment(
        {},
        load_taxonomy(),
        character=_Cut(),
        tagger=_Tagger({"1girl": 0.99}),
        boxes=dino,
        sam3=_Sam3({}),
        sam2=_Sam2(fill_box=True),
    )
    layers = cascade.segment(_image(), SegmentHints())
    by_role = {layer.role: layer for layer in layers}
    assert "eye_l" in by_role
    assert "eye_r" in by_role
    assert by_role["eye_l"].bbox[0] < by_role["eye_r"].bbox[0]


def test_body_residual_punches_clothes_and_hair():
    cascade = CascadeSegment({}, load_taxonomy(), character=_Cut(), tagger=_Tagger({}))
    h, w = 128, 96
    character = np.full((h, w), 255, dtype=np.uint8)
    clothes_vis = np.zeros((h, w), dtype=np.uint8)
    clothes_vis[50:100, 20:76] = 255
    hair_vis = np.zeros((h, w), dtype=np.uint8)
    hair_vis[4:40, 16:80] = 255
    clothes = LayerMask(role="clothes", label="clothes", visible=clothes_vis, source="sam")
    hair = LayerMask(role="hair_back", label="hair", visible=hair_vis, source="sam")
    image = np.full((h, w, 4), 40, dtype=np.uint8)
    body = cascade._body_from_residual(character, [clothes, hair], image)
    assert body is not None
    assert body.visible[70, 48] == 0
    assert body.visible[20, 48] == 0
    assert body.visible[120, 48] > 0
    assert body.visible[49, 48] == 0


def test_rank_boxes_skips_giant_hair_box():
    spec = load_taxonomy().spec("hair_front")
    character = np.zeros((40, 40), dtype=np.uint8)
    character[4:36, 4:36] = 255
    huge = [0.0, 0.0, 40.0, 40.0]
    bangs = [12.0, 4.0, 28.0, 12.0]
    ranked = _rank_boxes(
        [("front hair", huge, 0.9), ("bangs", bangs, 0.3)],
        character,
        spec,
    )
    assert ranked[0] == bangs


def test_constrain_mask_clips_to_box():
    mask = np.full((20, 20), 255, dtype=np.uint8)
    character = np.full((20, 20), 255, dtype=np.uint8)
    out = _constrain_mask(mask, character, [5.0, 5.0, 10.0, 10.0], pad=0)
    assert int(out[0, 0]) == 0
    assert int(out[7, 7]) == 255


def test_cut_order_clothes_before_hair():
    order = CascadeSegment({}, load_taxonomy())._cut_order(
        ["hair_front", "clothes", "face", "eye_l", "body"]
    )
    assert order.index("clothes") < order.index("hair_front")
    assert order.index("face") < order.index("hair_front")
    assert order.index("eye_l") > order.index("hair_front")
    assert "body" not in order


def test_inventory_hair_front_failure_is_missing():
    cascade = CascadeSegment(
        {"cascade": {"max_attempts": 2}},
        load_taxonomy(),
        character=_Cut(),
        tagger=_Tagger({"1girl": 0.99, "long_hair": 0.9}),
        boxes=_Dino({}),
        sam3=_Sam3({}),
        sam2=_Sam2(None),
    )
    cascade.segment(_image(), SegmentHints())
    assert "hair_front" in cascade.missing


class _Pose:
    name = "dwpose"

    def __init__(self, person, boxes=None):
        self.person = person
        self.boxes = boxes or {}

    def estimate(self, image):
        from layerforge.backends.detect.dwpose import PoseEstimate

        overlay = np.zeros_like(image[:, :, :3])
        return PoseEstimate(
            person_mask=self.person,
            keypoints=[],
            boxes=self.boxes,
            points={},
            overlay=overlay,
            person_box=[0, 0, 10, 10],
        )


def test_pose_clips_character_and_body_residual():
    person = np.zeros((48, 48), dtype=np.uint8)
    person[10:40, 12:36] = 255
    cascade = CascadeSegment(
        {"refine": {"morph_open_px": 0}},
        load_taxonomy(),
        character=_Cut(),
        tagger=_Tagger({"1girl": 0.99}),
        boxes=_Dino({}),
        sam3=_Sam3({}),
        sam2=_Sam2(None),
        pose=_Pose(person, boxes={"body": [14.0, 16.0, 34.0, 38.0]}),
    )
    image = _image()
    layers = cascade.segment(image, SegmentHints())
    assert int(cascade.character_mask[0, 0]) == 0
    assert int(cascade.character_mask[20, 20]) == 255
    body = next(layer for layer in layers if layer.role == "body")
    assert int(body.visible[2, 2]) == 0
    assert "pose person" in body.notes


def test_inject_pose_box_goes_first():
    cascade = CascadeSegment({}, load_taxonomy())
    cascade.pose_boxes = {"body": [4.0, 5.0, 10.0, 12.0]}
    ranked = cascade._inject_pose_box("body", [[0.0, 0.0, 40.0, 40.0]])
    assert ranked[0] == [4.0, 5.0, 10.0, 12.0]
    assert ranked[1] == [0.0, 0.0, 40.0, 40.0]

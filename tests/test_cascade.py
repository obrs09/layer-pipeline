from __future__ import annotations

import numpy as np

from layerforge.backends.segment.base import LayerMask, SegmentHints
from layerforge.backends.segment.cascade import (
    CascadeSegment,
    _constrain_mask,
    _neighbor_negatives,
    _rank_boxes,
)
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


class _Sam2Crop(_Sam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crop_calls = 0

    def predict_on_crop(self, box, positive, negative, pad_frac=0.18, min_side=1024):
        self.crop_calls += 1
        return self.predict_box_points(box, positive, negative)


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


def test_cut_order_clothes_hair_then_face():
    order = CascadeSegment({}, load_taxonomy())._cut_order(
        ["hair_front", "clothes", "face", "eye_l", "hair_back", "body"]
    )
    assert "hair_front" not in order
    assert order.index("clothes") < order.index("hair_back")
    assert order.index("hair_back") < order.index("face")
    assert order.index("face") < order.index("eye_l")
    assert "body" not in order
    assert "neck" not in CascadeSegment({}, load_taxonomy())._cut_order(
        ["neck", "clothes", "face", "hair_back"]
    )


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


def test_pose_does_not_clip_character():
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
    cascade.segment(image, SegmentHints())
    assert int(cascade.character_mask[0, 0]) == 255
    assert int(cascade.character_mask[20, 20]) == 255
    assert cascade.pose_boxes["body"] == [14.0, 16.0, 34.0, 38.0]


def _blob(h, w, y0, x0, y1, x1):
    out = np.zeros((h, w), dtype=np.uint8)
    out[y0:y1, x0:x1] = 255
    return out


def test_hair_from_residual_around_head_when_sam_hair_failed():
    """64010c19: hair in inventory, SAM returned the whole figure, hair pixels were left over."""
    h, w = 160, 160
    character = _blob(h, w, 10, 20, 150, 140)
    face = LayerMask(role="face", label="face", visible=_blob(h, w, 30, 60, 70, 100), source="sam")
    clothes = LayerMask(role="clothes", label="clothes", visible=_blob(h, w, 70, 40, 150, 120), source="sam")
    cascade = CascadeSegment({"refine": {"morph_open_px": 0}}, load_taxonomy(), character=_Cut())
    cascade.inventory = ["face", "clothes", "hair_back", "hair_front", "body"]
    cascade.debug_boxes = [{"role": "hair_back", "query": "hair", "xyxy": [20.0, 10.0, 140.0, 80.0], "score": 0.6}]
    hair = cascade._hair_from_residual(character, [face, clothes])
    assert hair is not None
    assert hair.role == "hair_back"
    # residual beside the head (x 20..60 / 100..140, y 10..70) is hair
    assert int(hair.visible[40, 30]) == 255
    assert int(hair.visible[40, 120]) == 255
    # residual below the clothes hem is not hair
    assert int(hair.visible[155, 80]) == 0 if h > 155 else True
    assert not np.any((hair.visible > 0) & (face.visible > 0))
    assert not np.any((hair.visible > 0) & (clothes.visible > 0))


def test_hair_from_residual_not_used_without_hair_in_inventory():
    h, w = 160, 160
    character = _blob(h, w, 10, 20, 150, 140)
    face = LayerMask(role="face", label="face", visible=_blob(h, w, 30, 60, 70, 100), source="sam")
    cascade = CascadeSegment({}, load_taxonomy(), character=_Cut())
    cascade.inventory = ["face", "clothes", "body"]
    assert cascade._hair_from_residual(character, [face]) is None
    cascade.inventory = ["face", "hair_back", "body"]
    existing = LayerMask(role="hair_back", label="hair", visible=_blob(h, w, 10, 20, 30, 140), source="sam")
    assert cascade._hair_from_residual(character, [face, existing]) is None


def test_hair_from_residual_ignores_staff_centroid_in_hair_box():
    """64010c19: a mid-size DINO hair box can cover the staff; centroid-in-box is not hair."""
    h, w = 160, 220
    character = _blob(h, w, 10, 90, 150, 210)
    character[20:70, 16:55] = 255
    face = LayerMask(role="face", label="face", visible=_blob(h, w, 30, 120, 70, 170), source="sam")
    clothes = LayerMask(role="clothes", label="clothes", visible=_blob(h, w, 70, 90, 150, 190), source="sam")
    cascade = CascadeSegment({"refine": {"morph_open_px": 0}}, load_taxonomy(), character=_Cut())
    cascade.inventory = ["face", "clothes", "hair_back", "body"]
    cascade.debug_boxes = [{"role": "hair_back", "query": "hair", "xyxy": [16.0, 10.0, 200.0, 80.0], "score": 0.6}]
    hair = cascade._hair_from_residual(character, [face, clothes])
    assert hair is not None
    assert int(hair.visible[40, 100]) == 255
    assert int((hair.visible[20:70, 16:55] > 0).sum()) == 0


def test_hair_from_residual_pose_arms_split_held_staff():
    """Staff glued to leftover hair through the arm is punched by pose arm boxes."""
    h, w = 160, 220
    character = _blob(h, w, 10, 90, 150, 210)
    character[20:70, 16:40] = 255
    character[40:55, 40:120] = 255
    face = LayerMask(role="face", label="face", visible=_blob(h, w, 30, 120, 70, 170), source="sam")
    clothes = LayerMask(role="clothes", label="clothes", visible=_blob(h, w, 70, 110, 150, 190), source="sam")
    cascade = CascadeSegment({"refine": {"morph_open_px": 0}}, load_taxonomy(), character=_Cut())
    cascade.inventory = ["face", "clothes", "hair_back", "body"]
    cascade.debug_boxes = [{"role": "hair_back", "query": "hair", "xyxy": [16.0, 10.0, 200.0, 90.0], "score": 0.67}]
    cascade.pose_points = {"arm_r": [(80.0, 48.0)]}
    hair = cascade._hair_from_residual(character, [face, clothes])
    assert hair is not None
    assert int(hair.visible[40, 180]) == 255
    assert int((hair.visible[20:70, 16:40] > 0).sum()) == 0
    assert int((hair.visible[40:55, 40:90] > 0).sum()) == 0


def test_hair_from_residual_leaves_far_blobs_for_body():
    h, w = 200, 160
    character = _blob(h, w, 10, 20, 190, 140)
    face = LayerMask(role="face", label="face", visible=_blob(h, w, 20, 60, 60, 100), source="sam")
    clothes = LayerMask(role="clothes", label="clothes", visible=_blob(h, w, 60, 20, 150, 140), source="sam")
    cascade = CascadeSegment({"refine": {"morph_open_px": 0}}, load_taxonomy(), character=_Cut())
    cascade.inventory = ["face", "clothes", "hair_back", "body"]
    # a whole-figure "hair" box was tried too; it must not widen the hair region
    cascade.debug_boxes = [{"role": "hair_back", "query": "hair", "xyxy": [20.0, 10.0, 140.0, 190.0], "score": 0.3}]
    hair = cascade._hair_from_residual(character, [face, clothes])
    assert hair is not None
    # legs below the hem (y 150..190) are far from the head zone and in no hair box
    assert int((hair.visible[150:190, :] > 0).sum()) == 0
    # residual beside the face, inside the head zone, is hair
    assert int(hair.visible[30, 45]) == 255
    # residual far left of the head zone, chained only through the silhouette rim, is not
    assert int(hair.visible[30, 22]) == 0


def test_hair_split_occlusion_peels_back():
    h, w = 80, 60
    hair_vis = np.zeros((h, w), dtype=np.uint8)
    hair_vis[5:50, 10:50] = 255
    face_vis = np.zeros((h, w), dtype=np.uint8)
    face_vis[20:42, 18:42] = 255
    clothes = np.zeros((h, w), dtype=np.uint8)
    clothes[44:80, :] = 255
    cascade = CascadeSegment(
        {
            "cascade": {
                "hair_split": {
                    "enabled": True,
                    "strategy": "occlusion",
                    "compare": False,
                    "peel_back": True,
                    "min_front_px": 16,
                    "occlusion": {"grow_px": 8, "forehead_frac": 0.4, "bangs_up_frac": 0.1},
                }
            }
        },
        load_taxonomy(),
        character=_Cut(),
    )
    cascade.inventory = ["hair_back", "hair_front", "face"]
    kept = [
        LayerMask(role="hair_back", label="hair", visible=hair_vis.copy(), source="sam"),
        LayerMask(role="face", label="face", visible=face_vis, source="sam"),
        LayerMask(role="clothes", label="clothes", visible=clothes, source="sam"),
    ]
    image = np.zeros((h, w, 3), dtype=np.uint8)
    cascade._split_hair_front(image, kept)
    by_role = {layer.role: layer for layer in kept}
    assert "hair_front" in by_role
    assert "occlusion" in by_role["hair_front"].notes
    assert int(by_role["hair_front"].visible[22, 30]) == 255
    assert int(by_role["hair_front"].visible[48, 30]) == 0
    assert int(by_role["hair_back"].visible[48, 30]) == 255
    assert int((by_role["hair_front"].visible > 0).sum()) + int((by_role["hair_back"].visible > 0).sum()) == int(
        (hair_vis > 0).sum()
    )


def test_whole_hair_dump_written():
    class _Dump:
        def __init__(self):
            self.pngs: dict[str, np.ndarray] = {}
            self.jsons: dict[str, object] = {}

        def write_png(self, rel, array):
            self.pngs[rel] = array

        def write_json(self, rel, payload):
            self.jsons[rel] = payload

    hair = LayerMask(role="hair_back", label="hair", visible=_blob(40, 40, 4, 6, 28, 30), source="sam")
    dump = _Dump()
    cascade = CascadeSegment({}, load_taxonomy())
    cascade.bind_dump(dump)
    cascade._dump_hair_whole(np.full((40, 40, 4), 30, dtype=np.uint8), hair)
    assert "04_segment/hair_whole.png" in dump.pngs
    assert "04_segment/hair_whole.mask.png" in dump.pngs
    assert dump.jsons["04_segment/hair_whole.json"]["px"] == int((hair.visible > 0).sum())


def test_face_placeholder_is_not_kept():
    dino = _Dino(
        {
            "hair": [[4, 2, 44, 40]],
            "anime face": [[12, 10, 36, 28]],
            "face": [[12, 10, 36, 28]],
            "anime clothes": [[8, 24, 40, 44]],
        }
    )
    cascade = CascadeSegment(
        {},
        load_taxonomy(),
        character=_Cut(),
        tagger=_Tagger({"1girl": 0.99, "long_hair": 0.9}),
        boxes=dino,
        sam2=_Sam2(fill_box=True),
    )
    layers = cascade.segment(_image(), SegmentHints())
    assert cascade.peek_boxes.get("face")
    assert all(layer.source != "placeholder" for layer in layers)
    assert all(layer.role != "face" or "placeholder" not in (layer.notes or "") for layer in layers)


def test_hair_positive_not_face_center():
    from layerforge.ops.hair_hint import fill_box

    character = np.ones((48, 48), dtype=np.uint8) * 255
    face = LayerMask(
        role="face",
        label="face",
        visible=fill_box((48, 48), [14.0, 16.0, 34.0, 32.0]),
        source="placeholder",
    )
    clothes = LayerMask(role="clothes", label="clothes", visible=_mask(28, 8, 46, 40), source="sam")
    cascade = CascadeSegment({}, load_taxonomy())
    image = np.full((48, 48, 3), 70, dtype=np.uint8)
    image[2:14, 8:40] = (40, 40, 170)
    points = cascade._positive_points(image, character, "hair_back", [8.0, 2.0, 40.0, 40.0], [clothes, face])
    assert points
    x, y = points[0]
    assert y < 16.0
    assert not (14.0 <= x <= 34.0 and 16.0 <= y <= 32.0)


def test_whole_hair_cut_does_not_query_back_hair():
    dino = _Dino(
        {
            "hair": [[6, 4, 42, 36]],
            "anime clothes": [[8, 20, 40, 44]],
            "anime face": [[14, 8, 34, 26]],
            "face": [[14, 8, 34, 26]],
        }
    )
    cascade = CascadeSegment(
        {},
        load_taxonomy(),
        character=_Cut(),
        tagger=_Tagger({"1girl": 0.99, "long_hair": 0.9}),
        boxes=dino,
        sam2=_Sam2(fill_box=True),
    )
    cascade.segment(_image(), SegmentHints())
    queried = [query for call in dino.calls for query in call]
    assert "hair" in queried
    assert "back hair" not in queried
    assert "bangs" not in queried
    assert "front hair" not in queried


def test_hair_split_skipped_when_disabled():
    h, w = 40, 40
    hair = LayerMask(role="hair_back", label="hair", visible=_blob(h, w, 2, 4, 30, 36), source="sam")
    face = LayerMask(role="face", label="face", visible=_blob(h, w, 8, 12, 22, 28), source="sam")
    cascade = CascadeSegment({"cascade": {"hair_split": {"enabled": False}}}, load_taxonomy())
    cascade.inventory = ["hair_back", "hair_front", "face"]
    kept = [hair, face]
    cascade._split_hair_front(np.zeros((h, w, 3), dtype=np.uint8), kept)
    assert all(layer.role != "hair_front" for layer in kept)


def test_sam_crop_used_for_face():
    sam2 = _Sam2Crop(fill_box=True)
    cascade = CascadeSegment(
        {"cascade": {"sam_crop": {"enabled": True, "roles": ["face"], "min_side": 32}}},
        load_taxonomy(),
        character=_Cut(),
        tagger=_Tagger({"1girl": 0.99}),
        boxes=_Dino({"anime face": [[12, 6, 36, 28]], "face": [[12, 6, 36, 28]]}),
        sam3=_Sam3({}),
        sam2=sam2,
    )
    layers = cascade.segment(_image(), SegmentHints())
    assert any(layer.role == "face" for layer in layers)
    assert sam2.crop_calls >= 1


def test_neighbor_negatives_avoid_face_box():
    hair = np.zeros((40, 40), dtype=np.uint8)
    hair[2:6, 10:30] = 255
    hair[10:24, 12:28] = 255
    layer = LayerMask(role="hair_back", label="hair", visible=hair, source="sam")
    avoided = _neighbor_negatives([layer], ("hair_back",), avoid_box=[10.0, 8.0, 30.0, 26.0])
    raw = _neighbor_negatives([layer], ("hair_back",))
    assert avoided and raw
    assert avoided[0][1] < 8.0
    assert raw[0][1] > avoided[0][1]


def test_inject_pose_box_goes_first():
    cascade = CascadeSegment({}, load_taxonomy())
    cascade.pose_boxes = {"body": [4.0, 5.0, 10.0, 12.0]}
    ranked = cascade._inject_pose_box("body", [[0.0, 0.0, 40.0, 40.0]])
    assert ranked[0] == [4.0, 5.0, 10.0, 12.0]
    assert ranked[1] == [0.0, 0.0, 40.0, 40.0]

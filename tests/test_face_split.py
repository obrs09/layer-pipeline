from __future__ import annotations

import numpy as np

from layerforge.backends.segment.base import LayerMask
from layerforge.ops.face_split import split_face_colors
from layerforge.taxonomy import load_taxonomy


def _cfg(**overrides) -> dict:
    cfg = {
        "enabled": True,
        "expand_px": 1,
        "punch_roles": ["eye_l", "eye_r", "mouth"],
        "neck_role": "neck",
        "hair_role": "hair_front",
        "hair_fallback": "hair_back",
        "skin_dist": 24,
        "hair_dist": 24,
        "hair_slack": 8,
        "l_weight": 0.08,
        "jaw_frac": 0.82,
        "jaw_pad_px": 2,
        "neck_width_frac": 0.62,
        "min_neck_px": 40,
        "min_hair_px": 20,
        "cheek_dilate_px": 12,
        "skin_mode": "grow",
        "local_dist": 20,
        "cut_neck": False,
        "edge_stop": True,
        "canny_low": 50,
        "canny_high": 120,
    }
    cfg.update(overrides)
    return cfg


def _layer(role: str, vis: np.ndarray) -> LayerMask:
    return LayerMask(role=role, label=role, visible=vis.astype(np.uint8) * 255, source="sam")


def _scene():
    h, w = 80, 64
    image = np.zeros((h, w, 3), dtype=np.uint8)
    skin = (220, 168, 148)
    hair_c = (190, 195, 215)
    other_c = (30, 200, 40)
    face_vis = np.zeros((h, w), dtype=bool)
    face_vis[8:64, 14:50] = True
    image[8:50, 14:50] = skin
    image[50:64, 22:42] = skin
    image[8:22, 14:20] = hair_c
    image[8:22, 44:50] = hair_c
    image[28:34, 14:18] = other_c
    eye_l = np.zeros((h, w), dtype=bool)
    eye_r = np.zeros((h, w), dtype=bool)
    mouth = np.zeros((h, w), dtype=bool)
    eye_l[22:28, 22:28] = True
    eye_r[22:28, 36:42] = True
    mouth[38:44, 28:36] = True
    image[eye_l] = (40, 20, 20)
    image[eye_r] = (40, 20, 20)
    image[mouth] = (180, 70, 90)
    layers = [
        _layer("face", face_vis),
        _layer("eye_l", eye_l),
        _layer("eye_r", eye_r),
        _layer("mouth", mouth),
        _layer("hair_front", np.zeros((h, w), dtype=bool)),
    ]
    return image, layers


def test_face_split_disabled_is_noop():
    image, layers = _scene()
    before = int((layers[0].visible > 0).sum())
    out, report = split_face_colors(image, layers, load_taxonomy(), {})
    assert report["enabled"] is False
    assert int((out[0].visible > 0).sum()) == before


def test_face_split_peels_hair_and_other():
    image, layers = _scene()
    out, report = split_face_colors(image, layers, load_taxonomy(), _cfg())
    by_role = {layer.role: layer for layer in out}
    assert report["applied"]
    face = by_role["face"].visible > 0
    hair = by_role["hair_front"].visible > 0
    assert int(face.sum()) > 0
    assert int(hair.sum()) >= 20
    assert report.get("cut_neck") is False
    assert "neck" not in by_role
    # Gray bangs left the face and landed on hair.
    assert not face[12, 16]
    assert hair[12, 16]
    assert not face[12, 47]
    # Chin/lower skin stays on the face while neck cut is off.
    assert face[56, 30]
    # Eyes/mouth stay punched out of face.
    assert not face[24, 24]
    assert not face[40, 32]
    # Green leftover is unclaimed: not face, not hair.
    assert not face[30, 15]
    assert not hair[30, 15]
    assert report["unclaimed_px"] >= 10


def test_face_split_without_hair_layer_creates_hair_front():
    image, layers = _scene()
    layers = [layer for layer in layers if layer.role != "hair_front"]
    out, report = split_face_colors(image, layers, load_taxonomy(), _cfg())
    assert report.get("hair_role") == "hair_front"
    hair = [layer for layer in out if layer.role == "hair_front"]
    assert hair and int((hair[0].visible > 0).sum()) >= 20
    assert hair[0].source == "sam"


def test_white_hair_loses_to_hair_seed_not_skin():
    h, w = 48, 40
    image = np.zeros((h, w, 3), dtype=np.uint8)
    vis = np.zeros((h, w), dtype=bool)
    vis[6:40, 8:32] = True
    image[6:40, 8:32] = (225, 175, 155)
    image[6:18, 8:14] = (245, 245, 250)
    image[6:18, 26:32] = (245, 245, 250)
    hair_layer = np.zeros((h, w), dtype=bool)
    hair_layer[4:12, 4:36] = True
    image[4:12, 4:36] = (245, 245, 250)
    layers = [
        _layer("face", vis),
        _layer("hair_front", hair_layer),
        _layer("eye_l", np.zeros((h, w), dtype=bool)),
        _layer("eye_r", np.zeros((h, w), dtype=bool)),
    ]
    out, report = split_face_colors(image, layers, load_taxonomy(), _cfg(min_neck_px=400))
    face = next(layer for layer in out if layer.role == "face").visible > 0
    hair = next(layer for layer in out if layer.role == "hair_front").visible > 0
    assert report["applied"]
    assert not face[10, 10]
    assert hair[10, 10]
    assert face[20, 20]


def test_cut_neck_can_still_split_when_enabled():
    image, layers = _scene()
    out, report = split_face_colors(image, layers, load_taxonomy(), _cfg(cut_neck=True))
    neck = next(layer for layer in out if layer.role == "neck")
    face = next(layer for layer in out if layer.role == "face").visible > 0
    assert report.get("cut_neck") is True
    assert int((neck.visible > 0).sum()) >= 40
    assert neck.visible[56, 30]
    assert not face[56, 30]


def test_second_split_does_not_recut_neck():
    image, layers = _scene()
    once, _ = split_face_colors(image, layers, load_taxonomy(), _cfg(cut_neck=True))
    neck = next(layer for layer in once if layer.role == "neck")
    px = int((neck.visible > 0).sum())
    twice, report = split_face_colors(image, once, load_taxonomy(), _cfg(cut_neck=True))
    neck2 = next(layer for layer in twice if layer.role == "neck")
    assert report.get("neck_skipped") is True
    assert int((neck2.visible > 0).sum()) == px


def test_bangs_prefer_hair_front_even_if_hair_back_exists():
    image, layers = _scene()
    layers = [layer for layer in layers if layer.role != "hair_front"]
    back = np.zeros(image.shape[:2], dtype=bool)
    back[70:78, 20:40] = True
    image[70:78, 20:40] = (190, 195, 215)
    layers.append(_layer("hair_back", back))
    out, report = split_face_colors(image, layers, load_taxonomy(), _cfg())
    assert report.get("hair_role") == "hair_front"
    front = next(layer for layer in out if layer.role == "hair_front")
    assert int((front.visible > 0).sum()) >= 20
    assert front.visible[12, 16]


def test_grow_crosses_blush_but_stops_at_hair():
    """Pale skin next to blush should still join; a gray bang should not."""
    h, w = 48, 36
    image = np.zeros((h, w, 3), dtype=np.uint8)
    vis = np.zeros((h, w), dtype=bool)
    vis[6:40, 6:30] = True
    image[6:40, 6:30] = (225, 180, 165)
    image[14:28, 14:30] = (220, 150, 140)
    image[6:14, 6:16] = (210, 210, 220)
    eye = np.zeros((h, w), dtype=bool)
    eye[18:22, 18:22] = True
    layers = [
        _layer("face", vis),
        _layer("eye_l", eye),
        _layer("eye_r", np.zeros((h, w), dtype=bool)),
    ]
    out, report = split_face_colors(image, layers, load_taxonomy(), _cfg(min_neck_px=400))
    face = next(layer for layer in out if layer.role == "face").visible > 0
    assert report.get("skin_mode") == "grow"
    assert face[22, 22]
    assert face[30, 18]
    assert not face[8, 10]


def test_grow_stops_at_sharp_hair_edge():
    """A connected gray strip should not join peach skin just because it touches."""
    h, w = 40, 32
    image = np.zeros((h, w, 3), dtype=np.uint8)
    vis = np.zeros((h, w), dtype=bool)
    vis[8:32, 6:26] = True
    image[8:32, 6:26] = (220, 168, 148)
    image[8:16, 6:12] = (200, 200, 210)
    eye = np.zeros((h, w), dtype=bool)
    eye[18:22, 12:16] = True
    layers = [
        _layer("face", vis),
        _layer("eye_l", eye),
        _layer("eye_r", np.zeros((h, w), dtype=bool)),
    ]
    out, report = split_face_colors(image, layers, load_taxonomy(), _cfg(min_neck_px=400))
    face = next(layer for layer in out if layer.role == "face").visible > 0
    assert report.get("skin_mode") == "grow"
    assert face[24, 16]
    assert not face[10, 8]


def test_grow_keeps_pale_highlight_near_hair():
    """A peach highlight a bit closer to hair still stays face (hair_slack)."""
    h, w = 40, 32
    image = np.zeros((h, w, 3), dtype=np.uint8)
    vis = np.zeros((h, w), dtype=bool)
    vis[8:32, 6:26] = True
    image[8:32, 6:26] = (220, 168, 148)
    image[18:24, 12:20] = (232, 196, 186)
    hair = np.zeros((h, w), dtype=bool)
    hair[2:8, 6:26] = True
    image[2:8, 6:26] = (200, 200, 210)
    eye = np.zeros((h, w), dtype=bool)
    eye[12:16, 12:16] = True
    layers = [
        _layer("face", vis),
        _layer("eye_l", eye),
        _layer("eye_r", np.zeros((h, w), dtype=bool)),
        _layer("hair_front", hair),
    ]
    out, report = split_face_colors(image, layers, load_taxonomy(), _cfg(min_neck_px=400))
    face = next(layer for layer in out if layer.role == "face").visible > 0
    assert report.get("skin_mode") == "grow"
    assert face[20, 16]


def test_grow_stops_at_lineart():
    """A dark outline between peach and pale hair should halt grow."""
    h, w = 40, 32
    image = np.zeros((h, w, 3), dtype=np.uint8)
    vis = np.zeros((h, w), dtype=bool)
    vis[8:32, 6:26] = True
    image[8:32, 6:26] = (220, 168, 148)
    image[8:18, 6:13] = (210, 210, 220)
    image[8:18, 13] = (20, 16, 16)
    eye = np.zeros((h, w), dtype=bool)
    eye[20:24, 16:20] = True
    layers = [
        _layer("face", vis),
        _layer("eye_l", eye),
        _layer("eye_r", np.zeros((h, w), dtype=bool)),
    ]
    out, report = split_face_colors(image, layers, load_taxonomy(), _cfg(min_neck_px=400))
    face = next(layer for layer in out if layer.role == "face").visible > 0
    assert report.get("edge_stop") is True
    assert face[24, 18]
    assert not face[12, 8]

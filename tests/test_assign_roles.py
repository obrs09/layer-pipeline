from __future__ import annotations

import numpy as np

from layerforge.backends.segment.base import LayerMask
from layerforge.ops.assign_roles import assign_roles
from layerforge.taxonomy import load_taxonomy


def _layer(role: str, visible: np.ndarray, label: str = "") -> LayerMask:
    return LayerMask(role=role, label=label or role, visible=visible, source="sam")


def _canvas() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = 64, 40
    rgb = np.full((h, w, 3), 250, dtype=np.uint8)
    hair = np.zeros((h, w), dtype=np.uint8)
    hair[2:16, 8:32] = 255
    rgb[2:16, 8:32] = (180, 40, 160)
    face = np.zeros((h, w), dtype=np.uint8)
    face[14:28, 12:28] = 255
    rgb[14:28, 12:28] = (240, 190, 160)
    clothes = np.zeros((h, w), dtype=np.uint8)
    clothes[28:50, 10:30] = 255
    rgb[28:50, 10:30] = (40, 60, 200)
    body = np.zeros((h, w), dtype=np.uint8)
    body[16:58, 10:30] = 255
    rgb[50:58, 14:26] = (240, 190, 160)
    rgba = np.dstack([rgb, np.full((h, w), 255, dtype=np.uint8)])
    return rgba, body, hair, face, clothes


def test_assigns_hair_face_clothes_on_flat_unknowns():
    rgba, body, hair, face, clothes = _canvas()
    layers = [
        _layer("body", body),
        _layer("unknown_1", hair),
        _layer("unknown_2", face),
        _layer("unknown_3", clothes),
    ]
    out = assign_roles(layers, rgba, load_taxonomy())
    roles = {layer.role for layer in out}
    assert "body" in roles
    assert "face" in roles
    assert "clothes" in roles
    assert "hair_back" in roles or "hair_front" in roles
    assert not any(layer.role.startswith("unknown") for layer in out)


def test_keeps_already_mapped_roles():
    rgba, body, hair, face, clothes = _canvas()
    layers = [
        _layer("body", body),
        _layer("hair_front", hair),
        _layer("unknown_2", face),
    ]
    out = assign_roles(layers, rgba, load_taxonomy())
    assert out[1].role == "hair_front"
    assert out[2].role == "face"


def test_leftover_unknown_becomes_acc():
    h, w = 32, 32
    rgb = np.full((h, w, 3), 250, dtype=np.uint8)
    body = np.zeros((h, w), dtype=np.uint8)
    body[4:28, 4:28] = 255
    rgb[4:28, 4:28] = (30, 30, 30)
    speck = np.zeros((h, w), dtype=np.uint8)
    speck[6:9, 20:24] = 255
    rgb[6:9, 20:24] = (20, 180, 40)
    rgba = np.dstack([rgb, np.full((h, w), 255, dtype=np.uint8)])
    layers = [_layer("body", body), _layer("unknown_9", speck)]
    out = assign_roles(layers, rgba, load_taxonomy())
    assert out[1].role == "acc"

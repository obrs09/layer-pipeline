from __future__ import annotations

import numpy as np

from layerforge.ops.inventory import build_inventory
from layerforge.taxonomy import load_taxonomy


def test_required_eyes_and_face_always_wanted():
    tax = load_taxonomy()
    wanted = build_inventory({"1girl": 0.99}, tax)
    assert "eye_l" in wanted
    assert "eye_r" in wanted
    assert "face" in wanted
    assert "body" in wanted
    assert "clothes" in wanted
    assert "hair_back" in wanted
    assert "hair_front" not in wanted
    assert "acc" not in wanted


def test_bangs_tag_requests_front_hair():
    tax = load_taxonomy()
    wanted = build_inventory({"1girl": 0.9, "bangs": 0.8, "long_hair": 0.7}, tax)
    assert "hair_front" in wanted
    assert "hair_back" in wanted


def test_white_hair_requests_front_hair():
    tax = load_taxonomy()
    wanted = build_inventory({"1girl": 0.9, "white_hair": 0.8}, tax)
    assert "hair_back" in wanted
    assert "hair_front" in wanted


def test_bald_skips_hair():
    tax = load_taxonomy()
    wanted = build_inventory({"1girl": 0.9, "bald": 0.9, "bangs": 0.8}, tax)
    assert "hair_front" not in wanted
    assert "hair_back" not in wanted


def test_ribbon_only_when_confident():
    tax = load_taxonomy()
    low = build_inventory({"1girl": 0.9, "ribbon": 0.2}, tax)
    high = build_inventory({"1girl": 0.9, "ribbon": 0.7}, tax)
    assert "acc" not in low
    assert "acc" in high

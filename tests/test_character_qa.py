from __future__ import annotations

import json

import numpy as np

from layerforge.backends.detect.anime_segmentation import _seed_variant
from layerforge.backends.segment.cascade import CascadeSegment
from layerforge.ops.character_qa import (
    choose_peer_character,
    connected_component_metrics,
    dice_coef,
    evaluate_character_qa,
    shannon_entropy,
)
from layerforge.ops.trace import StepDump
from layerforge.taxonomy import load_taxonomy


def _blob_mask(h=64, w=64):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[8:56, 10:54] = 255
    return mask


def test_seed_variant_cycles_threshold_and_noise():
    a = _seed_variant(0, 1024)
    b = _seed_variant(1, 1024)
    c = _seed_variant(2, 1024)
    assert a == (1024, 0.50, 0.0)
    assert b[1] != a[1] and b[2] > 0
    assert c[1] != a[1] and c[2] > 0


def test_largest_component_too_small():
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[2:22, 2:22] = 255
    mask[30:50, 30:50] = 255
    mask[58:78, 10:30] = 255
    cc = connected_component_metrics(mask, min_component_px=50)
    assert cc["n_kept"] == 3
    assert cc["largest_frac"] < 0.60
    report = evaluate_character_qa(mask, (mask > 0).astype(np.float32), {})
    assert not report.ok
    assert any(item.startswith("largest_frac=") for item in report.reasons)


def test_too_many_islands():
    mask = np.zeros((160, 160), dtype=np.uint8)
    mask[40:120, 40:120] = 255
    for i in range(16):
        y = 4 + (i // 8) * 12
        x = 4 + (i % 8) * 12
        mask[y : y + 8, x : x + 8] = 255
    report = evaluate_character_qa(mask, (mask > 0).astype(np.float32), {})
    assert report.metrics["n_kept"] > 15
    assert not report.ok
    assert any(item.startswith("components=") for item in report.reasons)


def test_gray_zone_ratio():
    mask = _blob_mask()
    prob = np.full(mask.shape, 0.05, dtype=np.float32)
    prob[mask > 0] = 0.5
    report = evaluate_character_qa(mask, prob, {})
    assert not report.ok
    assert any(item.startswith("gray_frac=") for item in report.reasons)


def test_interior_entropy():
    mask = _blob_mask()
    prob = np.full(mask.shape, 0.05, dtype=np.float32)
    prob[mask > 0] = 0.5
    entropy = shannon_entropy(prob)
    assert float(entropy[mask > 0].mean()) > 0.5
    report = evaluate_character_qa(mask, prob, {}, {"max_gray_frac": 1.0})
    assert not report.ok
    assert any(item.startswith("interior_entropy=") for item in report.reasons)


def test_dice_disagreement():
    mask = _blob_mask()
    other = np.zeros_like(mask)
    other[40:62, 40:62] = 255
    assert dice_coef(mask, other) < 0.85
    report = evaluate_character_qa(
        mask,
        (mask > 0).astype(np.float32),
        {"toonout": other},
    )
    assert not report.ok
    assert any(item.startswith("dice_toonout=") for item in report.reasons)


def test_missing_peer_is_skipped():
    mask = _blob_mask()
    report = evaluate_character_qa(
        mask,
        (mask > 0).astype(np.float32),
        {"toonout": None, "modnet": None},
    )
    assert report.ok
    assert report.metrics["dice"]["toonout"] is None


class _RetryCut:
    name = "anime_segmentation"

    def __init__(self, masks):
        self.masks = masks
        self.seeds = []

    def cut(self, image):
        return self.predict(image, seed=0)[0]

    def predict(self, image, seed=0):
        self.seeds.append(seed)
        mask = self.masks[seed]
        return mask, (mask > 0).astype(np.float32)


class _Peer:
    name = "toonout"

    def __init__(self, mask):
        self.mask = mask

    def cut(self, image):
        return self.mask


def test_retry_uses_later_seed_then_clears_flag():
    bad = np.zeros((80, 80), dtype=np.uint8)
    bad[4:24, 4:24] = 255
    bad[30:50, 30:50] = 255
    bad[56:76, 8:28] = 255
    good = _blob_mask(80, 80)
    cut = _RetryCut({0: bad, 1: good, 2: good})
    cascade = CascadeSegment({}, load_taxonomy(), character=cut, dry_run=True)
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    mask = cascade._cut_character(image)
    assert int((mask > 0).sum()) == int((good > 0).sum())
    assert cut.seeds == [0, 1]
    assert cascade.character_qa["flagged"] is False
    assert cascade.character_qa["chosen_seed"] == 1
    assert "character" not in cascade.needs_click


def test_third_attempt_still_bad_is_flagged(tmp_path):
    bad = np.zeros((80, 80), dtype=np.uint8)
    bad[4:24, 4:24] = 255
    bad[30:50, 30:50] = 255
    bad[56:76, 8:28] = 255
    cut = _RetryCut({0: bad, 1: bad, 2: bad})
    dump = StepDump(tmp_path, enabled=True)
    dump.reset()
    cascade = CascadeSegment({}, load_taxonomy(), character=cut, dry_run=True)
    cascade.bind_dump(dump)
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    cascade._cut_character(image)
    cascade._dump_character(image, cascade.character_mask)
    assert cut.seeds == [0, 1, 2]
    assert cascade.character_qa["flagged"] is True
    assert "character" in cascade.needs_click
    qa = json.loads((tmp_path / "steps" / "01_character" / "qa.json").read_text(encoding="utf-8"))
    assert qa["flagged"] is True
    assert (tmp_path / "steps" / "01_character" / "FLAGGED.json").exists()
    from PIL import Image

    overlay = np.array(Image.open(tmp_path / "steps" / "01_character" / "overlay.png"))
    assert overlay[..., :3].max() > 0


def test_peer_dice_triggers_retry():
    good = _blob_mask(80, 80)
    disagree = np.zeros_like(good)
    disagree[40:78, 40:78] = 255
    cut = _RetryCut({0: good, 1: good, 2: good})
    cascade = CascadeSegment(
        {},
        load_taxonomy(),
        character=cut,
        character_peers=[_Peer(disagree)],
        dry_run=True,
    )
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    cascade._cut_character(image)
    assert cut.seeds == [0, 1, 2]
    assert cascade.character_qa["flagged"] is True
    last = cascade.character_qa["attempts"][-1]
    assert any(item.startswith("dice_toonout=") for item in last["reasons"])


class _NamedPeer(_Peer):
    def __init__(self, name, mask):
        self.name = name
        self.mask = mask


def test_choose_peer_and_when_isnet_includes_furniture():
    body = _blob_mask(80, 80)
    isnet = body.copy()
    isnet[50:78, 48:78] = 255
    toon = body.copy()
    mod = body.copy()
    mod[9:12, 12:16] = 0
    choice = choose_peer_character(isnet, {"toonout": toon, "modnet": mod}, {})
    assert choice is not None
    assert choice["source"] == "peer_and"
    assert int(choice["mask"][60, 60]) == 0
    assert int(choice["mask"][30, 30]) == 255


def test_choose_peer_keeps_isnet_when_all_agree():
    body = _blob_mask(80, 80)
    choice = choose_peer_character(body, {"toonout": body, "modnet": body}, {})
    assert choice is None


def test_choose_peer_ignores_tiny_disagreement():
    body = _blob_mask(80, 80)
    junk = np.zeros_like(body)
    junk[40:78, 40:78] = 255
    choice = choose_peer_character(body, {"toonout": junk}, {})
    assert choice is None


def test_cascade_replaces_isnet_with_peer_and():
    body = _blob_mask(80, 80)
    isnet = body.copy()
    isnet[50:78, 48:78] = 255
    cut = _RetryCut({0: isnet, 1: isnet, 2: isnet})
    cascade = CascadeSegment(
        {},
        load_taxonomy(),
        character=cut,
        character_peers=[_NamedPeer("toonout", body), _NamedPeer("modnet", body)],
        dry_run=True,
    )
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    mask = cascade._cut_character(image)
    assert cascade.character_qa["chosen_source"] == "peer_and"
    assert cascade.character_qa["flagged"] is False
    assert "character" not in cascade.needs_click
    assert int(mask[60, 60]) == 0
    assert int(mask[30, 30]) == 255


def test_cut_character_records_qa_without_flag():
    cascade = CascadeSegment(
        {},
        load_taxonomy(),
        character=_RetryCut({0: _blob_mask(80, 80)}),
        dry_run=True,
    )
    cascade._cut_character(np.zeros((80, 80, 3), dtype=np.uint8))
    assert cascade.character_qa is not None
    assert cascade.character_qa["flagged"] is False

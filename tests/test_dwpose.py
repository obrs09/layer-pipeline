from __future__ import annotations

import numpy as np

from layerforge.backends.detect.dwpose import (
    DwPoseEstimate,
    clip_character_to_person,
    coco17_to_openpose,
    person_mask_from_pose,
)


def _coco_standing(h=80, w=60):
    kpts = np.zeros((17, 2), dtype=np.float32)
    scores = np.ones(17, dtype=np.float32)
    kpts[0] = (30, 10)  # nose
    kpts[1] = (26, 8)  # leye
    kpts[2] = (34, 8)  # reye
    kpts[3] = (24, 10)  # lear
    kpts[4] = (36, 10)  # rear
    kpts[5] = (22, 22)  # lsho
    kpts[6] = (38, 22)  # rsho
    kpts[7] = (18, 34)  # lelb
    kpts[8] = (42, 34)  # relb
    kpts[9] = (16, 44)  # lwri
    kpts[10] = (44, 44)  # rwri
    kpts[11] = (24, 46)  # lhip
    kpts[12] = (36, 46)  # rhip
    kpts[13] = (24, 62)  # lknee
    kpts[14] = (36, 62)  # rknee
    kpts[15] = (24, 74)  # lank
    kpts[16] = (36, 74)  # rank
    return kpts, scores


def test_openpose_adds_neck():
    kpts, scores = _coco_standing()
    pose_k, pose_s = coco17_to_openpose(kpts, scores)
    assert pose_k.shape == (18, 2)
    assert pose_k[1, 0] == 30
    assert pose_s[1] == 1.0


def test_person_mask_covers_torso_not_corners():
    kpts, scores = _coco_standing()
    pose_k, pose_s = coco17_to_openpose(kpts, scores)
    mask = person_mask_from_pose(pose_k, pose_s, (80, 60), min_score=0.3, dilate_px=4)
    assert int(mask[30, 30]) == 255
    assert int(mask[2, 2]) == 0


def test_spread_limbs_do_not_fill_bed():
    kpts, scores = _coco_standing()
    pose_k, pose_s = coco17_to_openpose(kpts, scores)
    pose_k[4] = (58, 8)
    pose_k[7] = (2, 8)
    mask = person_mask_from_pose(pose_k, pose_s, (80, 60), min_score=0.3, dilate_px=2)
    assert int(mask[40, 4]) == 0
    assert int(mask[34, 30]) == 255


def test_clip_character_drops_furniture():
    character = np.full((40, 40), 255, dtype=np.uint8)
    person = np.zeros((40, 40), dtype=np.uint8)
    person[8:32, 10:30] = 255
    clipped = clip_character_to_person(character, person)
    assert int(clipped[0, 0]) == 0
    assert int(clipped[20, 20]) == 255


def test_clip_keeps_isnet_if_pose_too_small():
    character = np.full((40, 40), 255, dtype=np.uint8)
    person = np.zeros((40, 40), dtype=np.uint8)
    person[18:22, 18:22] = 255
    clipped = clip_character_to_person(character, person, min_keep_frac=0.12)
    assert int((clipped > 0).sum()) == int((character > 0).sum())


def test_missing_weights_returns_none():
    est = DwPoseEstimate({"paths": {"dwpose_dir": "model/missing-dwpose"}})
    image = np.zeros((32, 24, 3), dtype=np.uint8)
    assert est.estimate(image) is None

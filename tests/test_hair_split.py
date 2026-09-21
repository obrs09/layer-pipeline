from __future__ import annotations

import numpy as np

from layerforge.ops.hair_split import (
    bangs_region,
    sample_points,
    split_depth,
    split_occlusion,
    split_parsing,
)


def _face(h=40, w=40, box=(12, 14, 28, 32)):
    face = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = box
    face[y0:y1, x0:x1] = 255
    return face


def test_bangs_region_uses_forehead_not_chin():
    face = _face()
    region = bangs_region(face, forehead_frac=0.4, bangs_up_frac=0.0)
    assert int(region[16, 20]) == 255
    assert int(region[30, 20]) == 0


def test_occlusion_front_covers_face_back_is_leftover():
    hair = np.zeros((40, 40), dtype=np.uint8)
    hair[4:36, 8:32] = 255
    face = _face()
    clothes = np.zeros((40, 40), dtype=np.uint8)
    clothes[28:40, :] = 255
    result = split_occlusion(hair, face, clothes=clothes, grow_px=6, min_front_px=8)
    assert result.strategy == "occlusion"
    assert int(result.front[16, 20]) == 255
    assert int(result.front[34, 20]) == 0
    assert int(result.back[34, 20]) == 255
    assert int((result.front > 0).sum()) + int((result.back > 0).sum()) == int((hair > 0).sum())
    # wrap-around under the chin/clothes stays back
    assert int(result.front[30, 20]) == 0


def test_occlusion_does_not_use_face_box_as_front():
    hair = np.zeros((40, 40), dtype=np.uint8)
    hair[2:10, 8:32] = 255
    hair[10:36, 4:8] = 255
    face = np.ones((40, 40), dtype=np.uint8) * 255
    result = split_occlusion(hair, face, grow_px=4, forehead_frac=0.2, min_front_px=8)
    # side lock below the forehead band must not all become front just because face is a filled frame
    assert int(result.back[30, 6]) == 255


def test_depth_closer_than_face_is_front():
    hair = np.zeros((20, 20), dtype=np.uint8)
    hair[2:18, 2:18] = 255
    face = np.zeros((20, 20), dtype=np.uint8)
    face[8:14, 8:14] = 255
    depth = np.full((20, 20), 10.0, dtype=np.float32)
    depth[2:8, 2:18] = 4.0  # closer bangs
    depth[14:18, 2:18] = 16.0  # farther back
    result = split_depth(hair, face, depth, eps=0.05, min_front_px=4)
    assert int(result.front[4, 10]) == 255
    assert int(result.back[16, 10]) == 255


def test_parsing_uses_eroded_core():
    hair = np.ones((30, 30), dtype=np.uint8) * 255
    coarse = np.zeros((30, 30), dtype=np.uint8)
    coarse[8:22, 8:22] = 255
    back = np.zeros((30, 30), dtype=np.uint8)
    back[0:6, :] = 255
    result = split_parsing(hair, coarse, back, erode_px=3, min_front_px=4)
    assert int(result.front[15, 15]) == 255
    assert int(result.front[8, 8]) == 0
    assert sample_points(result.front, 3)


def test_landmark_bangs_are_above_brows():
    face = _face()
    kpts = [{"name": f"face_{i:02d}", "x": 14.0 + (i - 17) * 1.2, "y": 18.0, "score": 0.9} for i in range(17, 27)]
    region = bangs_region(face, kpts, bangs_up_frac=0.4)
    assert int(region[12, 20]) == 255
    assert int(region[28, 20]) == 0

from __future__ import annotations

import numpy as np

from layerforge.ops.figure_parts import split_figure


def test_split_keeps_clothes_intersection_and_skin_and_remainder():
    body = np.zeros((40, 40), dtype=np.uint8)
    body[4:36, 8:32] = 255
    arm = np.zeros_like(body)
    arm[8:28, 8:14] = 255
    torso = np.zeros_like(body)
    torso[8:28, 14:28] = 255
    clothes = np.zeros_like(body)
    clothes[8:20, 8:28] = 255
    chain = np.zeros_like(body)
    chain[30:34, 10:20] = 255
    split = split_figure(body, {"arm_l": arm, "torso": torso}, clothes)
    assert int((split.clothes_parts["arm_l"] > 0).sum()) == int(((arm > 0) & (clothes > 0)).sum())
    assert int((split.skins["torso"] > 0).sum()) == int(((torso > 0) & (clothes == 0)).sum())
    assert int(((chain > 0) & (split.remainder > 0)).sum()) == int((chain > 0).sum())
    assert int(((split.skins["torso"] > 0) & (split.clothes > 0)).sum()) == 0
    assert int(((split.skins["arm_l"] > 0) & (split.clothes_parts["arm_l"] > 0)).sum()) == 0

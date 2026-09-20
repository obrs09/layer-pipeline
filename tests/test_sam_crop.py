from layerforge.ops.sam_crop import (
    box_to_crop,
    crop_origin,
    expand_box,
    points_to_crop,
    to_crop_xy,
    upscale_hw,
)


def test_expand_box_pads_and_clips():
    out = expand_box([10.0, 10.0, 20.0, 20.0], 0.2, 40, 40)
    assert out[0] == 8.0
    assert out[1] == 8.0
    assert out[2] == 22.0
    assert out[3] == 22.0
    clipped = expand_box([0.0, 0.0, 10.0, 10.0], 0.5, 12, 12)
    assert clipped[0] == 0.0
    assert clipped[1] == 0.0
    assert clipped[2] == 12.0


def test_upscale_hw_never_shrinks():
    h, w, scale = upscale_hw(200, 400, 1024)
    assert scale == 1024 / 200
    assert h == 1024
    assert w == 2048
    same_h, same_w, same_scale = upscale_hw(1200, 1100, 1024)
    assert same_scale == 1.0
    assert (same_h, same_w) == (1200, 1100)


def test_box_and_points_remap_to_crop():
    origin = crop_origin([20.0, 10.0, 40.0, 30.0], 0.0, 80, 60)
    assert origin == (20, 10, 40, 30)
    box = box_to_crop([20.0, 10.0, 40.0, 30.0], (20, 10), 2.0)
    assert box == [0.0, 0.0, 40.0, 40.0]
    pts = points_to_crop([(22.0, 12.0), (1.0, 1.0)], (20, 10), 2.0, 40, 40)
    assert pts == [to_crop_xy(22.0, 12.0, (20, 10), 2.0)]
    assert pts[0] == (4.0, 4.0)

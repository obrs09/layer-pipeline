from __future__ import annotations

from layerforge.backends.detect.grounding_dino import post_process_boxes


class _ProcV5:
    def post_process_grounded_object_detection(
        self, outputs, input_ids=None, threshold=0.25, text_threshold=0.25, target_sizes=None
    ):
        return [{"kwargs": {"threshold": threshold, "text_threshold": text_threshold}}]


class _ProcV4:
    def post_process_grounded_object_detection(
        self, outputs, input_ids=None, box_threshold=0.25, text_threshold=0.25, target_sizes=None
    ):
        return [{"kwargs": {"box_threshold": box_threshold, "text_threshold": text_threshold}}]


def test_post_process_uses_threshold_not_box_threshold():
    out = post_process_boxes(_ProcV5(), outputs=None, input_ids=None, threshold=0.4, target_sizes=[(8, 8)])
    assert "box_threshold" not in out["kwargs"]
    assert out["kwargs"]["threshold"] == 0.4


def test_post_process_legacy_box_threshold():
    out = post_process_boxes(_ProcV4(), outputs=None, input_ids=None, threshold=0.3, target_sizes=[(8, 8)])
    assert out["kwargs"]["box_threshold"] == 0.3
    assert "threshold" not in out["kwargs"]

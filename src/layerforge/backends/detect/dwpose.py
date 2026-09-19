"""DWPose (YOLOX + RTMPose wholebody) as a PoseEstimator.

Hints only: clip the character cut to the person, and feed part boxes/points
to SAM. Inventory still comes from WDTagger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from layerforge.config import resolve_path
from layerforge.ops.morph import dilate_mask
from layerforge.ops.normalize import to_rgba

# OpenPose-18 after COCO-17 conversion.
OPENPOSE_NAMES = [
    "nose",
    "neck",
    "rsho",
    "relb",
    "rwri",
    "lsho",
    "lelb",
    "lwri",
    "rhip",
    "rknee",
    "rank",
    "lhip",
    "lknee",
    "lank",
    "reye",
    "leye",
    "rear",
    "lear",
]
OPENPOSE_LIMBS = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
    (2, 5),
    (8, 11),
]
TORSO_IDX = (1, 2, 5, 8, 11)
HEAD_IDX = (0, 1, 14, 15, 16, 17)
ROLE_KPTS = {
    "face": (0, 1, 14, 15, 16, 17),
    "body": (1, 2, 5, 8, 9, 10, 11, 12, 13),
    "arm_r": (2, 3, 4),
    "arm_l": (5, 6, 7),
}


@dataclass
class PoseEstimate:
    person_mask: np.ndarray
    keypoints: list[dict]
    boxes: dict[str, list[float]]
    points: dict[str, list[tuple[float, float]]]
    overlay: np.ndarray
    person_box: list[float] = field(default_factory=list)
    notes: str = ""


def coco17_to_openpose(kpts: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """COCO-17 (or wholebody-133 prefix) → OpenPose-18 with a derived neck."""
    body = np.asarray(kpts[:17], dtype=np.float32)
    sc = np.asarray(scores[:17], dtype=np.float32)
    neck = (body[5] + body[6]) * 0.5
    neck_s = float(min(sc[5], sc[6]))
    order = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    out_k = np.zeros((18, 2), dtype=np.float32)
    out_s = np.zeros(18, dtype=np.float32)
    for i, src in enumerate(order):
        if src is None:
            out_k[i] = neck
            out_s[i] = neck_s
        else:
            out_k[i] = body[src]
            out_s[i] = sc[src]
    return out_k, out_s


def person_mask_from_pose(
    kpts: np.ndarray,
    scores: np.ndarray,
    shape: tuple[int, int],
    *,
    min_score: float = 0.3,
    dilate_px: int = 72,
) -> np.ndarray:
    """Thick skeleton + torso hull. Do not hull all limbs — that fills beds between spread arms."""
    h, w = shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    pts = np.asarray(kpts, dtype=np.float32)
    sc = np.asarray(scores, dtype=np.float32)
    thick = max(10, int(0.045 * max(h, w)))
    for a, b in OPENPOSE_LIMBS:
        if a >= len(pts) or b >= len(pts):
            continue
        if sc[a] < min_score or sc[b] < min_score:
            continue
        pa = (int(round(pts[a, 0])), int(round(pts[a, 1])))
        pb = (int(round(pts[b, 0])), int(round(pts[b, 1])))
        cv2.line(canvas, pa, pb, 255, thickness=thick, lineType=cv2.LINE_AA)
    for (x, y), score in zip(pts, sc):
        if score < min_score:
            continue
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < w and 0 <= iy < h:
            cv2.circle(canvas, (ix, iy), max(8, thick // 2), 255, -1)
    torso = []
    for i in TORSO_IDX:
        if i >= len(pts) or sc[i] < min_score:
            continue
        torso.append([int(round(pts[i, 0])), int(round(pts[i, 1]))])
    if len(torso) >= 3:
        hull = cv2.convexHull(np.array(torso, dtype=np.int32))
        cv2.fillConvexPoly(canvas, hull, 255)
    head_r = max(int(dilate_px * 1.8), int(0.11 * max(h, w)))
    for i in HEAD_IDX:
        if i >= len(pts) or sc[i] < min_score:
            continue
        cv2.circle(
            canvas,
            (int(round(pts[i, 0])), int(round(pts[i, 1]))),
            head_r,
            255,
            -1,
        )
    return dilate_mask(canvas, dilate_px)


def part_boxes_from_pose(
    kpts: np.ndarray,
    scores: np.ndarray,
    shape: tuple[int, int],
    *,
    min_score: float = 0.25,
    pad_frac: float = 0.28,
) -> dict[str, list[float]]:
    h, w = shape
    boxes: dict[str, list[float]] = {}
    for role, idxs in ROLE_KPTS.items():
        pts = []
        for i in idxs:
            if i >= len(kpts) or scores[i] < min_score:
                continue
            pts.append(kpts[i])
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        bw, bh = max(8.0, x1 - x0), max(8.0, y1 - y0)
        pad_x, pad_y = bw * pad_frac, bh * pad_frac
        boxes[role] = [
            float(max(0.0, x0 - pad_x)),
            float(max(0.0, y0 - pad_y)),
            float(min(w - 1.0, x1 + pad_x)),
            float(min(h - 1.0, y1 + pad_y)),
        ]
    return boxes


def part_points_from_pose(
    kpts: np.ndarray,
    scores: np.ndarray,
    *,
    min_score: float = 0.35,
) -> dict[str, list[tuple[float, float]]]:
    points: dict[str, list[tuple[float, float]]] = {}
    for role, idxs in ROLE_KPTS.items():
        pts = []
        for i in idxs:
            if i >= len(kpts) or scores[i] < min_score:
                continue
            pts.append((float(kpts[i, 0]), float(kpts[i, 1])))
        if pts:
            points[role] = pts
    return points


def draw_pose_overlay(
    image: np.ndarray,
    kpts: np.ndarray,
    scores: np.ndarray,
    *,
    min_score: float = 0.3,
) -> np.ndarray:
    rgb = to_rgba(image)[:, :, :3].copy()
    colors = [
        (255, 0, 0),
        (255, 85, 0),
        (255, 170, 0),
        (255, 255, 0),
        (170, 255, 0),
        (85, 255, 0),
        (0, 255, 0),
        (0, 255, 85),
        (0, 255, 170),
        (0, 255, 255),
        (0, 170, 255),
        (0, 85, 255),
        (0, 0, 255),
        (85, 0, 255),
        (170, 0, 255),
        (255, 0, 255),
        (255, 0, 170),
        (255, 0, 85),
    ]
    thick = max(2, int(0.004 * max(rgb.shape[0], rgb.shape[1])))
    for i, (a, b) in enumerate(OPENPOSE_LIMBS):
        if a >= len(kpts) or b >= len(kpts):
            continue
        if scores[a] < min_score or scores[b] < min_score:
            continue
        pa = (int(round(kpts[a, 0])), int(round(kpts[a, 1])))
        pb = (int(round(kpts[b, 0])), int(round(kpts[b, 1])))
        cv2.line(rgb, pa, pb, colors[i % len(colors)], thick, cv2.LINE_AA)
    for i, ((x, y), score) in enumerate(zip(kpts, scores)):
        if score < min_score:
            continue
        cv2.circle(rgb, (int(round(x)), int(round(y))), thick + 2, colors[i % len(colors)], -1)
    return rgb


def clip_character_to_person(
    character: np.ndarray,
    person: np.ndarray,
    *,
    min_keep_frac: float = 0.12,
) -> np.ndarray:
    """Keep isnet if the pose person is empty or clips away almost everything."""
    if person is None or int((person > 0).sum()) < 64:
        return character
    clipped = ((character > 0) & (person > 0)).astype(np.uint8) * 255
    orig = max(1, int((character > 0).sum()))
    if int((clipped > 0).sum()) / orig < min_keep_frac:
        return character
    return clipped


def build_pose_estimate(
    image: np.ndarray,
    kpts: np.ndarray,
    scores: np.ndarray,
    person_box: list[float],
    *,
    min_score: float = 0.3,
    dilate_px: int = 72,
) -> PoseEstimate:
    h, w = image.shape[:2]
    pose_k, pose_s = coco17_to_openpose(kpts, scores)
    person = person_mask_from_pose(pose_k, pose_s, (h, w), min_score=min_score, dilate_px=dilate_px)
    overlay = draw_pose_overlay(image, pose_k, pose_s, min_score=min_score)
    named = [
        {
            "name": OPENPOSE_NAMES[i],
            "x": round(float(pose_k[i, 0]), 1),
            "y": round(float(pose_k[i, 1]), 1),
            "score": round(float(pose_s[i]), 4),
        }
        for i in range(len(pose_k))
    ]
    return PoseEstimate(
        person_mask=person,
        keypoints=named,
        boxes=part_boxes_from_pose(pose_k, pose_s, (h, w), min_score=min_score * 0.8),
        points=part_points_from_pose(pose_k, pose_s, min_score=min_score),
        overlay=overlay,
        person_box=[float(v) for v in person_box],
    )


class DwPoseEstimate:
    name = "dwpose"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        cascade = cfg.get("cascade") or {}
        self.min_score = float(cascade.get("pose_score", 0.3))
        self.dilate_px = int(cascade.get("pose_dilate_px", 72))
        self._det = None
        self._pose = None

    def estimate(self, image: np.ndarray) -> PoseEstimate | None:
        det_path, pose_path = self._onnx_paths()
        if det_path is None or pose_path is None:
            return None
        rgb = to_rgba(image)[:, :, :3]
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        boxes = self._detect(det_path, bgr)
        h, w = bgr.shape[:2]
        if boxes is None or len(boxes) == 0:
            boxes = np.array([[0.0, 0.0, float(w), float(h)]], dtype=np.float32)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(np.argmax(areas))
        person_box = [float(v) for v in boxes[idx]]
        kpts, scores = self._pose_run(pose_path, boxes[idx : idx + 1], bgr)
        if kpts is None or len(kpts) == 0:
            return None
        return build_pose_estimate(
            rgb,
            kpts[0],
            scores[0],
            person_box,
            min_score=self.min_score,
            dilate_px=self.dilate_px,
        )

    def _onnx_paths(self) -> tuple[Path | None, Path | None]:
        paths = self.cfg.get("paths") or {}
        det = self._one_onnx(
            paths.get("dwpose_det"),
            paths.get("dwpose_dir", "model/dwpose"),
            ("yolox_l.onnx", "yolox.onnx"),
        )
        pose = self._one_onnx(
            paths.get("dwpose_pose"),
            paths.get("dwpose_dir", "model/dwpose"),
            ("dw-ll_ucoco_384.onnx", "dwpose.onnx"),
        )
        return det, pose

    def _one_onnx(self, explicit, directory, names: tuple[str, ...]) -> Path | None:
        if explicit:
            path = resolve_path(explicit)
            if path.exists():
                return path
        search = resolve_path(directory)
        if not search.exists():
            return None
        for name in names:
            cand = search / name
            if cand.exists():
                return cand
        found = sorted(search.glob("*.onnx"))
        return found[0] if found else None

    def _session(self, path: Path, cache_attr: str):
        cached = getattr(self, cache_attr)
        if cached is not None:
            return cached
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        setattr(self, cache_attr, sess)
        return sess

    def _detect(self, path: Path, bgr: np.ndarray) -> np.ndarray:
        from layerforge.backends.detect._dwpose_rt import inference_detector

        return inference_detector(self._session(path, "_det"), bgr)

    def _pose_run(self, path: Path, boxes: np.ndarray, bgr: np.ndarray):
        from layerforge.backends.detect._dwpose_rt import inference_pose

        return inference_pose(self._session(path, "_pose"), boxes, bgr)

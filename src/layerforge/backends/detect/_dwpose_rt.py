"""YOLOX person + RTMPose wholebody. Adapted from IDEA-Research DWPose ONNX."""

from __future__ import annotations

import cv2
import numpy as np


def nms(boxes, scores, nms_thr):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= nms_thr)[0] + 1]
    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    final_dets = []
    for cls_ind in range(scores.shape[1]):
        cls_scores = scores[:, cls_ind]
        valid = cls_scores > score_thr
        if valid.sum() == 0:
            continue
        keep = nms(boxes[valid], cls_scores[valid], nms_thr)
        if not keep:
            continue
        cls_inds = np.ones((len(keep), 1)) * cls_ind
        dets = np.concatenate(
            [boxes[valid][keep], cls_scores[valid][keep, None], cls_inds],
            1,
        )
        final_dets.append(dets)
    if not final_dets:
        return None
    return np.concatenate(final_dets, 0)


def demo_postprocess(outputs, img_size, p6=False):
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]
    grids = []
    expanded_strides = []
    for stride in strides:
        hsize = img_size[0] // stride
        wsize = img_size[1] // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((*grid.shape[:2], 1), stride))
    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs


def _det_preprocess(img, input_size=(640, 640), swap=(2, 0, 1)):
    padded = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    ratio = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized = cv2.resize(
        img,
        (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    blob = np.ascontiguousarray(padded.transpose(swap), dtype=np.float32)
    return blob, ratio


def inference_detector(session, bgr: np.ndarray, score_thr: float = 0.1) -> np.ndarray:
    input_shape = (640, 640)
    blob, ratio = _det_preprocess(bgr, input_shape)
    output = session.run(None, {session.get_inputs()[0].name: blob[None, :, :, :]})
    predictions = demo_postprocess(output[0], input_shape)[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is None:
        return np.zeros((0, 4), dtype=np.float32)
    final_boxes, final_scores, final_cls = dets[:, :4], dets[:, 4], dets[:, 5]
    keep = (final_scores > score_thr) & (final_cls == 0)
    return final_boxes[keep]


def bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.0):
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding
    if dim == 1:
        return center[0], scale[0]
    return center, scale


def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    w, h = np.hsplit(bbox_scale, [1])
    return np.where(
        w > h * aspect_ratio,
        np.hstack([w, w / aspect_ratio]),
        np.hstack([h * aspect_ratio, h]),
    )


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    return np.array([[cs, -sn], [sn, cs]]) @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def get_warp_matrix(center, scale, rot, output_size, shift=(0.0, 0.0), inv=False):
    shift = np.array(shift)
    src_w = scale[0]
    dst_w, dst_h = output_size
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0.0, src_w * -0.5]), rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5])
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def top_down_affine(input_size, bbox_scale, bbox_center, img):
    w, h = input_size
    bbox_scale = _fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)
    warp_mat = get_warp_matrix(bbox_center, bbox_scale, 0, output_size=(w, h))
    warped = cv2.warpAffine(img, warp_mat, (int(w), int(h)), flags=cv2.INTER_LINEAR)
    return warped, bbox_scale


def _pose_preprocess(img, out_bbox, input_size):
    img_shape = img.shape[:2]
    out_img, out_center, out_scale = [], [], []
    if len(out_bbox) == 0:
        out_bbox = [[0, 0, img_shape[1], img_shape[0]]]
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    for box in out_bbox:
        bbox = np.array([box[0], box[1], box[2], box[3]])
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)
        resized, scale = top_down_affine(input_size, scale, center, img)
        out_img.append((resized - mean) / std)
        out_center.append(center)
        out_scale.append(scale)
    return out_img, out_center, out_scale


def get_simcc_maximum(simcc_x, simcc_y):
    n, k, _ = simcc_x.shape
    simcc_x = simcc_x.reshape(n * k, -1)
    simcc_y = simcc_y.reshape(n * k, -1)
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    locs[max_val_x <= 0.0] = -1
    return locs.reshape(n, k, 2), max_val_x.reshape(n, k)


def decode(simcc_x, simcc_y, simcc_split_ratio):
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints /= simcc_split_ratio
    return keypoints, scores


def inference_pose(session, out_bbox, bgr: np.ndarray):
    h, w = session.get_inputs()[0].shape[2:]
    model_input_size = (w, h)
    resized, center, scale = _pose_preprocess(bgr, out_bbox, model_input_size)
    all_out = []
    for item in resized:
        blob = [item.transpose(2, 0, 1)]
        names = [out.name for out in session.get_outputs()]
        all_out.append(session.run(names, {session.get_inputs()[0].name: blob}))
    all_key = []
    all_score = []
    for i, outputs in enumerate(all_out):
        simcc_x, simcc_y = outputs
        keypoints, scores = decode(simcc_x, simcc_y, 2.0)
        keypoints = keypoints / model_input_size * scale[i] + center[i] - scale[i] / 2
        all_key.append(keypoints[0])
        all_score.append(scores[0])
    return np.array(all_key), np.array(all_score)

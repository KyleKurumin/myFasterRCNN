from typing import Tuple

import torch
from torch import Tensor


def nms(boxes: Tensor,
        scores: Tensor,
        iou_threshold: float) -> Tensor:
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes: Tensor,
                scores: Tensor,
                idxs: Tensor,
                iou_threshold: float) -> Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    max_coordinate = boxes.max()

    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def remove_small_boxes(boxes: Tensor,
                       min_size: float) -> Tensor:
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]  # 预测boxes的宽和高

    keep = torch.logical_and(torch.ge(ws, min_size), torch.ge(hs, min_size))

    keep = torch.where(keep)[0]
    return keep


def clip_boxes_to_image(boxes: Tensor,
                        size: Tuple[int, int]) -> Tensor:
    """
    裁剪预测的boxes信息，将越界的坐标调整到图片边界上
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]  # x1, x2
    boxes_y = boxes[..., 1::2]  # y1, y2
    height, width = size

    boxes_x = boxes_x.clamp(min=0, max=width)  # 限制x坐标范围在[0,width]之间
    boxes_y = boxes_y.clamp(min=0, max=height)  # 限制y坐标范围在[0,height]之间

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def bbox_area(box: torch.Tensor):
    # box -> [n, 4]
    # return -> [n]
    return (box[:, 3] - box[:, 1]) * (box[:, 2] - box[:, 0])


def box_iou(box1: torch.Tensor, box2: torch.Tensor):
    # area1 -> [n]
    # area2 -> [m]
    area1 = bbox_area(box1)
    area2 = bbox_area(box2)

    lt = torch.max(box1[:, None, :2], box2[None, :, :2])
    rb = torch.min(box1[:, None, 2:], box2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    iou = intersection / (area1[:, None] + area2 - intersection)

    return iou

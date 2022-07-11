from typing import OrderedDict, Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.det_utils import BoxCoder, Matcher, BalancedPositiveNegativeSampler, smooth_l1_loss
import boxes as box_op
from networks.image_list import ImageList


def permute_and_flatten(layer: torch.Tensor,
                        batch: int,
                        num_anchors: int,
                        num_channels: int,
                        height: int,
                        width: int) -> torch.Tensor:
    # shape[B, AxC, H, W] -> [B, -1, C]
    layer: torch.Tensor = layer.view(batch, num_anchors, num_channels, height, width)
    layer: torch.Tensor = torch.permute(layer, (0, 3, 4, 1, 2))
    layer: torch.Tensor = layer.reshape(batch, -1, num_channels)

    return layer


def concat_box_prediction_layers(box_cls: List[torch.Tensor],
                                 pred_bbox_deltas: List[torch.Tensor]) -> (torch.Tensor, torch.Tensor):
    box_cls_flattened = []
    box_regression_flattened = []

    for box_cls_per_level, box_reg_per_level in zip(box_cls, pred_bbox_deltas):
        batch, AxC, height, width = box_cls_per_level
        num_anchors = pred_bbox_deltas.shape[1] // 4
        num_classes = AxC // num_anchors

        box_cls_per_level = permute_and_flatten(box_cls_per_level, batch, num_anchors, num_classes, height, width)
        box_cls_flattened.append(box_cls_per_level)

        box_reg_per_level = permute_and_flatten(box_reg_per_level, batch, num_anchors, 4, height, width)
        box_regression_flattened.append(box_reg_per_level)

    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_reg = torch.cat(box_regression_flattened, dim=1).flatten(0, -2)

    return box_cls, box_reg


class RPNHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int) -> None:
        super(RPNHead, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=(1, 1), stride=(1, 1))
        self.bbox_preds = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=(1, 1), stride=(1, 1))

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x: List[torch.Tensor]) -> (List[torch.Tensor], List[torch.Tensor]):
        logits = []
        bbox_reg = []

        for i, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_preds(t))

        return logits, bbox_reg


class AnchorGenerator(nn.Module):
    def __init__(self, sizes: Tuple[Tuple[int]], aspect_ratios: Tuple[Tuple[float]]):
        super(AnchorGenerator, self).__init__()

        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)

        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = tuple((ratio,) for ratio in aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def cached_grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        key = str(grid_sizes) + '#' + str(strides)
        if key in self._cache:
            return self._cache[key]

        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return self._cache[key]

    def grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        anchors = []
        cell_anchors = self.cell_anchors

        for size, stride, base_anchor in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride

            device = base_anchor.device

            shifts_x = torch.arange(grid_width, dtype=torch.float32, device=device) * stride_width
            shifts_y = torch.arange(grid_height, dtype=torch.float32, device=device) * stride_height

            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

            shifts_anchor = (shifts[:, None, :] + base_anchor[None, :, :]).reshape(-1, 4)
            anchors.append(shifts_anchor)

        return anchors

    def generate_anchors(self, scales: Tuple[Tuple[int]],
                         aspect_ratios: Tuple[Tuple[float]],
                         dtype: torch.dtype = torch.float32,
                         device: torch.device = torch.device('cpu')) -> torch.Tensor:
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)

        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        hs = (scales[:, None] * w_ratios[None, :]).view(-1)
        ws = (scales[:, None] * h_ratios[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

        return base_anchors

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device) -> None:
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            if cell_anchors[0].device == device:
                return

        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype, device) for sizes, aspect_ratios in
            zip(self.sizes, self.aspect_ratios)
        ]

        self.cell_anchors = cell_anchors

    def forward(self, image_list: ImageList, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        grid_sizes = [feature_map[-2:] for feature_map in feature_maps]
        image_size = image_list.tensors.shape[-2:]

        dtype, device = feature_maps[0].device

        strides = [
            [torch.tensor(image_size[0] // grid[0], dtype=torch.int64, device=device),
             torch.tensor(image_size[1] // grid[1], dtype=torch.int64, device=device)]
            for grid in grid_sizes]

        self.set_cell_anchors(dtype, device)

        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)

        anchors = []
        for i in range(len(image_list.image_sizes)):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)

        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        self._cache.clear()

        return anchors


class RegionProposalNetwork(nn.Module):
    def __init__(self, anchor_generator: AnchorGenerator,
                 head: RPNHead,
                 fg_iou_thresh: float,
                 bg_iou_thresh: float,
                 batch_size_per_image: float,
                 positive_fraction: float,
                 pre_nms_top_n: Dict[str, int],
                 post_nms_top_n: Dict[str, int],
                 nms_thresh: float):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = BoxCoder(weights=(1., 1., 1., 1.))

        self.similarity = box_op.box_iou

        self.proposal_matcher = Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True
        )

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1e-3

    @property
    def pre_nms_top_n(self) -> int:
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self) -> int:
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def assign_targets_to_anchors(self, anchors: List[torch.Tensor],
                                  targets: List[Dict[str, torch.Tensor]]) -> \
            Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        计算每个anchors最匹配的gt，并划分为正样本，背景以及废弃的样本
        """
        labels = []
        matched_gt_boxes = []

        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:

                match_quality_matrix = box_op.box_iou(gt_boxes, anchors_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)

                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_per_image[bg_indices] = 0.0

                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objects: torch.Tensor, num_anchors_per_level: List[int]) -> torch.Tensor:

        r = []
        offset = 0

        for ob in objects.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)

            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals: torch.Tensor,
                         objects: torch.Tensor,
                         image_shapes: List[Tuple[int, int]],
                         num_anchors_per_level: List[int]) -> (List[torch.Tensor], List[torch.Tensor]):

        num_images = proposals.shape[0]
        device = proposals.device

        objects = objects.detach()
        objects = objects.reshape(num_images, -1)

        levels = [torch.full((n,), idx, dtype=torch.int64, device=device)
                  for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, 0)

        levels = levels.reshape(1, -1).expand_as(objects)

        top_n_idx = self._get_top_n_idx(objects, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]  # [batch_size, 1]

        objects = objects[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        objects_prob = torch.sigmoid(objects)

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, objects_prob, levels, image_shapes):
            boxes = box_op.clip_boxes_to_image(boxes, img_shape)

            keep = box_op.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            keep = torch.where(torch.ge(scores, self.score_thresh))[0]  # ge: >=
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            keep = box_op.batched_nms(boxes, scores, lvl, self.nms_thresh)

            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(self, objects: torch.Tensor,
                     pred_bbox_deltas: torch.Tensor,
                     labels: List[torch.Tensor],
                     regression_targets: List[torch.Tensor]) -> (torch.Tensor, torch.Tensor):

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objects = objects.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            objects[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

    def forward(self, images: ImageList,
                features: OrderedDict[str, torch.Tensor],
                targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        features = list(features.values())

        objects, pred_bbox_deltas = self.head(features)

        anchors = self.anchor_generator(images, features)
        num_images = len(anchors)

        num_anchors_per_level_shape_tensors = [o[0].shape for o in objects]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        objects, pred_bbox_deltas = concat_box_prediction_layers(objects, pred_bbox_deltas)

        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)

        boxes, scores = self.filter_proposals(proposals, objects, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objects, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg
            }
        return boxes, losses

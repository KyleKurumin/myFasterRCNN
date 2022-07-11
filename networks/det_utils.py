import math
from typing import Tuple, List, Dict

import torch
from torch import nn, Tensor


def encode_boxes(reference_boxes: Tensor, proposals: Tensor, weights: Tuple[float, float, float, float]) -> Tensor:
    weight_x, weight_y, weight_width, weight_height = weights

    proposal_x1: Tensor = proposals[:, 0].unsqueeze(1)
    proposal_y1: Tensor = proposals[:, 1].unsqueeze(1)
    proposal_x2: Tensor = proposals[:, 2].unsqueeze(1)
    proposal_y2: Tensor = proposals[:, 3].unsqueeze(1)

    reference_x1: Tensor = reference_boxes[:, 0].unsqueeze(1)
    reference_y1: Tensor = reference_boxes[:, 1].unsqueeze(1)
    reference_x2: Tensor = reference_boxes[:, 2].unsqueeze(1)
    reference_y2: Tensor = reference_boxes[:, 3].unsqueeze(1)

    p_width, p_height = proposal_x2 - proposal_x1, proposal_y2 - proposal_y1
    p_ctr_x = proposal_x1 + p_width * .5
    p_ctr_y = proposal_y1 + p_height * .5

    r_width, r_height = reference_x2 - reference_x1, reference_y2 - reference_y1
    r_ctr_x = reference_x1 + r_width * .5
    r_ctr_y = reference_y1 + r_height * .5

    target_dx = weight_x * (r_ctr_x - p_ctr_x) / p_width
    target_dy = weight_y * (r_ctr_y - p_ctr_y) / p_width
    target_dw = weight_width * torch.log(r_width / p_width)
    target_dh = weight_height * torch.log(r_height / p_height)

    return torch.cat([target_dx, target_dy, target_dw, target_dh], dim=1)


class BoxCoder:
    def __init__(self, weights: Tuple[float, float, float, float], bbox_xform_clip: float = math.log(1000 / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes: List[Tensor], proposals: List[Tensor]) -> List[Tensor]:
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)

        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes: Tensor, proposals: Tensor) -> Tensor:
        targets = encode_boxes(reference_boxes, proposals, self.weights)

        return targets

    def decode(self, rel_codes: Tensor, boxes: List[Tensor]) -> Tensor:
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)

        box_sum = 0
        for val in boxes_per_image:
            box_sum += val

        pred_boxes = self.decode_single(
            rel_codes, concat_boxes
        )

        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)

        return pred_boxes

    def decode_single(self, relative_codes: Tensor, boxes: Tensor) -> Tensor:
        dx: Tensor = relative_codes[:, 0].unsqueeze(1)
        dy: Tensor = relative_codes[:, 1].unsqueeze(1)
        dw: Tensor = relative_codes[:, 2].unsqueeze(1)
        dh: Tensor = relative_codes[:, 3].unsqueeze(1)

        weight_x, weight_y, weight_w, weight_h = self.weights
        dx /= weight_x
        dy /= weight_y
        dw /= weight_w
        dh /= weight_h

        ref_x1, ref_y1 = boxes[:, 0], boxes[:, 1]
        ref_x2, ref_y2 = boxes[:, 2], boxes[:, 3]

        ref_width, ref_height = ref_x2 - ref_x1, ref_y2 - ref_y1
        ref_ctr_x, ref_ctr_y = ref_x1 + .5 * ref_width, ref_y1 + .5 * ref_height

        pred_ctr_x = dx * ref_width[:, None] + ref_ctr_x[:, None]
        pred_ctr_y = dy * ref_height[:, None] + ref_ctr_y[:, None]

        dw, dh = torch.clamp(dw, max=self.bbox_xform_clip), torch.clamp(dh, max=self.bbox_xform_clip)

        pred_width = torch.exp(dw) * ref_width[:, None]
        pred_height = torch.exp(dh) * ref_height[:, None]

        pred_x1 = pred_ctr_x - .5 * pred_width
        pred_y1 = pred_ctr_y - .5 * pred_height
        pred_x2 = pred_ctr_x + .5 * pred_width
        pred_y2 = pred_ctr_y + .5 * pred_height

        return torch.cat((pred_x1, pred_y1, pred_x2, pred_y2), dim=1)


class Matcher:
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        'BELOW_LOW_THRESHOLD': int,
        'BETWEEN_THRESHOLDS': int,
    }

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold  # 0.7
        self.low_threshold = low_threshold  # 0.3
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        if match_quality_matrix.numel() == 0:
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        matched_vals, matches = match_quality_matrix.max(dim=0)  # the dimension to reduce.
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
                matched_vals < self.high_threshold
        )
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD  # -1

        matches[between_thresholds] = self.BETWEEN_THRESHOLDS  # -2

        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)  # the dimension to reduce.

        gt_pred_pairs_of_highest_quality = torch.where(
            torch.eq(match_quality_matrix, highest_quality_foreach_gt[:, None])
        )

        pre_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pre_inds_to_update] = all_matches[pre_inds_to_update]


class BalancedPositiveNegativeSampler:
    def __init__(self, batch_size_per_image, positive_fraction):
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        pos_idx = []
        neg_idx = []

        for matched_idxs_per_image in matched_idxs:
            positive = torch.where(torch.ge(matched_idxs_per_image, 1))[0]
            negative = torch.where(torch.eq(matched_idxs_per_image, 0))[0]

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(negative.numel(), num_neg)

            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


def smooth_l1_loss(input: Tensor, target: Tensor, beta: float = 1. / 9, size_average: bool = True):
    n = torch.abs(input - target)
    cond = torch.lt(n, beta)
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()

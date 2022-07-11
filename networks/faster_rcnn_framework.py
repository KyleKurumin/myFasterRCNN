from collections import OrderedDict
from typing import List, Optional, Dict, Union

import torch
import torch.nn as nn

from networks.rpn_functions import RegionProposalNetwork, RPNHead


class FasterRCNNBase(nn.Module):
    def __init__(self, backbone: nn.Module,
                 rpn: RegionProposalNetwork,
                 roi_heads: RPNHead,
                 transform: nn.Module,
                 training: bool = True):
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self._has_warned = False
        self.training = training

    def eager_outputs(self, losses: Dict[str, torch.Tensor],
                      detections: List[Dict[str, torch.Tensor]]) \
            -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        if self.training:
            return losses
        return detections

    def forward(self, images: List[torch.Tensor],
                targets: Optional[List[Dict[str, torch.Tensor]]] = None) \
            -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        if self.training and targets is None:
            raise ValueError('Targets are supposed to be passed in training mode.')

        if self.training:
            for target in targets:
                boxes = target['boxes']
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(f'Shape of targets boxes are expected to be [N:4], got {boxes.shape}')
                else:
                    raise ValueError(f'Targets boxes are expected to be of type torch.Tensor, got {type(boxes)}.')

        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, target = self.transform(images, targets)

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        proposals, proposal_loss = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = detector_losses | proposal_loss

        return self.eager_outputs(losses, detections)

# class FasterRCNN(FasterRCNNBase):
#     def __init__(self, backbone):
#         super(FasterRCNN, self).__init__(backbone)

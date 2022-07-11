import math
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn

from networks.image_list import ImageList


def resize_boxes(bbox: torch.Tensor, original_size: Tuple[int, int], new_size: Tuple[int, int]) -> torch.Tensor:
    ratios = [
        torch.tensor(new,
                     dtype=torch.float32, device=bbox.device) / torch.tensor(original, dtype=torch.float32,
                                                                             device=bbox.device)
        for new, original in zip(new_size, original_size)
    ]

    height_ratio, width_ratio = ratios
    xmin, ymin, xmax, ymax = bbox.unbind(dim=1)
    xmin, xmax = xmin * width_ratio, xmax * width_ratio
    ymin, ymax = ymin * height_ratio, ymax * height_ratio

    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


class GeneralizedRCNNTransform(nn.Module):
    def __init__(self, min_size: int,
                 max_size: int,
                 image_mean: List[float],
                 image_std: List[float]):
        super(GeneralizedRCNNTransform, self).__init__()

        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def normalize(self, image: torch.Tensor):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k):
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    def max_by_axis(self, sizes: List[List[int]]) -> List[int]:
        values, indices = torch.max(torch.as_tensor(sizes), dim=0)
        return values.tolist()

    def post_process(self,
                     result: List[Dict[str, torch.Tensor]],
                     image_sizes: List[Tuple[int, int]],
                     original_sizes: List[Tuple[int, int]]) -> List[Dict[str, torch.Tensor]]:
        if self.training:
            return result
        for i, (pred, resized, original) in enumerate(zip(result, image_sizes, original_sizes)):
            boxes = pred['boxes']
            boxes = resize_boxes(boxes, resized, original)
            result[i] = boxes
        return result

    def resize(self,
               image: torch.Tensor,
               target: Optional[Dict[str, torch.Tensor]]) -> (torch.Tensor, Optional[Dict[str, torch.Tensor]]):
        h, w = image.shape[-2:]
        img_shape = torch.tensor(image.shape[-2:])
        min_size = float(torch.min(img_shape))
        max_size = float(torch.max(img_shape))

        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            size = float(self.min_size[-1])

        scale_factor = size / min_size

        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size

        image = torch.nn.functional.interpolate(image.unsqueeze(0), scale_factor=scale_factor, mode='bilinear',
                                                align_corners=False).squeeze(0)

        if target is not None:
            bbox = resize_boxes(target['boxes'], (h, w), image.shape[-2:])
            target['boxes'] = bbox

        return image, target

    def batch_images(self, images: List[torch.Tensor], size_divisible: int = 32) -> torch.Tensor:
        max_channel, max_height, max_width = self.max_by_axis([list(img.shape) for img in images])

        max_height = math.ceil(max_height / size_divisible) * size_divisible
        max_width = math.ceil(max_width / size_divisible) * size_divisible

        batch_shape = [len(images)] + [max_channel, max_height, max_width]
        batched_images = torch.zeros(batch_shape, dtype=torch.float32, device=images[0].device)

        for img, pad_img in zip(images, batched_images):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
        return batched_images

    def forward(self,
                images: List[torch.Tensor],
                targets: Optional[List[Dict[str, torch.Tensor]]] = None) \
            -> (ImageList, Optional[Dict[str, torch.Tensor]]):
        for i in range(len(images)):
            image = images[i]
            target = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError(f'Images is expected to be a list of 3d tensors of shape [C, H, W], got {image.shape}')

            image = self.normalize(image)
            image, target = self.resize(image, target)
            images[i] = image
            if targets is not None and target is not None:
                targets[i] = target

        image_sizes = [tuple(img.shape[-2:]) for img in images]

        images = self.batch_images(images)
        image_list = ImageList(images, image_sizes)

        return image_list, targets

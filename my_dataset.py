import json
import os
from typing import Optional, Dict, Union, List, Iterable

import lxml.etree
import torch
import torchvision.transforms
from PIL import Image
from lxml import etree

from torch.utils.data import Dataset

import transforms_utils
from networks.transforms import GeneralizedRCNNTransform

JSON_LIKE = Union[None, str, Iterable['JSON_LIKE']]


class VOCDataSet(Dataset):
    def __init__(self, voc_root: str,
                 transforms: transforms_utils.Compose,
                 train_set: bool = True) \
            -> None:
        self.root = os.path.join(voc_root, 'VOCdevkit', 'VOC2012')
        self.img_root = os.path.join(self.root, 'JPEGImages')
        self.annotations_root = os.path.join(self.root, 'Annotations')

        file_lists = os.path.join(self.root, 'ImageSets', 'Main', 'train.txt' if train_set else 'test.txt')
        with open(file_lists, 'r') as f:
            self.xml_lists = [os.path.join(self.annotations_root, line.strip() + '.xml') for line in f.readlines()]

        with open('./pascal_voc_classes.json', 'r') as f:
            self.class_dict = json.load(f)

        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.xml_lists)

    def __getitem__(self, idx: int) -> (torch.Tensor, Optional[Dict[str, torch.Tensor]]):
        xml_path = self.xml_lists[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)

        parsed_xml = self.parse_xml_obj(xml)
        image_path = os.path.join(self.img_root, parsed_xml['filename'])
        image = Image.open(image_path)
        if image.format != 'JPEG':
            raise ValueError("Image format is not JPEG.")

        boxes = []
        labels = []
        is_crowd = []

        for obj in parsed_xml.get('objects', []):
            bndbox = obj['bndbox']
            xmin = int(bndbox['xmin'])
            xmax = int(bndbox['xmax'])
            ymin = int(bndbox['ymin'])
            ymax = int(bndbox['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj['name']])
            is_crowd.append(int(obj['difficult']))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        is_crowd = torch.as_tensor(is_crowd, dtype=torch.int64)

        image_id = torch.tensor([idx])
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': areas,
            'is_crowd': is_crowd
        }

        if self.transforms:
            image, target = self.transforms(image, target)
        else:
            image = torchvision.transforms.ToTensor()(image)

        return image, target

    def parse_xml_obj(self, xml_obj: lxml.etree.Element) -> JSON_LIKE:
        if len(xml_obj) == 0:
            return xml_obj.text

        result = {}
        for child in xml_obj:
            if child.tag != 'object':
                result[child.tag] = self.parse_xml_obj(child)
            else:
                result.setdefault('objects', []).append(self.parse_xml_obj(child))

        return result

    def get_height_and_width(self, idx: int) -> (int, int):
        xml_path = self.xml_lists[idx]
        with open(xml_path, 'r') as fd:
            xml_str = fd.read()
        xml = etree.fromstring(xml_str)
        parsed_xml = self.parse_xml_obj(xml)
        width, height = int(parsed_xml['size']['width']), int(parsed_xml['size']['height'])
        return width, height


if __name__ == '__main__':
    transforms = transforms_utils.Compose([transforms_utils.ToTensor()])
    dataset = VOCDataSet('VOCtrainval_11-May-2012', transforms)

    images, targets = dataset[5]
    g = GeneralizedRCNNTransform(min_size=400, max_size=800, image_mean=[0.485, 0.456, 0.406],
                                 image_std=[0.229, 0.224, 0.225])
    g([images], [targets])
    pass

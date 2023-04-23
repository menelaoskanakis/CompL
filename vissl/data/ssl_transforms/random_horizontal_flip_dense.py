# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from typing import Any, Dict

import torch
import torchvision.transforms.functional as TF
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
import scipy.ndimage
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import PIL.Image as Image
import numpy as np


@register_transform("RandomHorizontalFlipDense")
class RandomHorizontalFlipDense(getattr(transforms, 'RandomHorizontalFlip')):
    def __call__(self, input):
        if torch.rand(1) < self.p:
            # image = F.hflip(image)
            # if semseg is not None:
            #     semseg = F.hflip(semseg)
            for key, tensor in input.items():
                input[key] = F.hflip(tensor)
                if key == 'normals':
                    _normals = np.array(tensor).astype(np.float32)
                    _normals = 2.0 * _normals / 255.0 - 1.0
                    _normals[:, :, 0] *= -1
                    _normals = (_normals + 1.0) / 2.0 * 255.0
                    tensor = Image.fromarray(_normals.astype(np.uint8))
        return input

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RandomHorizontalFlipDense":
        p = config.get("p", 0.5)
        return cls(p=p)

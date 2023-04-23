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


@register_transform("CenterCropDense")
class CenterCropDense(getattr(transforms, 'CenterCrop')):
    def __call__(self, input):
        # image = F.center_crop(image, self.size)
        # if semseg is not None:
        #     semseg = F.center_crop(semseg, self.size)
        for key, tensor in input.items():
            input[key] = F.center_crop(tensor, self.size)

        return input

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CenterCropDense":
        size = config.get("size", None)
        return cls(size=size)

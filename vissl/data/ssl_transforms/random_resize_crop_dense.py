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


@register_transform("RandomResizedCropDense")
class RandomResizedCropDense(getattr(transforms, 'RandomResizedCrop')):
    def __call__(self, image, semseg=None):
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        image = F.resized_crop(image, i, j, h, w, self.size, self.interpolation)
        if semseg is not None:
            semseg = F.resized_crop(semseg, i, j, h, w, self.size, interpolation=Image.NEAREST)

        return image, semseg

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RandomResizedCropDense":
        raise NotImplementedError
        size = config.get("size", None)
        scale = config.get("scale", (0.08, 1.0))
        ratio = config.get("ratio", (3. / 4., 4. / 3.))
        interpolation = config.get("interpolation", Image.BILINEAR)
        return cls(size=size, scale=scale, ratio=ratio, interpolation=interpolation)

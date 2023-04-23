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

@register_transform("ToTensorDense")
class ToTensorDense(getattr(transforms, 'ToTensor')):
    def __call__(self, input):
        # image = F.to_tensor(image)
        # if semseg is not None:
        #     semseg = torch.as_tensor(np.array(semseg), dtype=torch.int64)
        for key, tensor in input.items():
            if key in ['image']:
                input[key] = F.to_tensor(tensor)
            elif key in ['normals']:
                input[key] = F.to_tensor(tensor)
            elif key in ['semseg', 'edge']:
                input[key] = torch.as_tensor(np.array(tensor), dtype=torch.int64)
            elif key in ['depth']:
                input[key] = torch.as_tensor(np.array(tensor), dtype=torch.float)

        return input

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ToTensorDense":
        return cls()

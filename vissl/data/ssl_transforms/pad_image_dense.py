# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from typing import Any, Dict

import torch
import torchvision.transforms.functional as TF
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
import scipy.ndimage
import numpy as np
import cv2
import PIL
import numbers
from PIL import Image


@register_transform("PadImageDense")
class PadImageDense(ClassyTransform):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = tuple([int(size), int(size)])
        elif isinstance(size, list):
            self.size = tuple(size)
        else:
            raise ValueError('Crop size must be a number or a list of numbers')

        self.fill_index = {'image': [123, 116, 103],
                           'edge': 255,
                           'human_parts': 255,
                           'semseg': 255,
                           'normals': [0, 0, 0],
                           'sal': 255,
                           'depth': 0.
                           }

    def pad(self, key, unpadded):
        unpadded_shape = np.shape(unpadded)
        delta_height = max(self.size[0] - unpadded_shape[0], 0)
        delta_width = max(self.size[1] - unpadded_shape[1], 0)

        pad_value = self.fill_index[key]
        max_height = max(self.size[0], unpadded_shape[0])
        max_width = max(self.size[1], unpadded_shape[1])
        if key in {'image', 'normals'}:
            padded = PIL.Image.new("RGB", (max_width, max_height), color=tuple(pad_value))
            padded.paste(unpadded, (delta_width//2, delta_height//2))
        # elif key in {'semseg', 'human_parts', 'edge', 'sal'}:
        elif key in {'semseg', 'edge'}:
            padded = PIL.Image.new("L", (max_width, max_height), color=pad_value)
            padded.paste(unpadded, (delta_width // 2, delta_height // 2))
        elif key in {'depth'}:
            # padded = PIL.Image.new("L", (max_width, max_height))
            padded = Image.fromarray(np.ones((max_height, max_width)) * pad_value)
            padded.paste(unpadded, (delta_width // 2, delta_height // 2))
        else:
            raise ValueError('Key {} for input origin is not supported'.format(key))

        return padded

    def __call__(self, input):
        for key, tensor in input.items():
            input[key] = self.pad(key, tensor)
        # image = self.pad('image', image)
        # if semseg is not None:
        #     semseg = self.pad('semseg', semseg)
        return input

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PadImageDense":
        """
        Instantiates ImgRotatePil from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgRotatePil instance.
        """
        size = config.get("size", None)
        return cls(size=size)

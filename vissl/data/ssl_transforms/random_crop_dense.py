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
from .pad_image_dense import PadImageDense
import random


@register_transform("RandomCropDense")
class RandomCropDense(ClassyTransform):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = tuple([int(size), int(size)])
        elif isinstance(size, list):
            self.size = tuple(size)
        else:
            raise ValueError('Crop size must be a number or a list of numbers')

        self.padding = PadImageDense(size)

    def get_random_crop_loc(self, uncropped):
        """Gets a random crop location.
        Args:
            uncropped: Image or target to be cropped.
        Returns:
            Cropping region.
        """
        uncropped_shape = np.shape(uncropped)
        img_height = uncropped_shape[0]
        img_width = uncropped_shape[1]

        desired_height = self.size[0]
        desired_width = self.size[1]
        if img_height == desired_height and img_width == desired_width:
            return None
        else:
            # Get random offset uniformly from [0, max_offset)
            max_offset_height = img_height - desired_height
            max_offset_width = img_width - desired_width

            offset_height = random.randint(0, max_offset_height)
            offset_width = random.randint(0, max_offset_width)
            crop_loc = {'height': [offset_height, offset_height + desired_height],
                        'width': [offset_width, offset_width + desired_width],
                        }
            return crop_loc

    def __call__(self, input):

        # Ensure the image is at least as large as the desired size
        input = self.padding(input)

        crop_location = self.get_random_crop_loc(input['image'])
        if crop_location is None:
            return input

        # if semseg is not None:
        #     semseg = semseg.crop((crop_location['width'][0], crop_location['height'][0],
        #                         crop_location['width'][1], crop_location['height'][1]))
        for key, tensor in input.items():
            input[key] = tensor.crop((crop_location['width'][0], crop_location['height'][0],
                                            crop_location['width'][1], crop_location['height'][1]))
        return input

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RandomCropDense":
        """
        Instantiates ImgRotatePil from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgRotatePil instance.
        """
        size = config.get("size", None)
        return cls(size=size)

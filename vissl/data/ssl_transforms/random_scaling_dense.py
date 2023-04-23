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

@register_transform("RandomScalingDense")
class RandomScalingDense(ClassyTransform):

    def __init__(self, min_scale_factor=1.0, max_scale_factor=1.0, step_size=0):
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.step_size = step_size

    def get_random_scale(self, min_scale_factor, max_scale_factor, step_size):
        """Gets a random scaling value.
        Args:
            min_scale_factor: Minimum scale value.
            max_scale_factor: Maximum scale value.
            step_size: The step size from minimum to maximum value.
        Returns:
            A random scale value selected between minimum and maximum value.
        """
        if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
            raise ValueError('Unexpected value of min_scale_factor')

        if min_scale_factor == max_scale_factor:
            min_scale_factor = float(min_scale_factor)
            return min_scale_factor

        # Uniformly sampling of the value from [min, max) when step_size = 0
        if step_size == 0:
            return np.random.uniform(low=min_scale_factor, high=max_scale_factor)
        # Else, randomly select one discrete value from [min, max]
        else:
            num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
            rand_step = np.random.randint(num_steps)
            rand_scale = min_scale_factor + rand_step * step_size
            return rand_scale

    def scale(self, key, unscaled, scale=1.0):
        """Randomly scales image and label.
        Args:
            key: Key indicating the uscaled input origin
            unscaled: Image or target to be scaled.
            scale: The value to scale image and label.
        Returns:
            scaled: The scaled image or target
        """
        # No random scaling if scale == 1.
        if scale == 1.0:
            return unscaled
        image_shape = np.shape(unscaled)[0:2]
        new_dim = tuple([int(x * scale) for x in image_shape])
        # if key in {'image', 'normals'}:
        if key in {'image', 'depth', 'normals'}:
            scaled = unscaled.resize(new_dim[::-1], resample=PIL.Image.LINEAR)
        # elif key in {'semseg', 'human_parts', 'edge', 'sal'}:
        elif key in {'semseg', 'edge'}:
            scaled = unscaled.resize(new_dim[::-1], resample=PIL.Image.NEAREST)
        else:
            raise ValueError('Key {} for input origin is not supported'.format(key))
        # we adjust depth maps with rescaling
        if key == 'depth':
            np_scaled = np.array(scaled, dtype=np.float)
            np_scaled /= scale
            scaled = PIL.Image.fromarray(np_scaled)

        return scaled

    def __call__(self, input):
        random_scale = self.get_random_scale(self.min_scale_factor,
                                             self.max_scale_factor,
                                             self.step_size)
        # image = self.scale('image', image, scale=random_scale)
        # if semseg is not None:
        #     semseg = self.scale('semseg', semseg, scale=random_scale)
        for key, tensor in input.items():
            input[key] = self.scale(key, tensor, scale=random_scale)
        return input

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RandomScalingDense":
        """
        Instantiates ImgRotatePil from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgRotatePil instance.
        """
        min_scale_factor = config.get("min_scale_factor", 1.0)
        max_scale_factor = config.get("max_scale_factor", 1.0)
        step_size = config.get("step_size", 0)
        return cls(min_scale_factor=min_scale_factor, max_scale_factor=max_scale_factor, step_size=step_size)

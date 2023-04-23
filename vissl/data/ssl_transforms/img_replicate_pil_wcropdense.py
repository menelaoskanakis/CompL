# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from typing import Any, Dict

from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from torchvision.transforms import functional as F
import numpy as np
import PIL

@register_transform("ImgReplicatePilWCropDense")
class ImgReplicatePilWCropDense(ClassyTransform):
    """
    Adds the same image multiple times to the batch K times so that the batch.
    Size is now N*K. Use the simclr_collator to convert into batches.

    This transform is useful when generating multiple copies of the same image,
    for example, when training contrastive methods.
    """

    def __init__(self, num_times: int = 2, patch_scale: float = 0.1, patch_distance: int = 32):
        """
        Args:
            num_times (int): how many times should the image be replicated.
        """
        assert isinstance(
            num_times, int
        ), f"num_times must be an integer. Found {type(num_times)}"
        assert num_times == 2, f"num_times {num_times} must be 2."

        assert patch_scale > 0, f"patch_scale {patch_scale} must be greater than zero."

        assert isinstance(
            patch_distance, int
        ), f"patch_distance must be an integer. Found {type(patch_distance)}"
        assert patch_distance > 0, f"patch_distance {patch_distance} must be greater than zero."

        self.num_times = num_times
        self.patch_scale = patch_scale
        self.patch_distance = patch_distance

    def __call__(self, image):

        # Resize so that the smallest dimension is self.size
        image_shape = np.shape(image)[0:2]
        new_dim = tuple([int(x + (x * self.patch_scale)) for x in image_shape])
        scaled = image.resize(new_dim[::-1], resample=PIL.Image.LINEAR)

        # Get dimensions for the two crops
        # max_allowed_size = self.size - self.patch_distance - self.patch_size

        # max_offset_height = img_height - desired_height
        max_offset_height = int(np.floor(new_dim[0] - self.patch_distance - image_shape[0]))
        # max_offset_width = img_width - desired_width
        max_offset_width = int(np.floor(new_dim[1] - self.patch_distance - image_shape[1]))
        if max_offset_height > 0 and max_offset_width > 0:
            start_h = np.random.randint(max_offset_height)
            start_w = np.random.randint(max_offset_width)
        elif max_offset_height == 0 and max_offset_width == 0:
            start_h = 0
            start_w = 0
        else:
            raise ValueError("The maximum allowed size (new size - patch distance - original image shape) must be >= 0. Got {} and {} instead".format(max_offset_height, max_offset_width))

        output = []
        new_img_1 = scaled.copy()
        new_img_1 = F.crop(new_img_1, start_h, start_w, image_shape[0], image_shape[1])
        output.append(new_img_1)

        new_img_2 = scaled.copy()
        new_img_2 = F.crop(new_img_2, start_h + self.patch_distance, start_w + self.patch_distance, image_shape[0], image_shape[1])
        output.append(new_img_2)
        return output

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgReplicatePilWCropDense":
        """
        Instantiates ImgReplicatePil from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgReplicatePil instance.
        """
        num_times = config.get("num_times", 2)
        patch_scale = config.get("patch_scale", 0.1)
        patch_distance = config.get("patch_distance", 32)
        logging.info(f"ImgReplicatePilWCropDense | Using num_times: {num_times}")
        logging.info(f"ImgReplicatePilWCropDense | Using patch_scale: {patch_scale}")
        logging.info(f"ImgReplicatePilWCropDense | Using patch_distance: {patch_distance}")
        return cls(num_times=num_times, patch_scale=patch_scale, patch_distance=patch_distance)

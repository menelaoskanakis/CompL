# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from typing import Any, Dict

from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image

interpolation = {
    'image': Image.BILINEAR,
    'semseg': Image.NEAREST
}
@register_transform("ImgReplicatePilWCrop")
class ImgReplicatePilWCrop(ClassyTransform):
    """
    Adds the same image multiple times to the batch K times so that the batch.
    Size is now N*K. Use the simclr_collator to convert into batches.

    This transform is useful when generating multiple copies of the same image,
    for example, when training contrastive methods.
    """

    def __init__(self, num_times: int = 2, patch_size: int = 128, patch_distance: int = 32, size: int = 512):
        """
        Args:
            num_times (int): how many times should the image be replicated.
        """
        assert isinstance(
            num_times, int
        ), f"num_times must be an integer. Found {type(num_times)}"
        assert num_times == 2, f"num_times {num_times} must be 2."

        assert isinstance(
            patch_size, int
        ), f"patch_size must be an integer. Found {type(patch_size)}"
        assert patch_size > 0, f"patch_size {patch_size} must be greater than zero."

        assert isinstance(
            patch_distance, int
        ), f"patch_distance must be an integer. Found {type(patch_distance)}"
        assert patch_distance > 0, f"patch_distance {patch_distance} must be greater than zero."

        assert isinstance(
            size, int
        ), f"size must be an integer. Found {type(size)}"
        assert size > 0, f"size {size} must be greater than zero."

        self.num_times = num_times
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        self.size = size

    def __call__(self, input):
        max_allowed_size = self.size - self.patch_distance - self.patch_size
        if max_allowed_size > 0:
            start_h = np.random.randint(max_allowed_size)
            start_w = np.random.randint(max_allowed_size)
        elif max_allowed_size == 0:
            start_h = 0
            start_w = 0
        else:
            raise ValueError(
                "The maximum allowed size (size - patch_distance - patch_size) must be >= 0. Got {} instead".format(
                    max_allowed_size))

        output = {}
        for key, tensor in input.items():
            output[key] = []
            # Resize so that the smallest dimension is self.size
            _tensor = F.resize(tensor, self.size, interpolation=interpolation[key])
            # Centre crop to turn into square given the desired dimension
            _tensor = F.center_crop(_tensor, self.size)

            _tensor_1 = _tensor.copy()
            _tensor_1 = F.crop(_tensor_1, start_h, start_w, self.patch_size, self.patch_size)
            _tensor_1 = F.resize(_tensor_1, self.size, interpolation=interpolation[key])
            output[key].append(_tensor_1)

            _tensor_2 = _tensor.copy()
            _tensor_2 = F.crop(_tensor_2, start_h + self.patch_distance, start_w + self.patch_distance, self.patch_size,
                               self.patch_size)
            _tensor_2 = F.resize(_tensor_2, self.size, interpolation=interpolation[key])
            output[key].append(_tensor_2)
        return output

    # def __call__(self, image):
    #
    #     # Resize so that the smallest dimension is self.size
    #     image = F.resize(image, self.size)
    #     # Centre crop to turn into square given the desired dimension
    #     image = F.center_crop(image, self.size)
    #
    #     # Get dimensions for the two crops
    #     max_allowed_size = self.size - self.patch_distance - self.patch_size
    #     if max_allowed_size > 0:
    #         start_h = np.random.randint(max_allowed_size)
    #         start_w = np.random.randint(max_allowed_size)
    #     elif max_allowed_size == 0:
    #         start_h = 0
    #         start_w = 0
    #     else:
    #         raise ValueError("The maximum allowed size (size - patch_distance - patch_size) must be >= 0. Got {} instead".format(max_allowed_size))
    #
    #     output = []
    #     new_img_1 = image.copy()
    #     new_img_1 = F.crop(new_img_1, start_h, start_w, self.patch_size, self.patch_size)
    #     new_img_1 = F.resize(new_img_1, self.size)
    #     output.append(new_img_1)
    #
    #     new_img_2 = image.copy()
    #     new_img_2 = F.crop(new_img_2, start_h + self.patch_distance, start_w + self.patch_distance, self.patch_size, self.patch_size)
    #     new_img_2 = F.resize(new_img_2, self.size)
    #     output.append(new_img_2)
    #
    #     return output

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgReplicatePilWCrop":
        """
        Instantiates ImgReplicatePil from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgReplicatePil instance.
        """
        num_times = config.get("num_times", 2)
        patch_size = config.get("patch_size", 128)
        patch_distance = config.get("patch_distance", 32)
        size = config.get("size", 512)
        logging.info(f"ImgReplicatePilWCrop | Using num_times: {num_times}")
        logging.info(f"ImgReplicatePilWCrop | Using patch_size: {patch_size}")
        logging.info(f"ImgReplicatePilWCrop | Using patch_distance: {patch_distance}")
        logging.info(f"ImgReplicatePilWCrop | Using size: {size}")
        return cls(num_times=num_times, patch_size=patch_size, patch_distance=patch_distance, size=size)

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from typing import Any, Dict

from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
import torch
from math import pi

interpolation = {
    'image': Image.BILINEAR,
    'semseg': Image.NEAREST
}

def meshgrid(B, H, W, dtype, device, normalized=False):
    """Create mesh-grid given batch size, height and width dimensions.
    Parameters
    ----------
    B: int
        Batch size
    H: int
        Grid Height
    W: int
        Batch size
    dtype: torch.dtype
        Tensor dtype
    device: str
        Tensor device
    normalized: bool
        Normalized image coordinates or integer-grid.
    Returns
    -------
    xs: torch.Tensor
        Batched mesh-grid x-coordinates (BHW).
    ys: torch.Tensor
        Batched mesh-grid y-coordinates (BHW).
    """
    if normalized:
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    else:
        xs = torch.linspace(0, W-1, W, device=device, dtype=dtype)
        ys = torch.linspace(0, H-1, H, device=device, dtype=dtype)
    ys, xs = torch.meshgrid([ys, xs])
    return xs.repeat([B, 1, 1]), ys.repeat([B, 1, 1])


def image_grid(B, H, W, dtype, device, ones=True, normalized=False):
    """Create an image mesh grid with shape B3HW given image shape BHW
    Parameters
    ----------
    B: int
        Batch size
    H: int
        Grid Height
    W: int
        Batch size
    dtype: str
        Tensor dtype
    device: str
        Tensor device
    ones : bool
        Use (x, y, 1) coordinates
    normalized: bool
        Normalized image coordinates or integer-grid.
    Returns
    -------
    grid: torch.Tensor
        Mesh-grid for the corresponding image shape (B3HW)
    """
    xs, ys = meshgrid(B, H, W, dtype, device, normalized=normalized)
    coords = [xs, ys]
    if ones:
        coords.append(torch.ones_like(xs))  # BHW
    grid = torch.stack(coords, dim=1)  # B3HW
    return grid

@register_transform("ImgReplicatePilWHomAdapt")
class ImgReplicatePilWHomAdapt(ClassyTransform):
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

    def sample_homography(self,
            shape, perspective=True, scaling=True, rotation=True, translation=True,
            n_scales=100, n_angles=100, scaling_amplitude=0.1, perspective_amplitude=0.4,
            patch_ratio=0.8, max_angle=pi / 4):
        """ Sample a random homography that includes perspective, scale, translation and rotation operations."""

        hw_ratio = float(shape[0]) / float(shape[1])

        pts1 = np.stack([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], axis=0)
        pts2 = pts1.copy() * patch_ratio
        pts2[:, 1] *= hw_ratio

        if perspective:
            perspective_amplitude_x = np.random.normal(0., perspective_amplitude / 2, (2))
            perspective_amplitude_y = np.random.normal(0., hw_ratio * perspective_amplitude / 2, (2))

            perspective_amplitude_x = np.clip(perspective_amplitude_x, -perspective_amplitude / 2,
                                              perspective_amplitude / 2)
            perspective_amplitude_y = np.clip(perspective_amplitude_y, hw_ratio * -perspective_amplitude / 2,
                                              hw_ratio * perspective_amplitude / 2)

            pts2[0, 0] -= perspective_amplitude_x[1]
            pts2[0, 1] -= perspective_amplitude_y[1]

            pts2[1, 0] -= perspective_amplitude_x[0]
            pts2[1, 1] += perspective_amplitude_y[1]

            pts2[2, 0] += perspective_amplitude_x[1]
            pts2[2, 1] -= perspective_amplitude_y[0]

            pts2[3, 0] += perspective_amplitude_x[0]
            pts2[3, 1] += perspective_amplitude_y[0]

        if scaling:
            random_scales = np.random.normal(1, scaling_amplitude / 2, (n_scales))
            random_scales = np.clip(random_scales, 1 - scaling_amplitude / 2, 1 + scaling_amplitude / 2)

            scales = np.concatenate([[1.], random_scales], 0)
            center = np.mean(pts2, axis=0, keepdims=True)
            scaled = np.expand_dims(pts2 - center, axis=0) * np.expand_dims(
                np.expand_dims(scales, 1), 1) + center
            valid = np.arange(n_scales)  # all scales are valid except scale=1
            idx = valid[np.random.randint(valid.shape[0])]
            pts2 = scaled[idx]

        if translation:
            t_min, t_max = np.min(pts2 - [-1., -hw_ratio], axis=0), np.min([1., hw_ratio] - pts2, axis=0)
            pts2 += np.expand_dims(np.stack([np.random.uniform(-t_min[0], t_max[0]),
                                             np.random.uniform(-t_min[1], t_max[1])]),
                                   axis=0)

        if rotation:
            angles = np.linspace(-max_angle, max_angle, n_angles)
            angles = np.concatenate([[0.], angles], axis=0)

            center = np.mean(pts2, axis=0, keepdims=True)
            rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                           np.cos(angles)], axis=1), [-1, 2, 2])
            rotated = np.matmul(
                np.tile(np.expand_dims(pts2 - center, axis=0), [n_angles + 1, 1, 1]),
                rot_mat) + center

            valid = np.where(np.all((rotated >= [-1., -hw_ratio]) & (rotated < [1., hw_ratio]),
                                    axis=(1, 2)))[0]

            idx = valid[np.random.randint(valid.shape[0])]
            pts2 = rotated[idx]

        pts2[:, 1] /= hw_ratio

        def ax(p, q):
            return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

        def ay(p, q):
            return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

        a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
        p_mat = np.transpose(np.stack(
            [[pts2[i][j] for i in range(4) for j in range(2)]], axis=0))

        homography = np.matmul(np.linalg.pinv(a_mat), p_mat).squeeze()
        homography = np.concatenate([homography, [1.]]).reshape(3, 3)
        return homography

    def warp_homography(self, sources, homography):
        """Warp features given a homography
        Parameters
        ----------
        sources: torch.tensor (1,H,W,2)
            Keypoint vector.
        homography: torch.Tensor (3,3)
            Homography.
        Returns
        -------
        warped_sources: torch.tensor (1,H,W,2)
            Warped feature vector.
        """
        _, H, W, _ = sources.shape
        warped_sources = sources.clone().squeeze()
        warped_sources = warped_sources.view(-1, 2)
        warped_sources = torch.addmm(homography[:, 2], warped_sources, homography[:, :2].t())
        warped_sources.mul_(1 / warped_sources[:, 2].unsqueeze(1))
        warped_sources = warped_sources[:, :2].contiguous().view(1, H, W, 2)
        return warped_sources

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

        _tensor = F.resize(input['image'], self.size, interpolation=interpolation['image'])
        _tensor = F.center_crop(_tensor, self.size)

        output = {'image': []}
        _tensor_1 = _tensor.copy()
        _tensor_1 = F.crop(_tensor_1, start_h, start_w, self.patch_size, self.patch_size)
        _tensor_1 = F.resize(_tensor_1, self.size, interpolation=interpolation['image'])
        output['image'].append(_tensor_1)

        homography = self.sample_homography([self.size, self.size])
        print('ok1', homography)
        ma
        # homography = self.sample_homography([H, W])
                                            # patch_ratio=patch_ratio,
                                            # scaling_amplitude=scaling_amplitude,
                                            # max_angle=max_angle)
        homography = torch.from_numpy(homography).float().to(device)

        source_grid = image_grid(1, H, W,
                                 dtype=target_img.dtype,
                                 device=device,
                                 ones=False, normalized=True).clone().permute(0, 2, 3, 1)

        source_warped = warp_homography(source_grid, homography)
        source_img = torch.nn.functional.grid_sample(target_img, source_warped, align_corners=True)


        # output = {}
        # for key, tensor in input.items():
        #     output[key] = []
        #     # Resize so that the smallest dimension is self.size
        #     _tensor = F.resize(tensor, self.size, interpolation=interpolation[key])
        #     # Centre crop to turn into square given the desired dimension
        #     _tensor = F.center_crop(_tensor, self.size)
        #
        #     _tensor_1 = _tensor.copy()
        #     _tensor_1 = F.crop(_tensor_1, start_h, start_w, self.patch_size, self.patch_size)
        #     _tensor_1 = F.resize(_tensor_1, self.size, interpolation=interpolation[key])
        #     output[key].append(_tensor_1)
        #
        #     _tensor_2 = _tensor.copy()
        #     _tensor_2 = F.crop(_tensor_2, start_h + self.patch_distance, start_w + self.patch_distance, self.patch_size,
        #                        self.patch_size)
        #     _tensor_2 = F.resize(_tensor_2, self.size, interpolation=interpolation[key])
        #     output[key].append(_tensor_2)
        ma1
        return output

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgReplicatePilWHomAdapt":
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
        logging.info(f"ImgReplicatePilWHomAdapt | Using num_times: {num_times}")
        logging.info(f"ImgReplicatePilWHomAdapt | Using patch_size: {patch_size}")
        logging.info(f"ImgReplicatePilWHomAdapt | Using patch_distance: {patch_distance}")
        logging.info(f"ImgReplicatePilWHomAdapt | Using size: {size}")
        return cls(num_times=num_times, patch_size=patch_size, patch_distance=patch_distance, size=size)

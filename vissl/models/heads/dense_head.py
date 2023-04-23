# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List

import torch
import torch.nn as nn
from vissl.models.heads import register_model_head
from vissl.utils.hydra_config import AttrDict
from torch.nn import functional as F

@register_model_head("dense_head")
class DenseHead(nn.Module):
    """
    This module can be used to attach combination of {Linear, BatchNorm, Relu, Dropout}
    layers and they are fully configurable from the config file. The module also supports
    stacking multiple MLPs.

    Examples:
        Linear
        Linear -> BN
        Linear -> ReLU
        Linear -> Dropout
        Linear -> BN -> ReLU -> Dropout
        Linear -> ReLU -> Dropout
        Linear -> ReLU -> Linear -> ReLU -> ...
        Linear -> Linear -> ...
        ...

    Accepts a 2D input tensor. Also accepts 4D input tensor of shape `N x C x 1 x 1`.
    """

    def __init__(
        self,
        model_config: AttrDict,
        conv_dims: List[int],
        img_dims: List[int],
    ):
        """
        Args:
            model_config (AttrDict): dictionary config.MODEL in the config file
            use_bn (bool): whether to attach BatchNorm after Linear layer
            use_relu (bool): whether to attach ReLU after (Linear (-> BN optional))
            use_dropout (bool): whether to attach Dropout after
                                (Linear (-> BN -> relu optional))
            use_bias (bool): whether the Linear layer should have bias or not
            dims (int): dimensions of the linear layer. Example [8192, 1000] which
                        attaches `nn.Linear(8192, 1000, bias=True)`
        """
        super().__init__()
        self.conv_dims = nn.Conv2d(conv_dims[0], conv_dims[1], kernel_size=1, bias=True)
        self.h = img_dims[0]
        self.w = img_dims[1]

    def forward(self, batch: torch.Tensor):
        """
        Args:
            batch (torch.Tensor): 4D tensor of shape `N x C x H x W`
        Returns:
            out (torch.Tensor): 2D output torch tensor `N x C`
        """
        if isinstance(batch, list):
            assert (
                len(batch) == 1
            ), "MLP input should be either a tensor (4D) or list containing 1 tensor."
            batch = batch[0]
        out = self.conv_dims(batch)
        out = F.interpolate(out, size=(self.h, self.w), mode='bilinear', align_corners=False)
        return out

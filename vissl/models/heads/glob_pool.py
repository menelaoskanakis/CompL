# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List

import torch
import torch.nn as nn
from vissl.models.heads import register_model_head
from vissl.utils.hydra_config import AttrDict


@register_model_head("glob_pool")
class GlobPool(nn.Module):
    """
    This module can be used to define the pooling performed when coming from a dense architecture

    """

    def __init__(
        self,
        model_config: AttrDict,
        glob_pool: str,
    ):
        """
        Args:
            model_config (AttrDict): dictionary config.MODEL in the config file
            pool (int): Name of global pooling operation. Example [8192, 1000] which
                        attaches `nn.Linear(8192, 1000, bias=True)`
        """
        super().__init__()
        self.adapt_pool = getattr(torch.nn, glob_pool)(1)

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
        out = self.adapt_pool(batch)
        out = out.view(out.size(0), -1)
        return out

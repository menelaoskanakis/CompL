# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from enum import Enum
from typing import List

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import Bottleneck
from vissl.models.model_helpers import (
    Flatten,
    _get_norm,
    get_trunk_forward_outputs,
    transform_model_input_data_type,
)
from vissl.models.trunks import register_model_trunk
from vissl.models.additional_modules import hrnet
from vissl.utils.hydra_config import AttrDict
import os
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
import warnings


class SUPPORTED_DEPTHS(int, Enum):
    HN18 = 18
    HN30 = 30
    HN40 = 40
    HN48 = 48


class INPUT_CHANNEL(int, Enum):
    rgb = 3


@register_model_trunk("hrnet")
class HRNet(nn.Module):
    """
    Wrapper for TorchVison ResNet Model to support different depth and
    width_multiplier. We provide flexibility with LAB input, stride in last
    ResNet block and type of norm (BatchNorm, LayerNorm)
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        super(HRNet, self).__init__()
        self.model_config = model_config
        logging.info(
            "HRNet trunk, supports activation checkpointing. {}".format(
                "Activated"
                if self.model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
                else "Deactivated"
            )
        )

        self.trunk_config = self.model_config.TRUNK.TRUNK_PARAMS.HRNET
        self.depth = SUPPORTED_DEPTHS(self.trunk_config.DEPTH)
        self.use_checkpointing = (
            self.model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
        )
        self.num_checkpointing_splits = (
            self.model_config.ACTIVATION_CHECKPOINTING.NUM_ACTIVATION_CHECKPOINTING_SPLITS
        )

        model = getattr(hrnet, 'hrnet_w{}'.format(self.trunk_config.DEPTH))
        if 'PRETRAINED_PATH' not in self.trunk_config:
            pretrained_path = None
        else:
            pretrained_path = self.trunk_config.PRETRAINED_PATH

        model = model(model_dir=pretrained_path, pretrained=self.trunk_config.PRETRAINED)

        model_backbone = model

        # we mapped the layers of resnet model into feature blocks to facilitate
        # feature extraction at various layers of the model. The layers for which
        # to extract features is controlled by requested_feat_keys argument in the
        # forward() call.
        self._feature_blocks = nn.ModuleDict(
            [
                ("backbone", model_backbone),
            ]
        )

        # give a name mapping to the layers so we can use a common terminology
        # across models for feature evaluation purposes.
        self.feat_eval_mapping = {
            "backbone": "backbone",
        }

    def forward(
        self, x: torch.Tensor, out_feat_keys: List[str] = None
    ) -> List[torch.Tensor]:
        feat = transform_model_input_data_type(x, self.model_config)
        return get_trunk_forward_outputs(
            feat,
            out_feat_keys=out_feat_keys,
            feature_blocks=self._feature_blocks,
            feature_mapping=self.feat_eval_mapping,
            use_checkpointing=self.use_checkpointing,
            checkpointing_splits=self.num_checkpointing_splits,
        )

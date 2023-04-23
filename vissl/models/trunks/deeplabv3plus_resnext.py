# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from enum import Enum
from typing import List

import torch
import torch.nn as nn
from vissl.models.model_helpers import (
    _get_norm,
    get_trunk_forward_outputs,
    transform_model_input_data_type,
)
from vissl.models.trunks import register_model_trunk
from vissl.models.additional_modules import deeplabv3plus_resnet
from vissl.utils.hydra_config import AttrDict
import warnings


MODEL_URLS= {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


class SUPPORTED_DEPTHS(int, Enum):
    RN26 = 26
    RN50 = 50
    RN101 = 101


class INPUT_CHANNEL(int, Enum):
    rgb = 3


@register_model_trunk("deeplabv3plus_resnet")
class Deeplabv3plusResNeXt(nn.Module):
    """
    Wrapper for TorchVison ResNet Model to support different depth and
    width_multiplier. We provide flexibility with LAB input, stride in last
    ResNet block and type of norm (BatchNorm, LayerNorm)
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        super(Deeplabv3plusResNeXt, self).__init__()
        self.model_config = model_config
        logging.info(
            "DeepLabV3PlusResNeXT trunk, supports activation checkpointing. {}".format(
                "Activated"
                if self.model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
                else "Deactivated"
            )
        )

        self.trunk_config = self.model_config.TRUNK.TRUNK_PARAMS.RESNETS
        self.depth = SUPPORTED_DEPTHS(self.trunk_config.DEPTH)
        self.width_multiplier = self.trunk_config.WIDTH_MULTIPLIER
        warnings.warn('width_multiplier not implemented')
        self._norm_layer = _get_norm(self.trunk_config)
        warnings.warn('_norm_layer not implemented')
        self.groups = self.trunk_config.GROUPS
        warnings.warn('groups not implemented')
        self.zero_init_residual = self.trunk_config.ZERO_INIT_RESIDUAL
        warnings.warn('zero_init_residual not implemented')
        self.width_per_group = self.trunk_config.WIDTH_PER_GROUP
        warnings.warn('width_per_group not implemented')
        self.use_checkpointing = (
            self.model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
        )
        self.num_checkpointing_splits = (
            self.model_config.ACTIVATION_CHECKPOINTING.NUM_ACTIVATION_CHECKPOINTING_SPLITS
        )

        print("'resnet{}'.format(self.trunk_config.DEPTH)", 'resnet{}'.format(self.trunk_config.DEPTH))
        model = getattr(deeplabv3plus_resnet, 'resnet{}'.format(self.trunk_config.DEPTH))

        pretrained_path = self.trunk_config.get("PRETRAINED_PATH", None)
        pretrained = self.trunk_config.get("PRETRAINED", None)
        norm_layers = self.trunk_config.get("NORM_LAYERS", 'BatchNorm2d')
        train_norm_layers = self.trunk_config.get("TRAIN_NORM_LAYER", True)
        model = model(pretrained=pretrained, path=pretrained_path,
                      train_norm_layers=train_norm_layers,
                      norm_layers=norm_layers)

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

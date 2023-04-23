from typing import List

import torch
import torch.nn as nn
from vissl.models.heads import register_model_head
from vissl.utils.hydra_config import AttrDict
from torch.nn import init


def _init_weights(module, init_linear='normal'):
    assert init_linear in ['normal', 'kaiming'], \
        "Undefined init_linear: {}".format(init_linear)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if init_linear == 'normal':
                m.weight.data.normal_(0., 0.01)
            else:
                init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.GroupNorm, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


@register_model_head("densecl_head")
class DenseCLHead(nn.Module):
    def __init__(
        self,
        model_config: AttrDict,
        dims: List[int],
        num_grid: int = None,
    ):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(dims[0], dims[0]), nn.ReLU(inplace=True),
            nn.Linear(dims[0], dims[1]))

        self.with_pool = num_grid != None
        if self.with_pool:
            self.pool = nn.AdaptiveAvgPool2d((num_grid, num_grid))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], 1), nn.ReLU(inplace=True),
            nn.Conv2d(dims[0], dims[1], 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, batch: torch.Tensor):
        if isinstance(batch, list):
            assert (
                len(batch) == 1
            ), "MLP input should be either a tensor (4D) or list containing 1 tensor."
            batch = batch[0]

        avgpooled_x = self.avgpool(batch)
        avgpooled_x = self.mlp(avgpooled_x.view(avgpooled_x.size(0), -1))

        if self.with_pool:
            batch = self.pool(batch) # sxs
            identity = batch
        else:
            identity = batch
        batch = self.mlp2(batch) # sxs: bxdxsxs
        avgpooled_x2 = self.avgpool2(batch) # 1x1: bxdx1x1
        avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1) # bxd
        return [avgpooled_x, batch, avgpooled_x2, identity]

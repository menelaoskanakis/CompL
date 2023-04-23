import torch.nn as nn


class BCEWithLogitsLossMasked(nn.Module):
    """
    L1 masked loss
    """
    def __init__(self):
        super(BCEWithLogitsLossMasked, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, input, target):
        input = input.squeeze()
        target = target.squeeze()

        mask = (~(target > 1)).float()
        loss = self.criterion(input*mask, target*mask)/(mask.sum() + 1e-12)
        return loss
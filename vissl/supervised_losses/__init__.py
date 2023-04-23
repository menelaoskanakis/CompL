import torch
import numpy as np

from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, L1Loss
from .L1_masked_loss_depth import L1MaskedLossDepth
from .BCE_with_logits_loss_masked import BCEWithLogitsLossMasked
from .binary_cross_entropy_with_logits_weighted import BCEWithLogitsLossWeighted

key2loss = {
    "CrossEntropyLoss": CrossEntropyLoss,
    "BCEWithLogitsLoss": BCEWithLogitsLoss,
    "BCEWithLogitsLossMasked": BCEWithLogitsLossMasked,
    "BCEWithLogitsLossWeighted": BCEWithLogitsLossWeighted,
    "L1Loss": L1Loss,
    "L1MaskedLossDepth": L1MaskedLossDepth,
}


def get_loss_function(name=None):
    """Get loss function

    Args:
        loss (name): Desired loss function to be used
    Returns:
        Loss function
    """
    if name is None:
        raise NotImplementedError("Loss {} has not been defined".format(name))

    else:
        if name not in key2loss:
            raise NotImplementedError("Loss {} not implemented".format(name))

        return key2loss[name]


def get_loss_functions(name, params):
    loss_cl = get_loss_function(name)
    if params is not None:
        loss = loss_cl(**params)
    else:
        loss = loss_cl()

    return loss

from classy_vision.meters import ClassyMeter, register_meter
import numpy as np
from vissl.utils.hydra_config import AttrDict
from vissl.utils.env import get_machine_local_and_dist_rank
import logging
from classy_vision.generic.distributed_util import all_reduce_sum, gather_from_all
import torch


@register_meter("depth")
class Depth(ClassyMeter):
    """
    Add documentation on what this meter does

    Args:
        add documentation about each meter parameter
    """

    def __init__(self, meters_config: AttrDict):
        # implement what the init method should do like
        # setting variable to update etc.
        self.do_rmse = meters_config.get("do_rmse", False)
        self.do_log_rmse = meters_config.get("do_log_rmse", False)
        self._rmses = None
        self._log_rmses = None
        self._rmses_glob = None
        self._log_rmses_glob = None
        self.reset()

    @classmethod
    def from_config(cls, meters_config: AttrDict):
        """
        Get the MyNewMeter instance from the user defined config
        """
        return cls(meters_config)

    @property
    def name(self):
        """
        Name of the meter
        """
        return "depth"

    @property
    def value(self):
        """
        Value of the meter which has been globally synced. This is the value printed and
        recorded by user.
        """
        # implement how the value should be calculated/finalized/returned to user
        _, distributed_rank = get_machine_local_and_dist_rank()

        rmse = torch.mean(self._rmses_glob).item()
        rmse = float("{:.4f}".format(rmse))

        log_rmse = torch.median(self._log_rmses_glob).item()
        log_rmse = float("{:.4f}".format(log_rmse))

        logging.info(
            f"Rank: {distributed_rank} "
            f"rmse: {rmse}"
            f"log_rmse: {log_rmse}"
        )

        return {"rmse": rmse, "log_rmse": log_rmse}

    def sync_state(self):
        """
        Globally syncing the state of each meter across all the trainers.
        Should perform distributed communications like all_gather etc
        to correctly gather the global values to compute the metric

        """
        # implement what Communications should be done to globally sync the state
        if self._rmses is not None and self._log_rmses is not None:
            self._rmses = gather_from_all(self._rmses)
            self._log_rmses = gather_from_all(self._log_rmses)

        # update the meter variables to store these global gathered values
        self._rmses_glob = self._rmses
        self._log_rmses_glob = self._log_rmses

        # Reset values until next sync
        self._rmses = None
        self._log_rmses = None

    def reset(self):
        """
        Reset the meter. Should reset all the meter variables, values.
        """
        self._rmses = None
        self._log_rmses = None
        self._rmses_glob = None
        self._log_rmses_glob = None

    def __repr__(self):
        # implement what information about meter params should be
        # printed by print(meter). This is helpful for debugging
        return repr({"name": self.name, "value": self.value})

    def set_classy_state(self, state):
        """
        Set the state of meter. This is the state loaded from a checkpoint when the model
        is resumed
        """
        # implement how to set the state of the meter
        self.reset()
        self._rmses = state["rmses"].clone()
        self._log_rmses = state["log_rmses"].clone()

    def get_classy_state(self):
        """
        Returns the states of meter that will be checkpointed. This should include
        the variables that are global, updated and affect meter value.
        """
        return {
            "name": self.name,
            "rmses": self._rmses,
            "log_rmses": self._log_rmses,
        }

    def update(self, model_output, target):
        """
        Update the meter every time meter is calculated
        """
        model_output = model_output.squeeze(1)
        assert model_output.shape == target.shape, "Prediction and ground truth dimension missmatch"

        target[target == 0] = 1e-9
        model_output[model_output <= 0] = 1e-9

        valid_mask = (target != 0)
        batch_size = valid_mask.size(0)
        n_valid = torch.sum(valid_mask.view(batch_size, -1), dim=1)

        log_rmse_tmp = (torch.log(target) - torch.log(model_output)) ** 2
        log_rmse_tmp = torch.sqrt(torch.sum((log_rmse_tmp*valid_mask).view(batch_size, -1), dim=1) / n_valid)
        if self._log_rmses is None:
            self._log_rmses = log_rmse_tmp
        else:
            self._log_rmses = torch.cat([self._log_rmses, log_rmse_tmp], 0)

        rmse_tmp = (target - model_output) ** 2
        rmse_tmp = torch.sqrt(torch.sum((rmse_tmp * valid_mask).view(batch_size, -1), dim=1) / n_valid)
        if self._rmses is None:
            self._rmses = rmse_tmp
        else:
            self._rmses = torch.cat([self._rmses, rmse_tmp], 0)

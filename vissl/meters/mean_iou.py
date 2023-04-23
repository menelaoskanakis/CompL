from classy_vision.meters import ClassyMeter, register_meter
from vissl.utils.hydra_config import AttrDict
from vissl.utils.env import get_machine_local_and_dist_rank
import logging
from classy_vision.generic.distributed_util import all_reduce_sum
import torch


@register_meter("mean_iou")
class mIoU(ClassyMeter):

    def __init__(self, meters_config: AttrDict):
        self.n_classes = meters_config.get("n_classes")
        self._total_confusion_matrix = torch.zeros((self.n_classes, self.n_classes))
        self._current_confusion_matrix = torch.zeros((self.n_classes, self.n_classes))
        self.reset()

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = torch.bincount(
            n_class * label_true[mask] + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def nanmean(self, v):
        is_nan = torch.isnan(v)
        v[is_nan] = 0
        return v, v.sum() / (~is_nan).float().sum()

    @classmethod
    def from_config(cls, meters_config: AttrDict):
        return cls(meters_config)

    @property
    def name(self):
        return "mean_iou"

    @property
    def value(self):
        _, distributed_rank = get_machine_local_and_dist_rank()

        hist = self._total_confusion_matrix
        iu = torch.diagonal(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - torch.diagonal(hist))
        iu, mean_iu = self.nanmean(iu)
        mean_iu = mean_iu.item()
        mean_iu = float("{:.4f}".format(mean_iu))
        dict = {"mIoU": mean_iu}
        for ind, val in enumerate(iu):
            dict["IoU_{}".format(ind)] = float("{:.4f}".format(val.item()))
        print('value', get_machine_local_and_dist_rank()[0], mean_iu)
        logging.info(
            f"Rank: {distributed_rank} "
            f"mIoU: {mean_iu}"
        )
        return dict

    def sync_state(self):
        """
        Globally syncing the state of each meter across all the trainers.
        Should perform distributed communications like all_gather etc
        to correctly gather the global values to compute the metric
        """
        # implement what Communications should be done to globally sync the state
        self._current_confusion_matrix = all_reduce_sum(self._current_confusion_matrix)

        # update the meter variables to store these global gathered values
        self._total_confusion_matrix += self._current_confusion_matrix

        # Reset values until next sync
        self._current_confusion_matrix = torch.zeros((self.n_classes, self.n_classes))

    def reset(self):
        """
        Reset the meter. Should reset all the meter variables, values.
        """
        self._current_confusion_matrix = torch.zeros((self.n_classes, self.n_classes))
        self._total_confusion_matrix = torch.zeros((self.n_classes, self.n_classes))

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
        self._total_confusion_matrix = state["total_confusion_matrix"].clone()
        self._current_confusion_matrix = state["current_confusion_matrix"].clone()

    def get_classy_state(self):
        """
        Returns the states of meter that will be checkpointed. This should include
        the variables that are global, updated and affect meter value.
        """
        return {
            "name": self.name,
            "n_classes": self.n_classes,
            "total_confusion_matrix": self._total_confusion_matrix,
            "current_confusion_matrix": self._current_confusion_matrix,
        }

    def update(self, model_output, target):
        """
        Update the meter every time meter is calculated
        """
        model_output = torch.argmax(model_output, dim=1)
        self.validate(model_output, target)
        for lt, lp in zip(target, model_output):
            self._current_confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def validate(self, model_output, target):
        """
        Validate that the input to meter is valid
        """
        # implement how to enforce the validity of the meter inputs
        assert len(model_output.shape) == 3, "model_output should be a 3D tensor"
        assert len(target.shape) == 3, "target should be a 3D tensor"

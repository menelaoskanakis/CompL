# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This is the train step that"s most commonly used in most of the model trainings.
"""

import contextlib
from types import SimpleNamespace

import torch
from classy_vision.generic.distributed_util import all_reduce_mean
from classy_vision.tasks import ClassyTask
from classy_vision.tasks.classification_task import AmpType
from vissl.hooks import SSLClassyHookFunctions
from vissl.trainer.train_steps import register_train_step
from vissl.utils.activation_checkpointing import (
    manual_gradient_all_reduce,
    manual_sync_params,
)
from vissl.utils.misc import is_apex_available
from vissl.utils.perf_stats import PerfTimer

if is_apex_available():
    import apex

# LastBatchInfo will typically hold
# the last samples, target, loss and output.
# More attributes can be added as needed in dependent codeblocks
LastBatchInfo = SimpleNamespace


def construct_sample_for_model(batch_data, task):
    """
    Given the input batch from the dataloader, verify the input is
    as expected: the input data and target data is present in the
    batch.
    In case of multi-input trainings like PIRL, make sure the data
    is in right format i.e. the multiple input should be nested
    under a common key "input".
    """
    sample_key_names = task.data_and_label_keys
    inp_key, target_key = sample_key_names["input"], sample_key_names["target"]
    all_keys = inp_key + target_key

    assert len(inp_key) + len(target_key) <= len(
        batch_data
    ), "Number of input and target keys in batch and train config don't match."

    # every input should be a list. The list corresponds to various data sources
    # and hence could be used to handle several data modalities.
    for key in all_keys:
        assert isinstance(batch_data[key], list), f"key: {key} input is not a list"
        assert (
            len(batch_data[key]) == 1
        ), "Please modify your train step to handle multi-modal input"

    # single input case
    if len(sample_key_names["input"]) == 1 and len(sample_key_names["target"]) == 1:
        sample = {
            "input": batch_data[inp_key[0]][0],
            "target": batch_data[target_key[0]][0],
            "data_valid": batch_data["data_valid"][0],
        }

    # multi-input case (example in PIRL, we pass image and patches both).
    # we nest all these under the sample["input"]
    elif len(sample_key_names["input"]) > 1:
        sample = {"input": {}, "target": {}, "data_valid": None}
        for key in inp_key:
            sample["input"][key] = batch_data[key][0]

        if len(target_key) > 1:
            for key in target_key:
                sample["target"][key] = batch_data[key][0]
        else:
            sample["target"] = batch_data[target_key[0]][0]
        sample["data_valid"] = batch_data["data_valid"][0]

    # copy the other keys as-is, method dependent
    for k in batch_data.keys():
        if k not in all_keys:
            sample[k] = batch_data[k]

    return sample


@register_train_step("semseg_moco_train_step")
def semseg_moco_train_step(task):
    """
    Single training iteration loop of the model.

    Performs: data read, forward, loss computation, backward, optimizer step, parameter updates.

    Various intermediate steps are also performed:
    - logging the training loss, training eta, LR, etc to loggers
    - logging to tensorboard,
    - performing any self-supervised method specific operations (like in MoCo approach, the
    momentum encoder is updated), computing the scores in swav
    - checkpointing model if user wants to checkpoint in the middle
    of an epoch
    """
    task.num_epochs = task.num_train_phases
    assert isinstance(task, ClassyTask), "task is not instance of ClassyTask"

    # reset the last batch info at every step
    task.last_batch = LastBatchInfo()

    # We'll time train_step and some of its sections, and accumulate values
    # into perf_stats if it were defined in local_variables:
    perf_stats = task.perf_stats
    timer_train_step = PerfTimer("train_step_total", perf_stats)
    timer_train_step.start()

    # Process next sample
    with PerfTimer("read_sample", perf_stats):
        sample = next(task.data_iterator)

    if len(sample['data']) > 1:
        raise NotImplementedError('Currently only support 1 dataset')
    else:
        batch_size = sample['data'][0].size()[0]

    if task.train:
        # Pull a batch from the self-supervised branch and concatenate with the supervised one
        ss_sample = task.selfsuper_dataloaders.pull_batch()
        ss_sample['semseg'] = sample['semseg']
        for i in range(len(sample['data'])):
            ss_sample['data'][i] = torch.cat((sample['data'][i], ss_sample['data'][i]), 0)
        sample = ss_sample

    sample = construct_sample_for_model(sample, task)
    if len(sample['semseg']) > 1:
        raise NotImplementedError('Currently only support semseg for 1 dataset')
    else:
        sample['semseg'] = sample['semseg'][0]

    # Only need gradients during training
    grad_context = torch.enable_grad() if task.train else torch.no_grad()
    ddp_context = (
        task.model.no_sync()
        if task.enable_manual_gradient_reduction
        else contextlib.suppress()
    )
    torch_amp_context = (
        torch.cuda.amp.autocast()
        if task.amp_type == AmpType.PYTORCH
        else contextlib.suppress()
    )
    with grad_context, ddp_context, torch_amp_context:
        # Forward pass of the model
        with PerfTimer("forward", perf_stats):
            if task.enable_manual_gradient_reduction:
                # Manually sync params and buffers for DDP.
                manual_sync_params(task.model)
            model_output = task.model(sample["input"])
        # Confirm there are two outputs (since we are doing both semseg and rotations)
        if len(model_output) != 2:
            raise ValueError('This training step requires two outputs')

        task.last_batch.sample = sample
        task.last_batch.model_output = model_output

        if "TASKS" not in task.config:
            raise ValueError('Need to define which tasks are addressed')

        assert len(task.config["TASKS"]) == 2
        assert task.config["TASKS"][1] == "self_supervised"
        # Ensure the index of supervised/self-supervised
        supervised_ind = 0
        self_supervised_ind = 1

        # Run hooks on forward pass
        task.run_hooks(SSLClassyHookFunctions.on_forward.name)

        # Compute loss
        with PerfTimer("loss_compute", perf_stats):
            if task.train:
                supervised_local_loss = task.supervised_loss(model_output[supervised_ind][:batch_size], sample["semseg"])
                local_loss = task.loss(model_output[self_supervised_ind][batch_size:], sample["target"])
            else:
                supervised_local_loss = task.supervised_loss(model_output[supervised_ind], sample["semseg"])
                local_loss = torch.zeros_like(supervised_local_loss)

        # Reduce the loss value across all nodes and gpus.
        with PerfTimer("loss_all_reduce", perf_stats):
            supervised_loss = supervised_local_loss.detach().clone()
            loss = local_loss.detach().clone()
            task.last_batch.loss = [all_reduce_mean(supervised_loss), all_reduce_mean(loss)]

        list_append = []
        for last_batch_element in task.last_batch.loss:
            list_append.append(last_batch_element.data.cpu().item() * sample["semseg"].size(0))

        task.losses.append(list_append)

        # Update supervised meters
        if len(task.supervised_meters) > 0 and (
                (task.train and task.config["SUPERVISED_METERS"]["enable_training_meter"])
                or (not task.train)
        ):
            with PerfTimer("meters_update", perf_stats):
                if isinstance(model_output[supervised_ind], list):
                    if task.train:
                        model_output_cpu = [x.cpu() for x in model_output[supervised_ind][:batch_size]]
                    else:
                        model_output_cpu = [x.cpu() for x in model_output[supervised_ind]]
                else:
                    if task.train:
                        model_output_cpu = model_output[supervised_ind][:batch_size].cpu()
                    else:
                        model_output_cpu = model_output[supervised_ind].cpu()

                for meter in task.supervised_meters:
                    meter.update(model_output_cpu, sample["semseg"].detach().cpu())

        task.last_batch.model_output = model_output
        task.last_batch.target = sample["target"]
        task.last_batch.supervised_target = sample["semseg"]

        # Update the iteration number, check loss is not NaN and measure batch time
        # now if it's a test phase since test phase doesn't have update step.
        task.run_hooks(SSLClassyHookFunctions.on_loss_and_meter.name)

    # Run backward now and update the optimizer
    if task.train:
        merged_loss = task.config["TASKS_WEIGHTS"][supervised_ind] * supervised_local_loss + \
                      task.config["TASKS_WEIGHTS"][self_supervised_ind] * local_loss

        with PerfTimer("backward", perf_stats):

            task.optimizer.zero_grad()
            if task.amp_type == AmpType.APEX:
                with apex.amp.scale_loss(
                        merged_loss, task.optimizer.optimizer
                ) as scaled_loss:
                    scaled_loss.backward()
                    if task.enable_manual_gradient_reduction:
                        manual_gradient_all_reduce(task.model)

            elif task.amp_type == AmpType.PYTORCH:
                task.amp_grad_scaler.scale(merged_loss).backward()
                if task.enable_manual_gradient_reduction:
                    manual_gradient_all_reduce(task.model)
            else:
                merged_loss.backward()
                if task.enable_manual_gradient_reduction:
                    manual_gradient_all_reduce(task.model)

        task.run_hooks(SSLClassyHookFunctions.on_backward.name)

        # Stepping the optimizer also updates learning rate, momentum etc
        # according to the schedulers (if any).
        with PerfTimer("optimizer_step", perf_stats):
            assert task.where < 1.0, (
                "Optimizer being called with where=1.0. This should not happen "
                "as where=1.0 means training is already finished. Please debug your "
                "training setup. A common issue is the data sampler resuming "
                "where you are checkpointing model at every iterations but not using "
                "the stateful data sampler OR there's an issue in properly resuming the "
                "data sampler."
            )
            if task.amp_type == AmpType.PYTORCH:
                task.amp_grad_scaler.step(task.optimizer, where=task.where)
                task.amp_grad_scaler.update()
            else:
                task.optimizer.step(where=task.where)
        task.run_hooks(SSLClassyHookFunctions.on_update.name)
        task.num_updates += task.get_global_batchsize()

    timer_train_step.stop()
    timer_train_step.record()

    return task

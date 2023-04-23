# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging
import pprint
from collections import namedtuple

import torch
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from vissl.utils.misc import concat_all_gather


_DenseCLLossConfig = namedtuple(
    "_DenseCLLossConfig", ["embedding_dim", "queue_size", "momentum", "temperature", "loss_lambda"]
)

DATASETS = {'augmented_voc': 10582,
            'coco': 92516,
            'bdd': 7000,
            'bsds500': 300,
            'nyud': 795}


class DenseCLLossConfig(_DenseCLLossConfig):
    """ Settings for the DenseCL loss"""

    @staticmethod
    def defaults() -> "DenseCLLossConfig":
        return DenseCLLossConfig(
            embedding_dim=128, queue_size=65536, momentum=0.999, temperature=0.2, loss_lambda=0.5
        )


@register_loss("densecl_loss")
class DenseCLLoss(ClassyLoss):
    def __init__(self, config: DenseCLLossConfig):
        super().__init__()
        self.loss_config = config

        # Create the queue
        dataset_name = self.loss_config.get("dataset_name", None)
        if dataset_name is not None:
            if dataset_name not in DATASETS:
                raise ValueError('Currently do not support a dataset with name {}'.format(dataset_name))
            self.loss_config.queue_size = DATASETS[dataset_name]
            self.dataset_queue = True
        else:
            self.dataset_queue = False

        self.register_buffer(
            "queue",
            torch.randn(self.loss_config.embedding_dim, self.loss_config.queue_size),
        )
        self.register_buffer(
            "queue2",
            torch.randn(self.loss_config.embedding_dim, self.loss_config.queue_size),
        )

        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_ptr2", torch.zeros(1, dtype=torch.long))

        self.criterion = nn.CrossEntropyLoss()
        self.initialized = False

        self.key = None
        self.key2 = None

        self.sample = None
        self.densecl_encoder = None
        self.key_ids = None
        self.loss_lambda = self.loss_config.loss_lambda

        self.checkpoint = None
        self.update_bank = True

    @classmethod
    def from_config(cls, config: DenseCLLossConfig):
        return cls(config)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, key: torch.Tensor):
        keys = concat_all_gather(key)
        batch_size = keys.shape[0]

        assert self.loss_config.queue_size % batch_size == 0, (
            f"The queue size needs to be a multiple of the batch size. "
            f"Effective batch size: {batch_size}. Queue size:"
            f" {self.loss_config.queue_size}."
        )

        ptr = int(self.queue_ptr)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (
            ptr + batch_size
        ) % self.loss_config.queue_size  # move pointer, round robin

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, key: torch.Tensor):
        keys = concat_all_gather(key)
        batch_size = keys.shape[0]

        assert self.loss_config.queue_size % batch_size == 0, (
            f"The queue size needs to be a multiple of the batch size. "
            f"Effective batch size: {batch_size}. Queue size:"
            f" {self.loss_config.queue_size}."
        )

        ptr = int(self.queue_ptr2)
        self.queue2[:, ptr : ptr + batch_size] = keys.T
        ptr = (
            ptr + batch_size
        ) % self.loss_config.queue_size  # move pointer, round robin

        self.queue_ptr2[0] = ptr

    @torch.no_grad()
    def _update_queue_on_ids(self, key: torch.Tensor, id: torch.Tensor):
        keys = concat_all_gather(key)
        ids = concat_all_gather(id)
        batch_size = keys.shape[0]

        self.queue[:, ids] = keys.T

    @torch.no_grad()
    def _update_queue2_on_ids(self, key: torch.Tensor, id: torch.Tensor):
        keys = concat_all_gather(key)
        ids = concat_all_gather(id)
        batch_size = keys.shape[0]

        self.queue2[:, ids] = keys.T

    def forward(self, query: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if not self.initialized:
            # Send the queues to the same device as the query (using index 0 as a dummy value)
            self.queue = self.queue.to(query[0].device)
            self.queue2 = self.queue2.to(query[0].device)
            self.initialized = True

        # --
        # Normalize the encoder raw outputs
        q = nn.functional.normalize(query[0], dim=1)
        q_grid = nn.functional.normalize(query[1], dim=1)
        q_identity = nn.functional.normalize(query[3], dim=1)

        k, k_grid, k2, k_identity = self.key

        q_identity = q_identity.reshape(q_identity.size(0), q_identity.size(1), -1)
        k_identity = k_identity.reshape(k_identity.size(0), k_identity.size(1), -1)
        q_grid = q_grid.reshape(q_grid.size(0), q_grid.size(1), -1) # bxdxs^2
        k_grid = k_grid.reshape(k_grid.size(0), k_grid.size(1), -1) # bxdxs^2

        """
        Global Contrastive
        """
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        if self.dataset_queue:
            # Generate masks for negatives
            mask = torch.zeros(l_neg.size(0) * l_neg.size(1), dtype=torch.bool).to(q.device)  # or dtype=torch.ByteTensor
            indexes = torch.arange(self.key_ids.size(0)).to(q.device) * l_neg.size(1) + self.key_ids
            mask[indexes] = True
            mask = mask.reshape(l_neg.size())
            l_neg[mask] = float('-inf')

        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.loss_config.temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

        if self.update_bank:
            if self.dataset_queue:
                self._update_queue_on_ids(k, self.key_ids)
            else:
                self._dequeue_and_enqueue(k)

        loss_single = self.criterion(logits, labels)

        """
        Local Contrastive
        """
        backbone_sim_matrix = torch.matmul(q_identity.permute(0, 2, 1), k_identity)
        densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1]  # NxS^2
        indexed_k_grid = torch.gather(k_grid, 2,
                                      densecl_sim_ind.unsqueeze(1).expand(-1, k_grid.size(1), -1))  # NxCxS^2
        densecl_sim_q = (q_grid * indexed_k_grid).sum(1)  # NxS^2

        l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1)  # NS^2X1

        q_grid = q_grid.permute(0, 2, 1)
        q_grid = q_grid.reshape(-1, q_grid.size(2))
        l_neg_dense = torch.einsum('nc,ck->nk', [q_grid,
                                                 self.queue2.clone().detach()])

        if self.dataset_queue:
            mask_dense = []
            for individ_mask in torch.split(mask, 1):
                mask_dense.append(individ_mask.repeat(densecl_sim_q.size(1), 1).unsqueeze(0))
            mask_dense = torch.cat(mask_dense, 0).reshape(-1, mask.size(1))
            l_neg_dense[mask_dense] = float('-inf')

        logits_dense = torch.cat([l_pos_dense, l_neg_dense], dim=1)

        logits_dense /= self.loss_config.temperature

        labels_dense = torch.zeros(logits_dense.shape[0], dtype=torch.long).to(q.device)

        if self.update_bank:
            if self.dataset_queue:
                self._update_queue2_on_ids(k2, self.key_ids)
            else:
                self._dequeue_and_enqueue2(k2)

        loss_dense = self.criterion(logits_dense, labels_dense)

        return loss_single * (1 - self.loss_lambda) + loss_dense * self.loss_lambda

    def __repr__(self):
        repr_dict = {"name": self._get_name()}
        return pprint.pformat(repr_dict, indent=2)

    def load_state_dict(self, state_dict, *args, **kwargs):
        # If the encoder has been allocated, use the normal pytorch restoration
        if self.densecl_encoder is None:
            self.checkpoint = state_dict
            logging.info("Storing the checkpoint for later use")
        else:
            logging.info("Restoring checkpoint")
            super().load_state_dict(state_dict, *args, **kwargs)

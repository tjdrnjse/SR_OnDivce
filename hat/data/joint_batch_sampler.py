"""
Joint Batch infrastructure for mixing two heterogeneous training data streams.

Provides three building blocks used by hat/train.py when
``joint_batch_training: true`` is set in the YAML:

  StreamTaggedDataset  – transparent wrapper that adds a ``stream_id`` int
                         (0 = Stream A, 1 = Stream B) to every sample dict.

  JointBatchSampler    – BatchSampler that guarantees exactly
                         ``batch_size // 2`` indices from dataset A and
                         ``batch_size - batch_size // 2`` indices from
                         dataset B in *every* mini-batch.
                         Supports DDP (each rank gets its own shard).
                         Mirrors the ``set_epoch(epoch)`` API of BasicSR's
                         ``EnlargedSampler`` so the training loop needs no
                         special-casing.

  joint_collate_fn     – collate function for heterogeneous batches.
                         Keys present in all samples are stacked normally.
                         Keys missing in some samples are returned as a
                         plain Python list (with ``None`` for absent slots)
                         so that ``KDSRModel.feed_data`` can detect which
                         stream each sample belongs to.
"""

import math
import random
from itertools import cycle
from typing import Iterator, List

import torch
from torch.utils.data import BatchSampler, Dataset
from torch.utils.data.dataloader import default_collate


# ──────────────────────────────────────────────────────────────────────────────
# 1. StreamTaggedDataset
# ──────────────────────────────────────────────────────────────────────────────

class StreamTaggedDataset(Dataset):
    """Wraps an existing dataset and appends ``stream_id`` to every sample.

    The ``stream_id`` is a plain Python ``int`` (not a tensor) so that
    ``joint_collate_fn`` can detect it and stack it into a 1-D LongTensor.

    Args:
        dataset (Dataset): The underlying dataset to wrap.
        stream_id (int): Identifier for this stream (0 = A, 1 = B).
    """

    def __init__(self, dataset: Dataset, stream_id: int):
        self.dataset = dataset
        self.stream_id = stream_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        sample['stream_id'] = self.stream_id   # plain int; collated to LongTensor
        return sample


# ──────────────────────────────────────────────────────────────────────────────
# 2. JointBatchSampler
# ──────────────────────────────────────────────────────────────────────────────

class JointBatchSampler(BatchSampler):
    """Guarantees an A/B split in every mini-batch.

    Assumes the training dataset is a ``ConcatDataset([dataset_a_tagged,
    dataset_b_tagged])`` so that:

      * indices ``[0, n_a)``       → Stream A  (RealESRGANDataset)
      * indices ``[n_a, n_a+n_b)`` → Stream B  (SingleLRDataset)

    Each yielded batch contains::

        local_a  (n_a_per_batch indices from A)
        + local_b (n_b_per_batch indices from B)

    DDP support: ``world_size`` ranks each receive a different shard of
    indices.  ``rank`` selects the local portion from the globally-drawn pool.

    Args:
        n_a (int): Number of samples in dataset A.
        n_b (int): Number of samples in dataset B.
        batch_size (int): Per-rank batch size (= ``batch_size_per_gpu`` in YAML).
        iters_per_epoch (int): Number of batches to yield per epoch.
        seed (int): Base RNG seed; shuffled by epoch to vary across epochs.
        rank (int): Local DDP rank (0 for single-GPU).
        world_size (int): Total number of DDP ranks (1 for single-GPU).
    """

    def __init__(
        self,
        n_a: int,
        n_b: int,
        batch_size: int,
        iters_per_epoch: int,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ):
        # Bypass BatchSampler.__init__ (we don't use a base sampler)
        self.n_a = n_a
        self.n_b = n_b
        self.batch_size = batch_size
        self.n_a_per_batch = batch_size // 2
        self.n_b_per_batch = batch_size - self.n_a_per_batch
        self.iters_per_epoch = iters_per_epoch
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self._epoch = 0

    # BasicSR training loop calls sampler.set_epoch(epoch) each epoch
    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self.iters_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch * 31337)

        # Globally shuffled index pools (all ranks share the same permutation)
        a_perm: List[int] = torch.randperm(self.n_a, generator=g).tolist()
        b_perm: List[int] = (
            torch.randperm(self.n_b, generator=g) + self.n_a
        ).tolist()

        a_pool = cycle(a_perm)
        b_pool = cycle(b_perm)

        for _ in range(self.iters_per_epoch):
            # Draw enough indices for *all* ranks in one shot
            total_a = self.n_a_per_batch * self.world_size
            total_b = self.n_b_per_batch * self.world_size
            all_a = [next(a_pool) for _ in range(total_a)]
            all_b = [next(b_pool) for _ in range(total_b)]

            # Each rank takes its own slice
            lo_a = self.rank * self.n_a_per_batch
            hi_a = lo_a + self.n_a_per_batch
            lo_b = self.rank * self.n_b_per_batch
            hi_b = lo_b + self.n_b_per_batch

            yield all_a[lo_a:hi_a] + all_b[lo_b:hi_b]


# ──────────────────────────────────────────────────────────────────────────────
# 3. joint_collate_fn
# ──────────────────────────────────────────────────────────────────────────────

def joint_collate_fn(batch: list) -> dict:
    """Collate a mixed batch of dicts with potentially differing key sets.

    Algorithm:
      * Collect the union of all keys across every sample in ``batch``.
      * For a key present in **all** samples: attempt ``default_collate``
        (stacks tensors, handles strings, numbers, etc.).
      * For a key present in **only some** samples: build a plain ``list``
        of length ``len(batch)`` with ``None`` where the key is absent.
        The caller (``KDSRModel.feed_data``) is responsible for interpreting
        these partial lists.

    The ``stream_id`` key (added by ``StreamTaggedDataset``) is always an
    ``int`` in each sample and is therefore stacked into a LongTensor by
    ``default_collate``.
    """
    all_keys: set = set()
    for item in batch:
        all_keys.update(item.keys())

    result: dict = {}
    for key in all_keys:
        values = [item.get(key, None) for item in batch]

        if all(v is not None for v in values):
            # Homogeneous — try the standard collator first
            try:
                result[key] = default_collate(values)
                continue
            except (TypeError, RuntimeError):
                pass

        # Heterogeneous or non-collatable — keep as list
        result[key] = values

    return result

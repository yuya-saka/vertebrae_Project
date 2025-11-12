"""
Balanced Batch Sampler for class-imbalanced datasets.
"""

import torch
from torch.utils.data import Sampler
import numpy as np
from typing import Iterator, List


class BalancedBatchSampler(Sampler):
    """
    Batch sampler that ensures class balance within each batch.

    Each batch will have equal number of samples from each class (fracture : non-fracture = 1:1).
    This helps stabilize training on imbalanced datasets.

    Args:
        labels: List of all sample labels (0 or 1)
        batch_size: Batch size (must be even)
        drop_last: Whether to drop the last incomplete batch
    """

    def __init__(
        self,
        labels: List[int],
        batch_size: int,
        drop_last: bool = True
    ):
        if batch_size % 2 != 0:
            raise ValueError(f"batch_size must be even, got {batch_size}")

        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Separate indices by class
        self.positive_indices = np.where(self.labels == 1)[0].tolist()
        self.negative_indices = np.where(self.labels == 0)[0].tolist()

        self.n_positive = len(self.positive_indices)
        self.n_negative = len(self.negative_indices)

        # Samples per class in each batch
        self.samples_per_class = batch_size // 2

        # Calculate number of batches per epoch
        self.n_batches = self._calculate_n_batches()

        print(f"BalancedBatchSampler initialized:")
        print(f"  Positive samples: {self.n_positive}")
        print(f"  Negative samples: {self.n_negative}")
        print(f"  Batch size: {batch_size} ({self.samples_per_class} pos + {self.samples_per_class} neg)")
        print(f"  Batches per epoch: {self.n_batches}")

    def _calculate_n_batches(self) -> int:
        """Calculate number of batches per epoch."""
        n_batches_positive = self.n_positive // self.samples_per_class
        n_batches_negative = self.n_negative // self.samples_per_class

        # Use the minimum to ensure balanced batches
        n_batches = min(n_batches_positive, n_batches_negative)

        return n_batches

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches with class balance."""
        # Shuffle indices at the start of each epoch
        np.random.shuffle(self.positive_indices)
        np.random.shuffle(self.negative_indices)

        # Generate batches
        for batch_idx in range(self.n_batches):
            # Get samples from each class
            pos_start = batch_idx * self.samples_per_class
            pos_end = pos_start + self.samples_per_class

            neg_start = batch_idx * self.samples_per_class
            neg_end = neg_start + self.samples_per_class

            batch_positive = self.positive_indices[pos_start:pos_end]
            batch_negative = self.negative_indices[neg_start:neg_end]

            # Combine and shuffle
            batch = batch_positive + batch_negative
            np.random.shuffle(batch)

            yield batch

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return self.n_batches

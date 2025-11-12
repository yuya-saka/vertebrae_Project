"""
Common utility functions for reproducibility and data splitting.
"""

import random
import numpy as np
import torch
from typing import List, Tuple


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_patients(
    patient_ids: List[int],
    n_folds: int,
    fold_id: int
) -> Tuple[List[int], List[int]]:
    """
    Split patient IDs for K-fold cross-validation.

    Args:
        patient_ids: List of all patient IDs
        n_folds: Number of folds
        fold_id: Current fold index (0-indexed)

    Returns:
        Tuple of (train_patient_ids, val_patient_ids)
    """
    # Use fixed seed for reproducibility
    np.random.seed(42)
    shuffled_ids = np.array(patient_ids)
    np.random.shuffle(shuffled_ids)

    fold_size = len(shuffled_ids) // n_folds
    val_start = fold_id * fold_size
    val_end = val_start + fold_size if fold_id < n_folds - 1 else len(shuffled_ids)

    val_ids = shuffled_ids[val_start:val_end].tolist()
    train_ids = np.concatenate([
        shuffled_ids[:val_start],
        shuffled_ids[val_end:]
    ]).tolist()

    return train_ids, val_ids

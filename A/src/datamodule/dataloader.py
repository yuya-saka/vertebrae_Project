"""
DataLoader creation functions with patient-level splitting.
Updated for new PNG dataset structure.
"""

from pathlib import Path
from typing import List, Tuple
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from .dataset import MultiTaskDataset
from .sampler import BalancedBatchSampler


def create_dataloaders(
    train_patient_ids: List[int],
    val_patient_ids: List[int],
    cfg: DictConfig
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders with patient-level splitting.

    Args:
        train_patient_ids: List of patient IDs for training (numeric, e.g., [1003, 1015, ...])
        val_patient_ids: List of patient IDs for validation
        cfg: Configuration object

    Returns:
        Tuple of (train_loader, val_loader)
    """
    print(f"\nCreating datasets from CSV:")
    print(f"  CSV file: {cfg.data_direction.csv_file}")
    print(f"  Train patient IDs: {len(train_patient_ids)}")
    print(f"  Val patient IDs: {len(val_patient_ids)}")

    # Create datasets
    train_dataset = MultiTaskDataset(
        csv_file=cfg.data_direction.csv_file,
        project_root=cfg.data_direction.project_root_for_csv,
        patient_ids=train_patient_ids,
        image_size=cfg.data_direction.image_size,
        augmentation=cfg.data_direction.augmentation,
        is_training=True
    )

    val_dataset = MultiTaskDataset(
        csv_file=cfg.data_direction.csv_file,
        project_root=cfg.data_direction.project_root_for_csv,
        patient_ids=val_patient_ids,
        image_size=cfg.data_direction.image_size,
        augmentation=None,  # No augmentation for validation
        is_training=False
    )

    # Create BalancedBatchSampler for training
    train_sampler = BalancedBatchSampler(
        labels=train_dataset.get_labels(),
        batch_size=cfg.training.batch_size,
        drop_last=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,  # Use batch_sampler instead of batch_size/shuffle
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size * 2,  # Larger batch size for validation
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

"""
DataLoader creation functions with patient-level splitting.
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
        train_patient_ids: List of patient IDs for training
        val_patient_ids: List of patient IDs for validation
        cfg: Configuration object

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Get all CSV files in the image base directory
    image_base_path = Path(cfg.data_direction.image_base_dir)
    all_csv_files = list(image_base_path.glob("inp*/fracture_labels_inp*.csv"))

    # Filter CSV files by patient IDs
    train_csv_files = [
        str(f) for f in all_csv_files
        if int(f.parent.name[3:]) in train_patient_ids
    ]
    val_csv_files = [
        str(f) for f in all_csv_files
        if int(f.parent.name[3:]) in val_patient_ids
    ]

    print(f"\nCreating datasets:")
    print(f"  Train CSV files: {len(train_csv_files)}")
    print(f"  Val CSV files: {len(val_csv_files)}")

    # Create datasets
    train_dataset = MultiTaskDataset(
        csv_files=train_csv_files,
        image_base_dir=cfg.data_direction.image_base_dir,
        mask_base_dir=cfg.data_direction.mask_base_dir,
        hu_windows=cfg.data_direction.hu_windows,
        image_size=cfg.data_direction.image_size,
        augmentation=cfg.data_direction.augmentation,
        is_training=True
    )

    val_dataset = MultiTaskDataset(
        csv_files=val_csv_files,
        image_base_dir=cfg.data_direction.image_base_dir,
        mask_base_dir=cfg.data_direction.mask_base_dir,
        hu_windows=cfg.data_direction.hu_windows,
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

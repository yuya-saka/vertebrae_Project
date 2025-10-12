"""
DataLoader utilities for Vertebral Fracture Segmentation
"""

import glob
from typing import Dict, List, Tuple, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import (
    VertebralFractureDataset,
    get_patient_ids_from_csv,
    split_patients_for_fold,
    filter_csv_by_patients,
)


class VertebralFractureDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for vertebral fracture segmentation.

    Args:
        data_dir: Base directory containing train/test data
        hu_windows: HU window configurations
        image_size: Target image size (H, W)
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        n_folds: Total number of folds for cross-validation
        fold_id: Current fold ID (0-indexed)
        augmentation: Augmentation configuration
        oversample_fracture: Whether to oversample fracture slices
        oversample_factor: Factor for oversampling
    """

    def __init__(
        self,
        data_dir: str,
        hu_windows: Dict,
        image_size: Tuple[int, int] = (256, 256),
        batch_size: int = 16,
        num_workers: int = 4,
        n_folds: int = 5,
        fold_id: int = 0,
        augmentation: Optional[Dict] = None,
        oversample_fracture: bool = False,
        oversample_factor: int = 1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.hu_windows = hu_windows
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_folds = n_folds
        self.fold_id = fold_id
        self.augmentation = augmentation
        self.oversample_fracture = oversample_fracture
        self.oversample_factor = oversample_factor

        # These will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation/testing."""

        # Get all CSV files for training data
        train_csv_pattern = f"{self.data_dir}/slice_train/axial_mask/*/mask_labels_*.csv"
        train_csv_files = sorted(glob.glob(train_csv_pattern))

        if not train_csv_files:
            raise ValueError(f"No training CSV files found with pattern: {train_csv_pattern}")

        print(f"Found {len(train_csv_files)} training CSV files")

        # Get all patient IDs
        patient_ids = get_patient_ids_from_csv(train_csv_files)
        print(f"Total patients: {len(patient_ids)}")

        # Split into train/val based on fold
        train_patient_ids, val_patient_ids = split_patients_for_fold(
            patient_ids, self.n_folds, self.fold_id
        )
        print(f"Fold {self.fold_id}: Train patients: {len(train_patient_ids)}, Val patients: {len(val_patient_ids)}")

        # Filter CSV files by patient IDs
        train_csv_filtered = filter_csv_by_patients(train_csv_files, train_patient_ids)
        val_csv_filtered = filter_csv_by_patients(train_csv_files, val_patient_ids)

        # Training dataset
        if stage == "fit" or stage is None:
            self.train_dataset = VertebralFractureDataset(
                csv_files=train_csv_filtered,
                image_base_dir=f"{self.data_dir}/slice_train/axial",
                mask_base_dir=f"{self.data_dir}/slice_train/axial_mask",
                hu_windows=self.hu_windows,
                image_size=self.image_size,
                augmentation=self.augmentation,
                is_training=True,
                oversample_fracture=self.oversample_fracture,
                oversample_factor=self.oversample_factor,
            )

            self.val_dataset = VertebralFractureDataset(
                csv_files=val_csv_filtered,
                image_base_dir=f"{self.data_dir}/slice_train/axial",
                mask_base_dir=f"{self.data_dir}/slice_train/axial_mask",
                hu_windows=self.hu_windows,
                image_size=self.image_size,
                augmentation=None,
                is_training=False,
                oversample_fracture=False,
                oversample_factor=1,
            )

        # Test dataset
        if stage == "test" or stage is None:
            test_csv_pattern = f"{self.data_dir}/slice_test/axial_mask/*/mask_labels_*.csv"
            test_csv_files = sorted(glob.glob(test_csv_pattern))

            if test_csv_files:
                print(f"Found {len(test_csv_files)} test CSV files")
                self.test_dataset = VertebralFractureDataset(
                    csv_files=test_csv_files,
                    image_base_dir=f"{self.data_dir}/slice_test/axial",
                    mask_base_dir=f"{self.data_dir}/slice_test/axial_mask",
                    hu_windows=self.hu_windows,
                    image_size=self.image_size,
                    augmentation=None,
                    is_training=False,
                    oversample_fracture=False,
                    oversample_factor=1,
                )
            else:
                print(f"Warning: No test CSV files found with pattern: {test_csv_pattern}")

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        if self.test_dataset is None:
            raise ValueError("Test dataset not initialized!")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

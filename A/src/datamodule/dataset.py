"""
Multi-Task Dataset for CT images with segmentation masks.
Updated for PNG image dataset (normalized, already preprocessed).
"""

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class MultiTaskDataset(Dataset):
    """
    Dataset for multi-task learning (classification + segmentation).

    Loads PNG images and corresponding segmentation masks with fracture labels.
    Applies resizing and online augmentation.

    NOTE: This version is for PNG datasets that are already normalized.
    HU windowing is NOT applied.
    """

    def __init__(
        self,
        csv_file: str,
        project_root: str,
        patient_ids: List[int],
        image_size: Tuple[int, int] = (256, 256),
        augmentation: Optional[Dict] = None,
        is_training: bool = True,
    ):
        """
        Args:
            csv_file: Path to CSV file containing image/mask paths and labels
            project_root: Project root directory (CSV paths are relative to this)
            patient_ids: List of patient IDs to include (e.g., [1003, 1015, ...])
            image_size: Target image size (H, W)
            augmentation: Augmentation parameters (rotation_degrees, etc.)
            is_training: If True, apply online augmentation
        """
        self.project_root = Path(project_root)
        self.image_size = image_size
        self.augmentation = augmentation
        self.is_training = is_training

        # Load CSV file
        self.data = self._load_and_filter_csv(csv_file, patient_ids)

        print(f"Dataset initialized with {len(self.data)} samples")
        if 'has_fracture' in self.data.columns:
            fracture_count = (self.data['has_fracture'] == 1).sum()
            print(f"Fracture slices: {fracture_count} ({fracture_count/len(self.data)*100:.2f}%)")

    def _load_and_filter_csv(self, csv_file: str, patient_ids: List[int]) -> pd.DataFrame:
        """
        Load CSV file and filter by patient IDs.

        CSV format:
            image_path, mask_path, patient_id, vertebra_id, orientation, has_fracture

        patient_id format: "AI1003" (string)
        """
        df = pd.read_csv(csv_file)

        # Convert numeric patient IDs to string format "AI{id}"
        patient_id_strs = [f"AI{pid}" for pid in patient_ids]

        # Filter by patient IDs
        df_filtered = df[df['patient_id'].isin(patient_id_strs)].copy()

        print(f"Loaded CSV: {csv_file}")
        print(f"  Total samples in CSV: {len(df)}")
        print(f"  Filtered samples (patient_ids={len(patient_ids)}): {len(df_filtered)}")

        if len(df_filtered) == 0:
            raise ValueError(f"No samples found for patient IDs: {patient_id_strs[:5]}...")

        return df_filtered.reset_index(drop=True)

    def _apply_augmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply online data augmentation to image and mask simultaneously.

        Args:
            image: Input image (H, W, 3) - RGB
            mask: Input mask (H, W)

        Returns:
            Augmented image and mask
        """
        if self.augmentation is None or not self.is_training:
            return image, mask

        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Rotation
        if 'rotation_degrees' in self.augmentation and np.random.rand() < 0.5:
            angle = np.random.uniform(
                -self.augmentation['rotation_degrees'],
                self.augmentation['rotation_degrees']
            )
            M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M_rot, (w, h), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask, M_rot, (w, h), flags=cv2.INTER_NEAREST)

        # Translation
        if 'translation_pixels' in self.augmentation and np.random.rand() < 0.5:
            tx = np.random.uniform(
                -self.augmentation['translation_pixels'],
                self.augmentation['translation_pixels']
            )
            ty = np.random.uniform(
                -self.augmentation['translation_pixels'],
                self.augmentation['translation_pixels']
            )
            M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M_trans, (w, h), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask, M_trans, (w, h), flags=cv2.INTER_NEAREST)

        # Scale
        if 'scale_range' in self.augmentation and np.random.rand() < 0.5:
            scale = np.random.uniform(
                self.augmentation['scale_range'][0],
                self.augmentation['scale_range'][1]
            )
            M_scale = cv2.getRotationMatrix2D(center, 0, scale)
            image = cv2.warpAffine(image, M_scale, (w, h), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask, M_scale, (w, h), flags=cv2.INTER_NEAREST)

        # Horizontal flip
        if 'horizontal_flip_prob' in self.augmentation:
            if np.random.rand() < self.augmentation['horizontal_flip_prob']:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)

        # Contrast adjustment
        if 'contrast_range' in self.augmentation and np.random.rand() < 0.5:
            alpha = np.random.uniform(
                self.augmentation['contrast_range'][0],
                self.augmentation['contrast_range'][1]
            )
            image = np.clip(image * alpha, 0, 255).astype(np.uint8)

        return image, mask

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dict with keys: 'image', 'mask', 'label_class', 'metadata'
        """
        row = self.data.iloc[idx]

        # Load PNG image (RGB, 8-bit)
        image_path = self.project_root / row['image_path']
        image = cv2.imread(str(image_path))  # (H, W, 3) BGR
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Load mask (grayscale)
        mask_path = self.project_root / row['mask_path']
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # (H, W)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Get label
        label_class = float(row['has_fracture'])

        # Resize
        image_resized = cv2.resize(
            image,
            (self.image_size[1], self.image_size[0]),
            interpolation=cv2.INTER_LINEAR
        )
        mask_resized = cv2.resize(
            mask,
            (self.image_size[1], self.image_size[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # Apply augmentation (online)
        image_aug, mask_aug = self._apply_augmentation(image_resized, mask_resized)

        # Normalize image to [0, 1]
        image_normalized = image_aug.astype(np.float32) / 255.0

        # Convert to tensors
        # Image: (H, W, 3) -> (3, H, W)
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)  # (3, H, W)

        # Mask: (H, W) -> (1, H, W), normalize to [0, 1]
        mask_tensor = torch.from_numpy(mask_aug.astype(np.float32) / 255.0).unsqueeze(0)

        label_tensor = torch.tensor(label_class, dtype=torch.float32)

        # Metadata
        metadata = {
            'patient_id': str(row['patient_id']),
            'vertebra_id': str(row['vertebra_id']),
            'orientation': str(row['orientation'])
        }

        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'label_class': label_tensor,
            'metadata': metadata
        }

    def get_labels(self) -> List[int]:
        """
        Get all labels for BalancedBatchSampler.

        Returns:
            List of labels (0 or 1)
        """
        return self.data['has_fracture'].tolist()

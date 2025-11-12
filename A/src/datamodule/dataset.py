"""
Multi-Task Dataset for CT images with segmentation masks.
"""

import numpy as np
import pandas as pd
import nibabel as nib
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class MultiTaskDataset(Dataset):
    """
    Dataset for multi-task learning (classification + segmentation).

    Loads CT images and corresponding segmentation masks with fracture labels.
    Applies 3-channel HU windowing, resizing, and online augmentation.
    """

    def __init__(
        self,
        csv_files: List[str],
        image_base_dir: str,
        mask_base_dir: str,
        hu_windows: Dict,
        image_size: Tuple[int, int] = (256, 256),
        augmentation: Optional[Dict] = None,
        is_training: bool = True,
    ):
        """
        Args:
            csv_files: List of CSV file paths containing labels
            image_base_dir: Base directory for CT images
            mask_base_dir: Base directory for segmentation masks
            hu_windows: Dict with channel_1, channel_2, channel_3 HU window settings
            image_size: Target image size (H, W)
            augmentation: Augmentation parameters (rotation_degrees, etc.)
            is_training: If True, apply online augmentation
        """
        self.image_base_dir = Path(image_base_dir)
        self.mask_base_dir = Path(mask_base_dir)
        self.hu_windows = hu_windows
        self.image_size = image_size
        self.augmentation = augmentation
        self.is_training = is_training

        # Load CSV files
        self.data = self._load_csv_files(csv_files)

        # Construct mask paths
        self.data['MaskPath'] = self.data.apply(
            lambda row: self._construct_mask_path(row),
            axis=1
        )

        print(f"Dataset initialized with {len(self.data)} samples")
        fracture_count = (self.data['Fracture_Label'] == 1).sum()
        print(f"Fracture slices: {fracture_count} ({fracture_count/len(self.data)*100:.2f}%)")

    def _load_csv_files(self, csv_files: List[str]) -> pd.DataFrame:
        """Load and concatenate multiple CSV files."""
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dfs.append(df)

        if len(dfs) == 0:
            raise ValueError("No CSV files provided")

        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df

    def _construct_mask_path(self, row: pd.Series) -> str:
        """
        Construct mask path from CT image path.

        CT: .../axial/inp1003/27/slice_000.nii
        -> Mask: .../axial_mask/inp1003/27/mask_000.nii
        """
        case_id = f"inp{row['Case']}"
        vertebra = str(row['Vertebra'])
        slice_idx = row['SliceIndex']

        mask_path = self.mask_base_dir / case_id / vertebra / f"mask_{slice_idx:03d}.nii"
        return str(mask_path)

    def _normalize_hu_window(
        self,
        image: np.ndarray,
        window_min: float,
        window_max: float
    ) -> np.ndarray:
        """
        Apply HU windowing and normalize to [0, 1].

        Args:
            image: Input HU image
            window_min: Minimum HU value
            window_max: Maximum HU value

        Returns:
            Normalized image [0, 1]
        """
        image_clipped = np.clip(image, window_min, window_max)
        image_normalized = (image_clipped - window_min) / (window_max - window_min + 1e-8)
        return image_normalized.astype(np.float32)

    def _create_3channel_input(self, image: np.ndarray) -> np.ndarray:
        """
        Create 3-channel input with different HU windows.

        Args:
            image: Input HU image (H, W)

        Returns:
            3-channel image (3, H, W)
        """
        ch1 = self._normalize_hu_window(
            image.copy(),
            self.hu_windows['channel_1']['min'],
            self.hu_windows['channel_1']['max']
        )
        ch2 = self._normalize_hu_window(
            image.copy(),
            self.hu_windows['channel_2']['min'],
            self.hu_windows['channel_2']['max']
        )
        ch3 = self._normalize_hu_window(
            image.copy(),
            self.hu_windows['channel_3']['min'],
            self.hu_windows['channel_3']['max']
        )

        return np.stack([ch1, ch2, ch3], axis=0)  # (3, H, W)

    def _apply_augmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply online data augmentation to image and mask simultaneously.

        Args:
            image: Input image (3, H, W)
            mask: Input mask (H, W)

        Returns:
            Augmented image and mask
        """
        if self.augmentation is None or not self.is_training:
            return image, mask

        # Transpose for cv2: (3, H, W) -> (H, W, 3)
        image = np.transpose(image, (1, 2, 0))

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
            image = np.clip(image * alpha, 0, 1)

        # Transpose back: (H, W, 3) -> (3, H, W)
        image = np.transpose(image, (2, 0, 1))

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

        # Load CT image
        image_path = row['FullPath']
        image_nii = nib.load(image_path)
        image = image_nii.get_fdata().astype(np.float32)  # (H, W)

        # Load mask
        mask_path = row['MaskPath']
        mask_nii = nib.load(mask_path)
        mask = mask_nii.get_fdata().astype(np.float32)  # (H, W)

        # Get label
        label_class = float(row['Fracture_Label'])

        # Apply HU windowing (3 channels)
        image_3ch = self._create_3channel_input(image)  # (3, H, W)

        # Resize
        image_resized = np.zeros((3, self.image_size[0], self.image_size[1]), dtype=np.float32)
        for c in range(3):
            image_resized[c] = cv2.resize(
                image_3ch[c],
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

        # Convert to tensors
        image_tensor = torch.from_numpy(image_aug)  # (3, H, W)
        mask_tensor = torch.from_numpy(mask_aug).unsqueeze(0)  # (1, H, W)
        label_tensor = torch.tensor(label_class, dtype=torch.float32)

        # Metadata
        metadata = {
            'case': int(row['Case']),
            'vertebra': str(row['Vertebra']),
            'slice_index': int(row['SliceIndex'])
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
        return self.data['Fracture_Label'].tolist()

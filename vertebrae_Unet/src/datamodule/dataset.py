"""
Dataset class for Vertebral Fracture Segmentation
Handles 3-channel HU window input and online data augmentation
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import nibabel as nib
from pathlib import Path


class VertebralFractureDataset(Dataset):
    """
    Dataset for vertebral fracture segmentation with 3-channel HU window input.

    Args:
        csv_files: List of CSV file paths containing image metadata
        image_base_dir: Base directory for axial slice images
        mask_base_dir: Base directory for mask images
        hu_windows: Dict containing HU window configurations for 3 channels
        image_size: Tuple of (height, width) for resizing
        augmentation: Optional augmentation configuration
        is_training: Whether this is training mode (enables augmentation)
        oversample_fracture: Whether to oversample fracture slices
        oversample_factor: Factor for oversampling fracture slices
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
        oversample_fracture: bool = False,
        oversample_factor: int = 1,
    ):
        self.image_base_dir = Path(image_base_dir)
        self.mask_base_dir = Path(mask_base_dir)
        self.hu_windows = hu_windows
        self.image_size = image_size
        self.augmentation = augmentation if is_training else None
        self.is_training = is_training

        # Load all CSV files
        self.data = self._load_csv_files(csv_files)

        # Oversample fracture slices if enabled
        if oversample_fracture and is_training:
            self.data = self._oversample_fracture_slices(oversample_factor)

        print(f"Dataset initialized with {len(self.data)} samples")
        fracture_count = (self.data['Fracture_Label'] == 1).sum()
        print(f"Fracture slices: {fracture_count} ({fracture_count/len(self.data)*100:.2f}%)")

    def _load_csv_files(self, csv_files: List[str]) -> pd.DataFrame:
        """Load and concatenate multiple CSV files."""
        dfs = []
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                dfs.append(df)
            else:
                print(f"Warning: CSV file not found: {csv_file}")

        if not dfs:
            raise ValueError("No valid CSV files found!")

        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df

    def _oversample_fracture_slices(self, factor: int) -> pd.DataFrame:
        """Oversample fracture slices to balance the dataset."""
        fracture_df = self.data[self.data['Fracture_Label'] == 1]
        non_fracture_df = self.data[self.data['Fracture_Label'] == 0]

        # Repeat fracture slices
        fracture_oversampled = pd.concat([fracture_df] * factor, ignore_index=True)

        # Combine and shuffle
        combined = pd.concat([non_fracture_df, fracture_oversampled], ignore_index=True)
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

        return combined

    def _get_image_path(self, row: pd.Series) -> Path:
        """Get image path from CSV row - uses FullPath column if available."""
        # Use FullPath from CSV if it exists (recommended)
        if 'FullPath' in row and pd.notna(row['FullPath']):
            return Path(row['FullPath'])

        # Fallback: construct path manually
        case_id = f"inp{row['Case']}"
        vertebra = str(row['Vertebra'])
        slice_idx = f"slice_{row['SliceIndex']:03d}.nii"
        return self.image_base_dir / case_id / vertebra / slice_idx

    def _load_nifti_slice(self, path: Path) -> np.ndarray:
        """Load a single NIfTI slice."""
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        nii = nib.load(str(path))
        data = nii.get_fdata()

        # Ensure 2D
        if data.ndim == 3:
            data = data[:, :, 0]

        return data.astype(np.float32)

    def _normalize_hu_window(self, image: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
        """Normalize image with HU window and clip to [0, 1]."""
        # Clip to HU window
        image = np.clip(image, hu_min, hu_max)
        # Normalize to [0, 1]
        image = (image - hu_min) / (hu_max - hu_min)
        return image.astype(np.float32)

    def _create_3channel_input(self, image: np.ndarray) -> np.ndarray:
        """Create 3-channel input from single HU image using different windows."""
        # Channel 1: [0, 1800]
        ch1 = self._normalize_hu_window(
            image.copy(),
            self.hu_windows['channel_1']['min'],
            self.hu_windows['channel_1']['max']
        )

        # Channel 2: [-200, 300]
        ch2 = self._normalize_hu_window(
            image.copy(),
            self.hu_windows['channel_2']['min'],
            self.hu_windows['channel_2']['max']
        )

        # Channel 3: [200, 1200]
        ch3 = self._normalize_hu_window(
            image.copy(),
            self.hu_windows['channel_3']['min'],
            self.hu_windows['channel_3']['max']
        )

        # Stack channels: (3, H, W)
        return np.stack([ch1, ch2, ch3], axis=0)

    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size."""
        if image.shape[0] == 3:  # (C, H, W)
            channels = []
            for i in range(3):
                resized = cv2.resize(image[i], (target_size[1], target_size[0]),
                                   interpolation=cv2.INTER_LINEAR)
                channels.append(resized)
            return np.stack(channels, axis=0)
        else:  # (H, W)
            return cv2.resize(image, (target_size[1], target_size[0]),
                            interpolation=cv2.INTER_NEAREST)

    def _apply_augmentation(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to image and mask."""
        if self.augmentation is None:
            return image, mask

        # Image: (C, H, W), Mask: (H, W)
        # Move to (H, W, C) for augmentation
        image = np.transpose(image, (1, 2, 0))  # (H, W, C)

        # Random rotation
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-self.augmentation['rotation_degrees'],
                                     self.augmentation['rotation_degrees'])
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Random translation
        if np.random.rand() < 0.5:
            tx = np.random.randint(-self.augmentation['translation_pixels'],
                                  self.augmentation['translation_pixels'] + 1)
            ty = np.random.randint(-self.augmentation['translation_pixels'],
                                  self.augmentation['translation_pixels'] + 1)
            h, w = image.shape[:2]
            matrix = np.float32([[1, 0, tx], [0, 1, ty]])

            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Random scaling
        if np.random.rand() < 0.5:
            scale = np.random.uniform(self.augmentation['scale_range'][0],
                                    self.augmentation['scale_range'][1])
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, 0, scale)

            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Random horizontal flip
        if np.random.rand() < self.augmentation['horizontal_flip_prob']:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # Random brightness (apply to all channels)
        if np.random.rand() < 0.5:
            brightness = np.random.uniform(-0.1, 0.1)
            image = np.clip(image + brightness, 0, 1)

        # Random contrast
        if np.random.rand() < 0.5:
            contrast = np.random.uniform(self.augmentation['contrast_range'][0],
                                       self.augmentation['contrast_range'][1])
            image = np.clip(image * contrast, 0, 1)

        # Move back to (C, H, W)
        image = np.transpose(image, (2, 0, 1))

        return image, mask

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - image: (3, H, W) tensor
                - mask: (1, H, W) tensor
                - label: scalar tensor (0 or 1)
                - metadata: dict with case, vertebra, slice_index
        """
        row = self.data.iloc[idx]

        # Load image
        image_path = self._get_image_path(row)
        image = self._load_nifti_slice(image_path)

        # Create 3-channel input
        image = self._create_3channel_input(image)

        # Load mask
        mask_path = Path(row['MaskPath'])
        mask = self._load_nifti_slice(mask_path)

        # Resize
        image = self._resize_image(image, self.image_size)
        mask = self._resize_image(mask, self.image_size)

        # Binarize mask
        mask = (mask > 0.5).astype(np.float32)

        # Apply augmentation
        if self.is_training:
            image, mask = self._apply_augmentation(image, mask)

        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # (1, H, W)
        label = torch.tensor(row['Fracture_Label'], dtype=torch.long)

        # Metadata
        metadata = {
            'case': row['Case'],
            'vertebra': row['Vertebra'],
            'slice_index': row['SliceIndex'],
        }

        return {
            'image': image,
            'mask': mask,
            'label': label,
            'metadata': metadata,
        }


def get_patient_ids_from_csv(csv_files: List[str]) -> List[int]:
    """Extract unique patient IDs from CSV files."""
    dfs = []
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    patient_ids = combined_df['Case'].unique().tolist()
    return sorted(patient_ids)


def split_patients_for_fold(patient_ids: List[int], n_folds: int, fold_id: int) -> Tuple[List[int], List[int]]:
    """
    Split patients into training and validation sets for a specific fold.

    Args:
        patient_ids: List of patient IDs
        n_folds: Total number of folds
        fold_id: Current fold ID (0-indexed)

    Returns:
        Tuple of (train_patient_ids, val_patient_ids)
    """
    np.random.seed(42)  # For reproducibility
    shuffled_ids = np.array(patient_ids)
    np.random.shuffle(shuffled_ids)

    fold_size = len(shuffled_ids) // n_folds
    val_start = fold_id * fold_size
    val_end = val_start + fold_size if fold_id < n_folds - 1 else len(shuffled_ids)

    val_ids = shuffled_ids[val_start:val_end].tolist()
    train_ids = np.concatenate([shuffled_ids[:val_start], shuffled_ids[val_end:]]).tolist()

    return train_ids, val_ids


def filter_csv_by_patients(csv_files: List[str], patient_ids: List[int], output_dir: Optional[str] = None) -> List[str]:
    """
    Filter CSV files to include only specific patient IDs.

    Args:
        csv_files: List of input CSV file paths
        patient_ids: List of patient IDs to include
        output_dir: Optional directory to save filtered CSVs

    Returns:
        List of filtered CSV file paths (or temporary paths)
    """
    import tempfile

    filtered_paths = []

    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            continue

        df = pd.read_csv(csv_file)
        filtered_df = df[df['Case'].isin(patient_ids)]

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(csv_file))
        else:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            output_path = temp_file.name
            temp_file.close()

        filtered_df.to_csv(output_path, index=False)
        filtered_paths.append(output_path)

    return filtered_paths

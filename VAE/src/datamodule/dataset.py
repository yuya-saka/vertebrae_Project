"""
3D Vertebral Dataset for VQ-VAE Training
正常椎体データのみを読み込み、3D Data Augmentationを適用する
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class VertebraeVAEDataset(Dataset):
    """
    3D椎体ボリュームデータセット (VQ-VAE学習用)

    Args:
        data_dir: 3Dボリュームデータのベースディレクトリ (train_vae/)
        volume_files: 読み込むボリュームファイルのリスト
        augmentation: Data Augmentation設定
        is_training: 学習モードかどうか (augmentationの有効/無効)
    """

    def __init__(
        self,
        data_dir: str,
        volume_files: List[str],
        augmentation: Optional[Dict] = None,
        is_training: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.volume_files = volume_files
        self.augmentation = augmentation if is_training else None
        self.is_training = is_training

        # ファイルの存在確認
        self._validate_files()

        print(f"Dataset initialized with {len(self.volume_files)} volumes")

    def _validate_files(self):
        """ファイルの存在を確認"""
        valid_files = []
        for vol_file in self.volume_files:
            vol_path = self.data_dir / vol_file
            if vol_path.exists():
                valid_files.append(vol_file)
            else:
                print(f"Warning: File not found: {vol_path}")

        if not valid_files:
            raise ValueError(f"No valid volume files found in {self.data_dir}")

        self.volume_files = valid_files

    def _load_volume(self, vol_path: Path) -> np.ndarray:
        """
        3Dボリュームデータを読み込む (.npy形式)

        Returns:
            volume: (D, H, W) shape, float32, [0, 1]正規化済み
        """
        volume = np.load(vol_path)

        # データ型と範囲の確認
        if volume.dtype != np.float32:
            volume = volume.astype(np.float32)

        # 既に[0,1]正規化されているはずだが、念のため確認
        if volume.min() < 0 or volume.max() > 1:
            print(f"Warning: Volume {vol_path.name} has values outside [0,1]: min={volume.min()}, max={volume.max()}")
            volume = np.clip(volume, 0, 1)

        return volume

    def _apply_augmentation(self, volume: torch.Tensor) -> torch.Tensor:
        """
        3D Data Augmentationを適用
        修正点: scale適用後に元のサイズに戻す処理を削除しました
        """
        if self.augmentation is None:
            return volume

        # --- ランダム反転 ---
        if self.augmentation.get('horizontal_flip', False):
            if torch.rand(1).item() > 0.5:
                volume = torch.flip(volume, dims=[3])

        if self.augmentation.get('vertical_flip', False):
            if torch.rand(1).item() > 0.5:
                volume = torch.flip(volume, dims=[2])

        # --- ランダム90度回転 ---
        if self.augmentation.get('rotation_90', False):
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                volume = torch.rot90(volume, k=k, dims=[2, 3])

        # --- Z軸回転 ---
        if self.augmentation.get('rotation_z', False):
            if torch.rand(1).item() > 0.5:
                rotation_range = self.augmentation.get('rotation_z_range', [-15, 15])
                angle = torch.FloatTensor(1).uniform_(rotation_range[0], rotation_range[1]).item()
                volume = self._rotate_volume_z(volume, angle)

        # --- 平行移動 ---
        if self.augmentation.get('translation_xy', False):
            if torch.rand(1).item() > 0.5:
                translation_percent = self.augmentation.get('translation_xy_percent', 0.05)
                volume = self._translate_volume_xy(volume, translation_percent)

        # --- スケーリング (修正箇所) ---
        if self.augmentation.get('scale', False):
            scale_range = self.augmentation.get('scale_range', [0.9, 1.1])
            scale = torch.FloatTensor(1).uniform_(scale_range[0], scale_range[1]).item()

            if abs(scale - 1.0) > 0.01:
                _, d, h, w = volume.shape
                new_size = (int(d * scale), int(h * scale), int(w * scale))

                # リサイズを実行
                volume = F.interpolate(volume.unsqueeze(0), size=new_size, mode='trilinear', align_corners=False).squeeze(0)
                
                # 【削除済み】ここで元のサイズに戻す処理がありましたが削除しました。
                # 最終的なサイズ合わせは __getitem__ で一括して行います。

        # --- ガウスノイズ ---
        if self.augmentation.get('gaussian_noise', False):
            if torch.rand(1).item() > 0.5:
                noise_std = self.augmentation.get('noise_std', 0.01)
                noise = torch.randn_like(volume) * noise_std
                volume = torch.clamp(volume + noise, 0, 1)

        # --- 輝度・コントラスト ---
        if self.augmentation.get('brightness', False):
            if torch.rand(1).item() > 0.5:
                brightness_range = self.augmentation.get('brightness_range', [-0.1, 0.1])
                brightness = torch.FloatTensor(1).uniform_(brightness_range[0], brightness_range[1]).item()
                volume = torch.clamp(volume + brightness, 0, 1)

        if self.augmentation.get('contrast', False):
            if torch.rand(1).item() > 0.5:
                contrast_range = self.augmentation.get('contrast_range', [0.9, 1.1])
                contrast = torch.FloatTensor(1).uniform_(contrast_range[0], contrast_range[1]).item()
                volume = torch.clamp(volume * contrast, 0, 1)

        return volume

    def _center_crop_or_pad(self, volume: torch.Tensor, target_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        中央クロップまたはパディングで目標サイズに調整
        修正版: 3Dパディングの計算を正確にし、F.padの仕様(W, H, D順)に準拠
        """
        _, d, h, w = volume.shape
        td, th, tw = target_size

        # --- 1. パディング処理 (サイズが足りない場合) ---
        pad_d = max(0, td - d)
        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            # 前後・上下・左右に均等に割り振る (奇数の場合は後ろ/下/右に+1)
            pad_front = pad_d // 2
            pad_back = pad_d - pad_front
            
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            # F.padの引数順序: (left, right, top, bottom, front, back) 
            # ※最後の次元(W)から順に指定するのがルール
            padding = (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
            volume = F.pad(volume, padding, mode='constant', value=0)

        # --- 2. クロップ処理 (サイズが大きい場合) ---
        # パディング後の現在のサイズを再取得
        _, curr_d, curr_h, curr_w = volume.shape
        
        if curr_d > td or curr_h > th or curr_w > tw:
            # 中央を切り抜くための開始位置を計算
            start_d = max(0, (curr_d - td) // 2)
            start_h = max(0, (curr_h - th) // 2)
            start_w = max(0, (curr_w - tw) // 2)

            volume = volume[:, start_d:start_d+td, start_h:start_h+th, start_w:start_w+tw]

        return volume

    def _rotate_volume_z(self, volume: torch.Tensor, angle: float) -> torch.Tensor:
        """
        z軸回りに3Dボリュームを回転 (3Dアフィン変換を使用)

        Args:
            volume: (1, D, H, W) Tensor
            angle: 回転角度（度）

        Returns:
            rotated_volume: (1, D, H, W) Tensor
        """
        import math

        # 角度をラジアンに変換
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # 3Dアフィン変換行列の作成 (z軸回り)
        # Bx3x4 の形式
        theta = torch.tensor([
            [cos_a, -sin_a, 0, 0],
            [sin_a,  cos_a, 0, 0],
            [    0,      0, 1, 0]
        ], dtype=volume.dtype, device=volume.device).unsqueeze(0) # (1, 3, 4)

        # グリッドサンプリング用のグリッド生成
        # affine_grid, grid_sampleは (N, C, D, H, W) を期待
        grid = F.affine_grid(theta, volume.unsqueeze(0).size(), align_corners=False)

        # 回転を適用
        rotated_volume = F.grid_sample(
            volume.unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        return rotated_volume.squeeze(0) # 元の (C, D, H, W) 形式に戻す

    def _translate_volume_xy(self, volume: torch.Tensor, translation_percent: float) -> torch.Tensor:
        """
        xy方向に3Dボリュームを平行移動 (3Dアフィン変換を使用)

        Args:
            volume: (1, D, H, W) Tensor
            translation_percent: 移動量の割合 (ボリュームサイズに対する比率)

        Returns:
            translated_volume: (1, D, H, W) Tensor
        """
        _, d, h, w = volume.shape

        # 移動量を計算（ピクセル単位）
        max_shift_h = int(h * translation_percent)
        max_shift_w = int(w * translation_percent)

        # ランダムな移動量を決定
        shift_h = torch.randint(-max_shift_h, max_shift_h + 1, (1,)).item()
        shift_w = torch.randint(-max_shift_w, max_shift_w + 1, (1,)).item()

        # 正規化された移動量 ([-1, 1]の範囲)
        shift_h_norm = 2.0 * shift_h / h
        shift_w_norm = 2.0 * shift_w / w

        # 3Dアフィン変換行列 (平行移動のみ)
        theta = torch.tensor([
            [1, 0, 0, shift_w_norm],
            [0, 1, 0, shift_h_norm],
            [0, 0, 1, 0]
        ], dtype=volume.dtype, device=volume.device).unsqueeze(0)  # (1, 3, 4)

        # グリッドサンプリング用のグリッド生成
        grid = F.affine_grid(theta, volume.unsqueeze(0).size(), align_corners=False)

        # 平行移動を適用
        translated_volume = F.grid_sample(
            volume.unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        return translated_volume.squeeze(0)

    def __len__(self) -> int:
        return len(self.volume_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        1つのサンプルを取得

        Returns:
            volume: (1, D, H, W) Tensor, [0, 1]正規化済み
        """
        vol_file = self.volume_files[idx]
        vol_path = self.data_dir / vol_file

        # ボリュームデータ読み込み
        volume_np = self._load_volume(vol_path)  # (D, H, W)

        # Tensorに変換 (1, D, H, W)
        #volume = torch.from_numpy(volume).unsqueeze(0).float()
        volume = torch.tensor(volume_np, dtype=torch.float32).unsqueeze(0)

        # Data Augmentation適用
        if self.is_training:
            volume = self._apply_augmentation(volume)
            
        # 【修正】固定サイズ (Depth=126, Height=128, Width=128) に統一
        target_size = (128, 128, 128)
        volume = self._center_crop_or_pad(volume, target_size)

        return volume.clone()


def get_normal_volume_files(data_dir: str) -> List[str]:
    """
    正常椎体のボリュームファイルリストを取得

    Args:
        data_dir: train_vae/ ディレクトリパス

    Returns:
        vol_files: ボリュームファイル名のリスト
    """
    data_path = Path(data_dir)
    vol_files = sorted([f.name for f in data_path.glob("vol_*.npy")])

    print(f"Found {len(vol_files)} normal volume files in {data_dir}")

    return vol_files


def split_volumes_by_patients(
    vol_files: List[str],
    train_patient_ids: List[int],
    val_patient_ids: List[int]
) -> Tuple[List[str], List[str]]:
    """
    患者IDに基づいてボリュームファイルを訓練/検証に分割

    Args:
        vol_files: 全ボリュームファイルのリスト
        train_patient_ids: 訓練用患者IDリスト
        val_patient_ids: 検証用患者IDリスト

    Returns:
        (train_files, val_files): 訓練/検証用ファイルリストのタプル
    """
    train_files = []
    val_files = []

    for vol_file in vol_files:
        # ファイル名から患者IDを抽出 (例: vol_1003_T4.npy -> 1003)
        patient_id = int(vol_file.split('_')[1])

        if patient_id in train_patient_ids:
            train_files.append(vol_file)
        elif patient_id in val_patient_ids:
            val_files.append(vol_file)
        else:
            print(f"Warning: Patient ID {patient_id} not in train or val lists")

    print(f"Train volumes: {len(train_files)}, Val volumes: {len(val_files)}")

    return train_files, val_files

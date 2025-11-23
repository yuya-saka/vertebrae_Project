"""
DataModule for VQ-VAE Training with 5-Fold Cross Validation
患者レベルでのfold分割を実装
"""

from typing import Dict, List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import VertebraeVAEDataset, get_normal_volume_files, split_volumes_by_patients


# fold_plan.mdに基づくFold定義 (正常椎体数を記載)
FOLD_DEFINITION = {
    1: {
        'patients': [1045, 1062, 1021, 1010, 1073, 1083],
        'normal_count': 35,
    },
    2: {
        'patients': [1039, 1051, 1030, 1012, 1067, 1080],
        'normal_count': 45,
    },
    3: {
        'patients': [1049, 1060, 1052, 1015, 1069, 1079],
        'normal_count': 48,
    },
    4: {
        'patients': [1035, 1054, 1025, 1016, 1043, 1059],
        'normal_count': 48,
    },
    5: {
        'patients': [1047, 1082, 1061, 1017, 1070, 1075],
        'normal_count': 49,
    },
}


class VertebraeVAEDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for VQ-VAE Training

    Args:
        data_dir: 正常椎体データのディレクトリ (data/3d_data/train_vae/)
        batch_size: バッチサイズ
        num_workers: DataLoaderのワーカー数
        fold_id: 検証用Fold ID (1-5)
        augmentation: Data Augmentation設定
        pin_memory: GPUメモリへのピン留め
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 4,
        num_workers: int = 4,
        fold_id: int = 1,
        augmentation: Optional[Dict] = None,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold_id = fold_id
        self.augmentation = augmentation
        self.pin_memory = pin_memory

        # Fold定義の検証
        if fold_id not in FOLD_DEFINITION:
            raise ValueError(f"Invalid fold_id: {fold_id}. Must be in {list(FOLD_DEFINITION.keys())}")

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        データセットのセットアップ

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # 全ボリュームファイルを取得
        all_vol_files = get_normal_volume_files(self.data_dir)

        # 患者IDリストの取得
        val_patient_ids = FOLD_DEFINITION[self.fold_id]['patients']

        # 訓練用患者IDリスト (検証用以外の全て)
        all_patient_ids = []
        for fold_num, fold_info in FOLD_DEFINITION.items():
            all_patient_ids.extend(fold_info['patients'])

        train_patient_ids = [pid for pid in all_patient_ids if pid not in val_patient_ids]

        print(f"\n{'='*80}")
        print(f"Fold {self.fold_id} Setup")
        print(f"{'='*80}")
        print(f"Train Patient IDs ({len(train_patient_ids)}): {sorted(train_patient_ids)}")
        print(f"Val Patient IDs ({len(val_patient_ids)}): {sorted(val_patient_ids)}")

        # ボリュームファイルを訓練/検証に分割
        train_files, val_files = split_volumes_by_patients(
            all_vol_files,
            train_patient_ids,
            val_patient_ids
        )

        # Datasetの作成
        if stage == 'fit' or stage is None:
            # 訓練データセット (Augmentation有効)
            self.train_dataset = VertebraeVAEDataset(
                data_dir=self.data_dir,
                volume_files=train_files,
                augmentation=self.augmentation,
                is_training=True,
            )

            # 検証データセット (Augmentation無効)
            self.val_dataset = VertebraeVAEDataset(
                data_dir=self.data_dir,
                volume_files=val_files,
                augmentation=None,
                is_training=False,
            )

            print(f"\nTrain Dataset: {len(self.train_dataset)} volumes")
            print(f"Val Dataset: {len(self.val_dataset)} volumes")
            print(f"Expected normal count for Fold {self.fold_id}: {FOLD_DEFINITION[self.fold_id]['normal_count']}")
            print(f"{'='*80}\n")

    def train_dataloader(self) -> DataLoader:
        """訓練用DataLoaderを返す"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # 訓練時はシャッフル
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """検証用DataLoaderを返す"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # 検証時はシャッフルしない
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )


def get_all_train_patient_ids() -> List[int]:
    """
    全訓練患者IDのリストを取得 (testデータを除く30症例)

    Returns:
        patient_ids: 訓練用患者IDのリスト
    """
    all_ids = []
    for fold_info in FOLD_DEFINITION.values():
        all_ids.extend(fold_info['patients'])

    return sorted(all_ids)


def get_fold_patient_ids(fold_id: int) -> Dict[str, List[int]]:
    """
    指定したFoldの訓練/検証患者IDを取得

    Args:
        fold_id: Fold ID (1-5)

    Returns:
        {'train': [...], 'val': [...]} の辞書
    """
    if fold_id not in FOLD_DEFINITION:
        raise ValueError(f"Invalid fold_id: {fold_id}")

    val_ids = FOLD_DEFINITION[fold_id]['patients']
    all_ids = get_all_train_patient_ids()
    train_ids = [pid for pid in all_ids if pid not in val_ids]

    return {
        'train': sorted(train_ids),
        'val': sorted(val_ids),
    }

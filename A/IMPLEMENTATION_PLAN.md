# A/ Attention-Guided Multi-Task Model 完全実装計画

## 📑 実装概要

**A/フォルダで一から学習パイプラインを構築**します。Hydra+WandBによる設定管理と可視化を含む、完全な医療AIプロジェクトを実装します。

## 🎯 実装方針

- ✅ **データ準備完了** (3方向スライス + マスク + CSV)
- **学習時に256×256へリサイズ**
- **HU値ウィンドウを設定可能に** (3チャンネル入力)
- **クラス不均衡対策**: **バッチ内クラス均衡サンプリング** + **オンライン拡張** (強い回転拡張)
- **Hydra (YAML)** で設定管理
- **WandB** で学習曲線可視化

---

## 📁 プロジェクト構造

```
A/
├── src/
│   ├── model/
│   │   ├── __init__.py
│   │   ├── multitask_unet.py          # Y字型マルチタスクU-Net
│   │   └── attention_gate.py          # Attention Gate実装
│   ├── modelmodule/
│   │   ├── __init__.py
│   │   ├── multitask_loss.py          # 分類+セグメンテーション複合損失
│   │   └── metrics.py                 # Dice, IoU, PR-AUC計算
│   ├── datamodule/
│   │   ├── __init__.py
│   │   ├── dataset.py                 # MultiTaskDataset (CT + Mask + Label)
│   │   ├── sampler.py                 # BalancedBatchSampler (バッチ内クラス均衡)
│   │   └── dataloader.py              # DataLoader作成関数
│   └── utils/
│       ├── __init__.py
│       └── common.py                  # シード固定、患者分割ユーティリティ
├── run/
│   ├── conf/
│   │   ├── config.yaml                # メイン設定
│   │   ├── constants.yaml             # データパス、患者ID定義
│   │   ├── train.yaml                 # 学習ハイパーパラメータ
│   │   ├── model/
│   │   │   ├── multitask_unet_resnet18.yaml
│   │   │   └── multitask_unet_efficientnet.yaml
│   │   ├── data/
│   │   │   ├── axial.yaml             # Axial方向のデータ設定
│   │   │   ├── coronal.yaml
│   │   │   └── sagittal.yaml
│   │   └── split/
│   │       ├── fold_0.yaml            # 患者レベル分割
│   │       ├── fold_1.yaml
│   │       ├── fold_2.yaml
│   │       ├── fold_3.yaml
│   │       └── fold_4.yaml
│   └── scripts/
│       ├── train.py                   # 学習スクリプト
│       └── eval.py                    # 評価スクリプト (後で実装)
├── notebook/
│   └── data_verification.ipynb        # データ検証用 (オプション)
├── output/                             # 学習結果保存先（方向別・fold別に自動分割）
│   ├── axial/
│   │   ├── fold_0/
│   │   │   ├── checkpoints/          # モデルの重み
│   │   │   ├── logs/                 # 学習ログ
│   │   │   └── config.yaml           # 使用した設定
│   │   ├── fold_1/
│   │   ├── fold_2/
│   │   ├── fold_3/
│   │   └── fold_4/
│   ├── coronal/
│   │   ├── fold_0/
│   │   ├── fold_1/
│   │   ├── fold_2/
│   │   ├── fold_3/
│   │   └── fold_4/
│   └── sagittal/
│       ├── fold_0/
│       ├── fold_1/
│       ├── fold_2/
│       ├── fold_3/
│       └── fold_4/
├── .gitignore
├── pyproject.toml                      # uv管理
└── README.md
```

---

## 🛠️ 実装ステップ (Phase別)

### **Phase 1: プロジェクト基盤構築**

#### 1.1 ディレクトリ構造作成
- A/ 以下の全ディレクトリを作成
- `__init__.py` を各パッケージに配置
- `pyproject.toml` 作成 (uv用)
- `.gitignore` 作成

#### 1.2 依存関係定義

`pyproject.toml`:
```toml
[project]
name = "vertebrae-multitask"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "torchvision",
    "nibabel",
    "numpy",
    "pandas",
    "opencv-python",
    "hydra-core>=1.3.0",
    "wandb",
    "scikit-learn",
    "matplotlib",
    "tqdm"
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

### **Phase 2: データモジュール実装**

#### 2.1 `src/datamodule/dataset.py` - MultiTaskDataset

**主要機能:**
- CT画像とマスク画像のペア読み込み
- CSV: `FullPath`, `Fracture_Label` と対応する `MaskPath` をベースにデータ構築
- **3チャンネルHU Window変換** (設定可能):
  - Ch1: [0, 1800] (骨全体)
  - Ch2: [-200, 300] (軟部組織)
  - Ch3: [200, 1200] (骨条件)
- **256×256へリサイズ** (cv2.INTER_LINEAR for image, cv2.INTER_NEAREST for mask)
- **オンライン拡張** (is_training=Trueの時のみ、`__getitem__()`で毎回適用):
  - **回転: ±45度 (大きめ)**
  - 平行移動: ±20px
  - スケール: 0.8-1.2
  - 水平反転: 50%
  - 輝度/コントラスト: ±0%
- **ラベルリスト取得メソッド**: `get_labels()` で全サンプルのラベルを返す (BalancedBatchSampler用)

**返り値:**
```python
{
    'image': torch.Tensor (3, 256, 256),  # 3ch HU window
    'mask': torch.Tensor (1, 256, 256),   # セグマスク
    'label_class': torch.Tensor (scalar), # 0 or 1
    'metadata': {
        'case': int,
        'vertebra': str,
        'slice_index': int
    }
}
```

**実装のポイント:**
```python
class MultiTaskDataset(Dataset):
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
        # CSVファイル読み込み
        self.data = self._load_csv_files(csv_files)

        # マスクパスを構築 (CT画像パスから対応するマスクパスを生成)
        self.data['MaskPath'] = self.data.apply(
            lambda row: self._construct_mask_path(row, mask_base_dir),
            axis=1
        )

        # オーバーサンプリングは使用しない（BalancedBatchSamplerで対応）
        print(f"Dataset initialized with {len(self.data)} samples")
        fracture_count = (self.data['Fracture_Label'] == 1).sum()
        print(f"Fracture slices: {fracture_count} ({fracture_count/len(self.data)*100:.2f}%)")

    def _construct_mask_path(self, row, mask_base_dir):
        # CT: .../axial/inp1003/27/slice_000.nii
        # -> Mask: .../axial_mask/inp1003/27/mask_000.nii
        ct_path = Path(row['FullPath'])
        case_id = f"inp{row['Case']}"
        vertebra = str(row['Vertebra'])
        slice_idx = row['SliceIndex']

        mask_path = Path(mask_base_dir) / case_id / vertebra / f"mask_{slice_idx:03d}.nii"
        return str(mask_path)

    def _create_3channel_input(self, image: np.ndarray) -> np.ndarray:
        """3チャンネルHU Window変換"""
        ch1 = self._normalize_hu_window(image.copy(),
                                        self.hu_windows['channel_1']['min'],
                                        self.hu_windows['channel_1']['max'])
        ch2 = self._normalize_hu_window(image.copy(),
                                        self.hu_windows['channel_2']['min'],
                                        self.hu_windows['channel_2']['max'])
        ch3 = self._normalize_hu_window(image.copy(),
                                        self.hu_windows['channel_3']['min'],
                                        self.hu_windows['channel_3']['max'])
        return np.stack([ch1, ch2, ch3], axis=0)

    def _apply_augmentation(self, image: np.ndarray, mask: np.ndarray):
        """強いデータ拡張 (±45度回転など)"""
        if self.augmentation is None:
            return image, mask

        # 画像とマスクを同時に変換
        # 回転角度は ±45度
        if np.random.rand() < 0.5:
            angle = np.random.uniform(
                -self.augmentation['rotation_degrees'],
                self.augmentation['rotation_degrees']
            )
            # ... (回転処理)

        # ... (その他の拡張処理)
        return image, mask

    def get_labels(self) -> List[int]:
        """
        全サンプルのラベルリストを返す（BalancedBatchSampler用）

        Returns:
            ラベルのリスト [0, 1, 0, 1, ...]
        """
        return self.data['Fracture_Label'].tolist()
```

#### 2.2 `src/datamodule/sampler.py` - BalancedBatchSampler

**役割:**
- **バッチ内でクラス均衡を保つカスタムサンプラー**
- 各バッチで骨折:非骨折 = 1:1 になるようにサンプリング
- エポック全体でデータセット全体を効率的に網羅

**主要機能:**
- バッチサイズ16の場合: 骨折8枚 + 非骨折8枚
- 各エポックで骨折・非骨折のインデックスをシャッフル
- バッチごとに各クラスから均等にサンプリング

**実装例:**
```python
import torch
from torch.utils.data import Sampler
import numpy as np
from typing import Iterator, List

class BalancedBatchSampler(Sampler):
    """
    バッチ内でクラス均衡を保つサンプラー

    各バッチで骨折:非骨折 = 1:1 になるようにサンプリング

    Args:
        labels: 全サンプルのラベルリスト (0 or 1)
        batch_size: バッチサイズ（偶数である必要がある）
        drop_last: 最後の不完全なバッチを捨てるか
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

        # 骨折・非骨折のインデックスを分離
        self.positive_indices = np.where(self.labels == 1)[0].tolist()
        self.negative_indices = np.where(self.labels == 0)[0].tolist()

        self.n_positive = len(self.positive_indices)
        self.n_negative = len(self.negative_indices)

        # 各バッチでのクラスごとのサンプル数
        self.samples_per_class = batch_size // 2

        # エポック内のバッチ数を計算
        self.n_batches = self._calculate_n_batches()

        print(f"BalancedBatchSampler initialized:")
        print(f"  Positive samples: {self.n_positive}")
        print(f"  Negative samples: {self.n_negative}")
        print(f"  Batch size: {batch_size} ({self.samples_per_class} pos + {self.samples_per_class} neg)")
        print(f"  Batches per epoch: {self.n_batches}")

    def _calculate_n_batches(self) -> int:
        """エポック内のバッチ数を計算"""
        # 各クラスで利用可能なバッチ数
        n_batches_positive = self.n_positive // self.samples_per_class
        n_batches_negative = self.n_negative // self.samples_per_class

        # 少ない方に合わせる
        n_batches = min(n_batches_positive, n_batches_negative)

        return n_batches

    def __iter__(self) -> Iterator[List[int]]:
        """バッチのイテレータを返す"""
        # 各クラスのインデックスをシャッフル
        np.random.shuffle(self.positive_indices)
        np.random.shuffle(self.negative_indices)

        # バッチを生成
        for batch_idx in range(self.n_batches):
            # 各クラスからサンプルを取得
            pos_start = batch_idx * self.samples_per_class
            pos_end = pos_start + self.samples_per_class

            neg_start = batch_idx * self.samples_per_class
            neg_end = neg_start + self.samples_per_class

            batch_positive = self.positive_indices[pos_start:pos_end]
            batch_negative = self.negative_indices[neg_start:neg_end]

            # バッチを結合してシャッフル
            batch = batch_positive + batch_negative
            np.random.shuffle(batch)

            yield batch

    def __len__(self) -> int:
        """エポック内のバッチ数を返す"""
        return self.n_batches
```

#### 2.3 `src/datamodule/dataloader.py`

**主要機能:**
- `create_dataloaders()` 関数
- 患者レベル分割をサポート (患者IDリストを受け取る)
- **学習用: BalancedBatchSampler使用** (バッチ内クラス均衡)
- **検証用: 通常のサンプリング**

**実装例:**
```python
from torch.utils.data import DataLoader
from .sampler import BalancedBatchSampler

def create_dataloaders(
    train_patient_ids: List[int],
    val_patient_ids: List[int],
    cfg: DictConfig
) -> Tuple[DataLoader, DataLoader]:
    """
    患者IDリストから学習/検証用DataLoaderを作成
    """
    # 全CSVファイルをリストアップ
    all_train_csv_files = list(Path(cfg.image_base_dir).glob("inp*/fracture_labels_inp*.csv"))

    # 患者IDでフィルタリング
    train_csv_files = [
        str(f) for f in all_train_csv_files
        if int(f.parent.name[3:]) in train_patient_ids
    ]
    val_csv_files = [
        str(f) for f in all_train_csv_files
        if int(f.parent.name[3:]) in val_patient_ids
    ]

    # Dataset作成
    train_dataset = MultiTaskDataset(
        csv_files=train_csv_files,
        image_base_dir=cfg.image_base_dir,
        mask_base_dir=cfg.mask_base_dir,
        hu_windows=cfg.hu_windows,
        image_size=cfg.image_size,
        augmentation=cfg.augmentation,
        is_training=True,
        # oversample_fracture は削除 (BalancedBatchSamplerで対応)
    )

    val_dataset = MultiTaskDataset(
        csv_files=val_csv_files,
        image_base_dir=cfg.image_base_dir,
        mask_base_dir=cfg.mask_base_dir,
        hu_windows=cfg.hu_windows,
        image_size=cfg.image_size,
        is_training=False
    )

    # BalancedBatchSampler作成（学習用のみ）
    train_sampler = BalancedBatchSampler(
        labels=train_dataset.get_labels(),
        batch_size=cfg.training.batch_size,
        drop_last=True  # 不完全なバッチを捨てる
    )

    # DataLoader作成
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,  # batch_samplerを使用する場合、batch_size/shuffleは指定しない
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
```

#### 2.4 `src/utils/common.py`

**主要機能:**
- `set_seed(seed)`: torch/numpy/randomのシード固定
- `split_patients(patient_ids, n_folds, fold_id)`: 患者レベルでのCV分割

```python
import random
import numpy as np
import torch
from typing import List, Tuple

def set_seed(seed: int):
    """再現性のためのシード固定"""
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
    患者レベルでのK-fold分割

    Args:
        patient_ids: 全患者IDリスト
        n_folds: fold数
        fold_id: 現在のfold (0-indexed)

    Returns:
        (train_patient_ids, val_patient_ids)
    """
    np.random.seed(42)  # 再現性のため固定
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
```

---

### **Phase 3: モデル実装**

#### 3.1 `src/model/attention_gate.py` - AttentionGate

**役割:**
- U-NetのスキップコネクションにAttentionを追加
- エンコーダからの特徴とデコーダからの特徴をゲート処理
- 参考: [Attention U-Net論文](https://arxiv.org/abs/1804.03999)

**実装例:**
```python
import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    """
    Attention Gate for U-Net skip connections

    Args:
        F_g: Number of feature maps in gating signal (decoder)
        F_l: Number of feature maps in skip connection (encoder)
        F_int: Number of intermediate feature maps
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: gating signal from decoder (B, F_g, H, W)
            x: skip connection from encoder (B, F_l, H, W)

        Returns:
            Attention-weighted feature map (B, F_l, H, W)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
```

#### 3.2 `src/model/multitask_unet.py` - MultiTaskUNet

**アーキテクチャ:**
```
Input (3, 256, 256)
    ↓
[Encoder] (ResNet18/EfficientNet-B0のpretrained backbone)
    ├─ conv1 → encoder_features[0]
    ├─ conv2 → encoder_features[1]
    ├─ conv3 → encoder_features[2]
    ├─ conv4 → encoder_features[3]
    └─ conv5 (bottleneck) → encoder_features[4]
         ↓
    ┌────┴────┐
    │         │
[Branch 1] [Branch 2]
分類ヘッド  セグデコーダ
    │         │
   GAP   Attention-UNet Decoder
    ↓    (AttentionGate at skip connections)
   FC         ↓
    ↓    1×1 Conv
 Sigmoid   Sigmoid
    ↓         ↓
P_class   P_seg (1, 256, 256)
(scalar)
```

**実装のポイント:**
```python
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple
from .attention_gate import AttentionGate

class MultiTaskUNet(nn.Module):
    """
    Y字型マルチタスクU-Net
    - 共通Encoder (ResNet18/EfficientNet pretrained)
    - Branch 1: 分類ヘッド (GAP + FC)
    - Branch 2: セグメンテーションデコーダ (Attention Gates付き)
    """

    def __init__(self, cfg):
        super().__init__()

        # Encoder (ResNet18 pretrained)
        if cfg.encoder_name == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            # 最初のconv層を3チャンネル入力に対応
            self.encoder_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if cfg.encoder_weights == 'imagenet':
                # ImageNet pretrainedの重みをコピー
                self.encoder_conv1.weight.data = backbone.conv1.weight.data

            self.encoder_bn1 = backbone.bn1
            self.encoder_relu = backbone.relu
            self.encoder_maxpool = backbone.maxpool

            self.encoder_layer1 = backbone.layer1  # 64 channels
            self.encoder_layer2 = backbone.layer2  # 128 channels
            self.encoder_layer3 = backbone.layer3  # 256 channels
            self.encoder_layer4 = backbone.layer4  # 512 channels (bottleneck)

        # Branch 1: 分類ヘッド
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(cfg.classifier.dropout),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        # Branch 2: セグメンテーションデコーダ (Attention Gates付き)
        self.decoder_channels = cfg.decoder_channels  # [256, 128, 64, 32, 16]

        # Attention Gates
        self.att4 = AttentionGate(F_g=self.decoder_channels[0], F_l=256, F_int=128)
        self.att3 = AttentionGate(F_g=self.decoder_channels[1], F_l=128, F_int=64)
        self.att2 = AttentionGate(F_g=self.decoder_channels[2], F_l=64, F_int=32)
        self.att1 = AttentionGate(F_g=self.decoder_channels[3], F_l=64, F_int=16)

        # Decoder blocks
        self.up4 = self._make_decoder_block(512, self.decoder_channels[0])
        self.up3 = self._make_decoder_block(self.decoder_channels[0] + 256, self.decoder_channels[1])
        self.up2 = self._make_decoder_block(self.decoder_channels[1] + 128, self.decoder_channels[2])
        self.up1 = self._make_decoder_block(self.decoder_channels[2] + 64, self.decoder_channels[3])
        self.up0 = self._make_decoder_block(self.decoder_channels[3] + 64, self.decoder_channels[4])

        # 最終セグメンテーション出力
        self.seg_head = nn.Sequential(
            nn.Conv2d(self.decoder_channels[4], 1, kernel_size=1),
            nn.Sigmoid()
        )

    def _make_decoder_block(self, in_channels: int, out_channels: int):
        """Decoderブロック: Upsample + Conv"""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input image (B, 3, 256, 256)

        Returns:
            p_class: Classification probability (B,)
            p_seg: Segmentation probability map (B, 1, 256, 256)
        """
        # Encoder forward
        x0 = self.encoder_conv1(x)  # (B, 64, 128, 128)
        x0 = self.encoder_bn1(x0)
        x0 = self.encoder_relu(x0)

        x1 = self.encoder_maxpool(x0)  # (B, 64, 64, 64)
        x1 = self.encoder_layer1(x1)   # (B, 64, 64, 64)

        x2 = self.encoder_layer2(x1)   # (B, 128, 32, 32)
        x3 = self.encoder_layer3(x2)   # (B, 256, 16, 16)
        x4 = self.encoder_layer4(x3)   # (B, 512, 8, 8) - bottleneck

        # Branch 1: 分類ヘッド
        p_class = self.classifier(x4).squeeze(1)  # (B,)

        # Branch 2: セグメンテーションデコーダ
        d4 = self.up4(x4)  # (B, 256, 16, 16)
        x3_att = self.att4(g=d4, x=x3)
        d4 = torch.cat([d4, x3_att], dim=1)

        d3 = self.up3(d4)  # (B, 128, 32, 32)
        x2_att = self.att3(g=d3, x=x2)
        d3 = torch.cat([d3, x2_att], dim=1)

        d2 = self.up2(d3)  # (B, 64, 64, 64)
        x1_att = self.att2(g=d2, x=x1)
        d2 = torch.cat([d2, x1_att], dim=1)

        d1 = self.up1(d2)  # (B, 32, 128, 128)
        x0_att = self.att1(g=d1, x=x0)
        d1 = torch.cat([d1, x0_att], dim=1)

        d0 = self.up0(d1)  # (B, 16, 256, 256)

        p_seg = self.seg_head(d0)  # (B, 1, 256, 256)

        return p_class, p_seg

    def freeze_encoder(self):
        """Encoderの重みを凍結 (ファインチューニング用)"""
        for param in self.encoder_conv1.parameters():
            param.requires_grad = False
        for param in self.encoder_bn1.parameters():
            param.requires_grad = False
        for param in self.encoder_layer1.parameters():
            param.requires_grad = False
        for param in self.encoder_layer2.parameters():
            param.requires_grad = False
        for param in self.encoder_layer3.parameters():
            param.requires_grad = False
        for param in self.encoder_layer4.parameters():
            param.requires_grad = False

    def get_encoder_params(self):
        """Encoder パラメータを取得 (差分学習率用)"""
        encoder_params = []
        encoder_params.extend(self.encoder_conv1.parameters())
        encoder_params.extend(self.encoder_bn1.parameters())
        encoder_params.extend(self.encoder_layer1.parameters())
        encoder_params.extend(self.encoder_layer2.parameters())
        encoder_params.extend(self.encoder_layer3.parameters())
        encoder_params.extend(self.encoder_layer4.parameters())
        return encoder_params

    def get_decoder_params(self):
        """Decoder + Classifier パラメータを取得"""
        decoder_params = []
        decoder_params.extend(self.classifier.parameters())
        decoder_params.extend(self.att4.parameters())
        decoder_params.extend(self.att3.parameters())
        decoder_params.extend(self.att2.parameters())
        decoder_params.extend(self.att1.parameters())
        decoder_params.extend(self.up4.parameters())
        decoder_params.extend(self.up3.parameters())
        decoder_params.extend(self.up2.parameters())
        decoder_params.extend(self.up1.parameters())
        decoder_params.extend(self.up0.parameters())
        decoder_params.extend(self.seg_head.parameters())
        return decoder_params
```

---

### **Phase 4: 損失関数・メトリクス実装**

#### 4.1 `src/modelmodule/multitask_loss.py` - MultiTaskLoss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance

    Args:
        alpha: Weighting factor (0-1) for positive class
        gamma: Focusing parameter (typically 2.0)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities (B, 1, H, W) or (B, H, W)
            target: Ground truth binary mask (B, 1, H, W) or (B, H, W)
        """
        pred = pred.view(-1)
        target = target.view(-1)

        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce

        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)
        """
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class MultiTaskLoss(nn.Module):
    """
    マルチタスク損失関数
    - 分類損失 (BCELoss) × w_class
    - セグメンテーション損失 (FocalLoss/DiceLoss) × w_seg

    Args:
        w_class: 分類損失の重み (デフォルト: 1.0)
        w_seg: セグメンテーション損失の重み (デフォルト: 0.1)
        seg_loss_type: セグ損失のタイプ ('focal', 'dice', 'focal_dice')
        focal_alpha: Focal Lossのalphaパラメータ
        focal_gamma: Focal Lossのgammaパラメータ
    """
    def __init__(
        self,
        w_class: float = 1.0,
        w_seg: float = 0.1,
        seg_loss_type: str = 'focal',
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.w_class = w_class
        self.w_seg = w_seg
        self.seg_loss_type = seg_loss_type

        # セグメンテーション損失の選択
        if seg_loss_type == 'focal':
            self.seg_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif seg_loss_type == 'dice':
            self.seg_criterion = DiceLoss()
        elif seg_loss_type == 'focal_dice':
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            self.dice_loss = DiceLoss()
        else:
            raise ValueError(f"Unknown seg_loss_type: {seg_loss_type}")

    def forward(
        self,
        pred_class: torch.Tensor,
        pred_seg: torch.Tensor,
        target_class: torch.Tensor,
        target_seg: torch.Tensor
    ) -> dict:
        """
        Args:
            pred_class: Classification predictions (B,)
            pred_seg: Segmentation predictions (B, 1, H, W)
            target_class: Classification labels (B,)
            target_seg: Segmentation masks (B, 1, H, W)

        Returns:
            Dict with 'total', 'class', 'seg' losses
        """
        # 分類損失
        loss_class = F.binary_cross_entropy(pred_class, target_class)

        # セグメンテーション損失
        if self.seg_loss_type == 'focal_dice':
            loss_focal = self.focal_loss(pred_seg, target_seg)
            loss_dice = self.dice_loss(pred_seg, target_seg)
            loss_seg = (loss_focal + loss_dice) / 2.0
        else:
            loss_seg = self.seg_criterion(pred_seg, target_seg)

        # 総損失
        total_loss = self.w_class * loss_class + self.w_seg * loss_seg

        return {
            'total': total_loss,
            'class': loss_class.item(),
            'seg': loss_seg.item()
        }
```

#### 4.2 `src/modelmodule/metrics.py`

```python
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from typing import Tuple

def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Dice係数の計算

    Args:
        pred: Predicted probabilities (B, 1, H, W)
        target: Ground truth binary mask (B, 1, H, W)
        threshold: Binarization threshold

    Returns:
        Dice coefficient (0-1)
    """
    pred_binary = (pred > threshold).float()
    target_binary = target.float()

    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-8)

    return dice.item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    IoU (Intersection over Union) の計算

    Args:
        pred: Predicted probabilities (B, 1, H, W)
        target: Ground truth binary mask (B, 1, H, W)
        threshold: Binarization threshold

    Returns:
        IoU score (0-1)
    """
    pred_binary = (pred > threshold).float()
    target_binary = target.float()

    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection

    iou = intersection / (union + 1e-8)

    return iou.item()


def compute_pr_auc(pred_probs: np.ndarray, targets: np.ndarray) -> float:
    """
    PR-AUC (Precision-Recall Area Under Curve) の計算

    Args:
        pred_probs: Predicted probabilities (N,)
        targets: Ground truth labels (N,)

    Returns:
        PR-AUC score (0-1)
    """
    if len(np.unique(targets)) < 2:
        # Only one class present
        return 0.0

    precision, recall, _ = precision_recall_curve(targets, pred_probs)
    pr_auc = auc(recall, precision)

    return pr_auc


def compute_metrics_batch(
    pred_class: torch.Tensor,
    pred_seg: torch.Tensor,
    target_class: torch.Tensor,
    target_seg: torch.Tensor
) -> dict:
    """
    バッチ全体のメトリクスを計算

    Args:
        pred_class: Classification predictions (B,)
        pred_seg: Segmentation predictions (B, 1, H, W)
        target_class: Classification labels (B,)
        target_seg: Segmentation masks (B, 1, H, W)

    Returns:
        Dict with all metrics
    """
    # 分類精度
    pred_class_binary = (pred_class > 0.5).float()
    class_acc = (pred_class_binary == target_class).float().mean().item()

    # セグメンテーション精度
    dice = dice_coefficient(pred_seg, target_seg)
    iou = iou_score(pred_seg, target_seg)

    return {
        'class_acc': class_acc,
        'dice': dice,
        'iou': iou
    }
```

---

### **Phase 5: Hydra設定ファイル構築**

#### 5.1 `run/conf/config.yaml` (メイン設定)

```yaml
defaults:
  - constants
  - train
  - data: axial           # 実験ごとに変更: axial, coronal, sagittal
  - model: multitask_unet_resnet18
  - split: fold_0
  - _self_

experiment:
  # 実験名は自動生成: {axis}/fold_{fold_id}
  # name は train.py で data.axis と split.fold_id から自動生成される
  description: "Multi-task U-Net with ${model.encoder_name} on ${data.axis} slices"
  # tags も train.py で自動生成される

seed: 42

# WandB設定
wandb:
  project: "vertebrae_multitask"
  entity: null  # 自分のユーザー名を設定 (nullの場合はデフォルトアカウント)
  mode: "online"  # "online", "offline", "disabled"
  log_interval: 10  # ログを記録するステップ間隔
  # group と name は train.py で自動設定される
```

#### 5.2 `run/conf/constants.yaml` (データパス・患者ID)

```yaml
# プロジェクトルート (絶対パス)
project_root: "/mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka"

# データディレクトリ
data_dir: "${project_root}/data"
slice_train_dir: "${data_dir}/slice_train"
slice_test_dir: "${data_dir}/slice_test"

# 出力ディレクトリ（方向別・fold別に自動分割）
output_base_dir: "${project_root}/A/output"
# 実際の出力先は自動生成: {output_base_dir}/{axis}/fold_{fold_id}/

# 患者ID定義 (train/test分割)
train_patient_ids: [
  1003, 1015, 1017, 1025, 1027, 1030, 1035, 1038, 1039, 1043,
  1045, 1046, 1047, 1049, 1052, 1055, 1059, 1060, 1061, 1062,
  1067, 1069, 1070, 1073, 1074, 1075, 1077, 1080, 1082, 1083
]

test_patient_ids: [1010, 1012, 1016, 1021, 1051, 1054, 1079, 1084]
```

#### 5.3 `run/conf/train.yaml` (学習ハイパーパラメータ)

```yaml
# 学習ハイパーパラメータ
training:
  batch_size: 16
  num_workers: 4
  max_epochs: 100
  accumulation_steps: 1  # Gradient accumulation (GPUメモリ不足時は2以上に)

  # 早期終了
  early_stopping:
    enabled: true
    patience: 15
    monitor: "val_loss"  # "val_loss" or "val_pr_auc"
    mode: "min"          # "min" for loss, "max" for pr_auc

  # チェックポイント
  checkpoint:
    save_top_k: 3
    monitor: "val_pr_auc"
    mode: "max"
    save_last: true

# オプティマイザ
optimizer:
  name: "AdamW"
  lr: 0.001
  weight_decay: 0.0001

  # 差分学習率 (Encoder vs Decoder)
  use_differential_lr: true
  encoder_lr_factor: 0.1  # encoder_lr = lr * 0.1

# スケジューラ
scheduler:
  name: "ReduceLROnPlateau"  # "ReduceLROnPlateau", "CosineAnnealingLR", "StepLR"
  mode: "min"
  factor: 0.5
  patience: 5
  min_lr: 0.00001

# 損失関数の重み
loss:
  w_class: 1.0            # 分類損失の重み (主タスク)
  w_seg: 0.1              # セグ損失の重み (補助タスク)
  seg_loss_type: "focal"  # 'focal', 'dice', 'focal_dice'
  focal_alpha: 0.25
  focal_gamma: 2.0
```

#### 5.4 `run/conf/data/axial.yaml`

```yaml
# Axial方向のデータ設定
axis: "axial"

# データパス (constants.yamlの変数を使用)
image_base_dir: "${slice_train_dir}/axial"
mask_base_dir: "${slice_train_dir}/axial_mask"

# 画像設定
image_size: [256, 256]  # (H, W)

# HU Window設定 (3チャンネル)
hu_windows:
  channel_1:
    min: 0
    max: 1800
    description: "全骨条件"
  channel_2:
    min: -200
    max: 300
    description: "軟部組織"
  channel_3:
    min: 200
    max: 1200
    description: "骨条件"

# データ拡張設定
augmentation:
  rotation_degrees: 45      # ±45度 (大きめ)
  translation_pixels: 20    # ±20px
  scale_range: [0.8, 1.2]   # 0.8x ~ 1.2x
  horizontal_flip_prob: 0.5
  contrast_range: [0.9, 1.1]

# クラス不均衡対策
# BalancedBatchSamplerで対応（バッチ内で骨折:非骨折 = 1:1）
```

#### 5.5 `run/conf/data/coronal.yaml`

```yaml
# Coronal方向のデータ設定
axis: "coronal"

image_base_dir: "${slice_train_dir}/coronal"
mask_base_dir: "${slice_train_dir}/coronal_mask"

image_size: [256, 256]

hu_windows:
  channel_1:
    min: 0
    max: 1800
    description: "全骨条件"
  channel_2:
    min: -200
    max: 300
    description: "軟部組織"
  channel_3:
    min: 200
    max: 1200
    description: "骨条件"

augmentation:
  rotation_degrees: 45
  translation_pixels: 20
  scale_range: [0.8, 1.2]
  horizontal_flip_prob: 0.5
  contrast_range: [0.9, 1.1]

# クラス不均衡対策
# BalancedBatchSamplerで対応（バッチ内で骨折:非骨折 = 1:1）
```

#### 5.6 `run/conf/data/sagittal.yaml`

```yaml
# Sagittal方向のデータ設定
axis: "sagittal"

image_base_dir: "${slice_train_dir}/sagittal"
mask_base_dir: "${slice_train_dir}/sagittal_mask"

image_size: [256, 256]

hu_windows:
  channel_1:
    min: 0
    max: 1800
    description: "全骨条件"
  channel_2:
    min: -200
    max: 300
    description: "軟部組織"
  channel_3:
    min: 200
    max: 1200
    description: "骨条件"

augmentation:
  rotation_degrees: 45
  translation_pixels: 20
  scale_range: [0.8, 1.2]
  horizontal_flip_prob: 0.5
  contrast_range: [0.9, 1.1]

# クラス不均衡対策
# BalancedBatchSamplerで対応（バッチ内で骨折:非骨折 = 1:1）
```

#### 5.7 `run/conf/model/multitask_unet_resnet18.yaml`

```yaml
model:
  name: "MultiTaskUNet"

  # Encoder設定
  encoder_name: "resnet18"
  encoder_weights: "imagenet"  # "imagenet" or null (random init)
  in_channels: 3

  # Decoder設定
  decoder_channels: [256, 128, 64, 32, 16]
  decoder_attention_type: "scse"  # 'scse', 'cbam', null (no attention)

  # 分類ヘッド設定
  classifier:
    dropout: 0.2
    use_gap: true  # Global Average Pooling
```

#### 5.8 `run/conf/model/multitask_unet_efficientnet.yaml`

```yaml
model:
  name: "MultiTaskUNet"

  # Encoder設定
  encoder_name: "efficientnet-b0"
  encoder_weights: "imagenet"
  in_channels: 3

  # Decoder設定
  decoder_channels: [256, 128, 64, 32, 16]
  decoder_attention_type: "scse"

  # 分類ヘッド設定
  classifier:
    dropout: 0.3  # EfficientNetは過学習しやすいので少し大きめ
    use_gap: true
```

#### 5.9 `run/conf/split/fold_0.yaml`

```yaml
# K-fold CV設定
n_folds: 5
fold_id: 0

# この設定は自動的に患者を分割します
# train_patient_ids は constants.yaml から読み込み
```

#### 5.10 `run/conf/split/fold_1.yaml`

```yaml
n_folds: 5
fold_id: 1
```

#### 5.11 `run/conf/split/fold_2.yaml`

```yaml
n_folds: 5
fold_id: 2
```

#### 5.12 `run/conf/split/fold_3.yaml`

```yaml
n_folds: 5
fold_id: 3
```

#### 5.13 `run/conf/split/fold_4.yaml`

```yaml
n_folds: 5
fold_id: 4
```

---

### **Phase 6: 学習スクリプト実装**

#### 6.1 `run/scripts/train.py`

**完全な学習スクリプト** (長いので主要部分のみ記載):

```python
#!/usr/bin/env python3
"""
Multi-Task U-Net Training Script
"""

import os
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.model.multitask_unet import MultiTaskUNet
from src.modelmodule.multitask_loss import MultiTaskLoss
from src.modelmodule.metrics import compute_pr_auc, compute_metrics_batch
from src.datamodule.dataloader import create_dataloaders
from src.utils.common import set_seed, split_patients


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """メイン学習関数"""

    # ========================================
    # 0. 実験名とディレクトリの自動生成
    # ========================================
    axis = cfg.data.axis  # "axial", "coronal", "sagittal"
    fold_id = cfg.split.fold_id  # 0, 1, 2, 3, 4
    model_name = cfg.model.encoder_name  # "resnet18", "efficientnet-b0"

    # 実験名: axis/fold_X
    experiment_name = f"{axis}/fold_{fold_id}"

    # 出力ディレクトリ: A/output/axis/fold_X/
    output_dir = Path(cfg.output_base_dir) / axis / f"fold_{fold_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # サブディレクトリ作成
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    # 設定を動的に更新
    OmegaConf.set_struct(cfg, False)  # 構造を一時的に解除
    if 'experiment' not in cfg:
        cfg.experiment = {}
    cfg.experiment.name = experiment_name
    cfg.output_dir = str(output_dir)
    cfg.checkpoint_dir = str(checkpoint_dir)
    cfg.log_dir = str(log_dir)
    OmegaConf.set_struct(cfg, True)  # 構造を再度有効化

    # ========================================
    # ログファイルの設定
    # ========================================
    from datetime import datetime
    import logging

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    # ファイルハンドラを追加（コンソール出力とファイル出力の両方）
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.addHandler(file_handler)

    # ========================================
    # 1. 初期化
    # ========================================
    print("="*80)
    print("Multi-Task U-Net Training")
    print("="*80)
    print(f"Experiment: {experiment_name}")
    print(f"Axis: {axis}")
    print(f"Fold: {fold_id}")
    print(f"Model: {model_name}")
    print(f"Description: {cfg.experiment.description}")
    print(f"Output dir: {output_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Log file: {log_file}")
    print("="*80)

    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 使用した設定を保存
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        OmegaConf.save(cfg, f)
    print(f"Config saved: {config_save_path}")

    # ========================================
    # 2. WandB初期化（階層的な管理）
    # ========================================
    if cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=experiment_name,  # "axial/fold_0"
            group=axis,  # 同じ方向の実験をグループ化
            tags=[
                axis,
                f"fold_{fold_id}",
                model_name,
                "multitask",
                "attention"
            ],
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode
        )

        # WandBに追加情報をログ
        wandb.config.update({
            "experiment_name": experiment_name,
            "output_dir": str(output_dir),
            "axis": axis,
            "fold_id": fold_id
        })

    # 3. 患者分割
    train_ids, val_ids = split_patients(
        cfg.train_patient_ids,
        cfg.n_folds,
        cfg.fold_id
    )
    print(f"\nPatient Split (Fold {cfg.fold_id}/{cfg.n_folds}):")
    print(f"  Train patients: {len(train_ids)} - {train_ids[:5]}...")
    print(f"  Val patients: {len(val_ids)} - {val_ids}")

    # 4. データローダー作成
    print("\nCreating DataLoaders...")
    train_loader, val_loader = create_dataloaders(
        train_patient_ids=train_ids,
        val_patient_ids=val_ids,
        cfg=cfg
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # 5. モデル作成
    print("\nCreating Model...")
    model = MultiTaskUNet(cfg.model).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # 6. 損失関数
    criterion = MultiTaskLoss(
        w_class=cfg.loss.w_class,
        w_seg=cfg.loss.w_seg,
        seg_loss_type=cfg.loss.seg_loss_type,
        focal_alpha=cfg.loss.focal_alpha,
        focal_gamma=cfg.loss.focal_gamma
    )

    # 7. オプティマイザ
    if cfg.optimizer.use_differential_lr:
        optimizer = torch.optim.AdamW([
            {
                'params': model.get_encoder_params(),
                'lr': cfg.optimizer.lr * cfg.optimizer.encoder_lr_factor
            },
            {
                'params': model.get_decoder_params(),
                'lr': cfg.optimizer.lr
            }
        ], weight_decay=cfg.optimizer.weight_decay)
        print(f"\nOptimizer: AdamW with differential LR")
        print(f"  Encoder LR: {cfg.optimizer.lr * cfg.optimizer.encoder_lr_factor}")
        print(f"  Decoder LR: {cfg.optimizer.lr}")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )
        print(f"\nOptimizer: AdamW with LR={cfg.optimizer.lr}")

    # 8. スケジューラ
    if cfg.scheduler.name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.scheduler.mode,
            factor=cfg.scheduler.factor,
            patience=cfg.scheduler.patience,
            min_lr=cfg.scheduler.min_lr
        )

    # 9. 学習ループ
    best_val_metric = 0.0 if cfg.training.checkpoint.mode == "max" else float('inf')
    patience_counter = 0

    for epoch in range(cfg.training.max_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{cfg.training.max_epochs}")
        print(f"{'='*80}")

        # Training
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            cfg=cfg
        )

        # Validation
        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch
        )

        # Scheduler step
        if cfg.scheduler.name == "ReduceLROnPlateau":
            scheduler.step(val_metrics['val_loss'])

        # 現在の学習率
        current_lr = optimizer.param_groups[0]['lr']

        # メトリクス表示
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f} | Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Train PR-AUC: {train_metrics['train_pr_auc']:.4f} | Val PR-AUC: {val_metrics['val_pr_auc']:.4f}")
        print(f"  Val Dice: {val_metrics['val_dice']:.4f} | Val IoU: {val_metrics['val_iou']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # WandB logging
        if cfg.wandb.mode != "disabled":
            wandb.log({
                **train_metrics,
                **val_metrics,
                'lr': current_lr,
                'epoch': epoch
            })

        # チェックポイント保存
        monitor_metric = val_metrics[cfg.training.checkpoint.monitor]
        is_best = False

        if cfg.training.checkpoint.mode == "max":
            if monitor_metric > best_val_metric:
                best_val_metric = monitor_metric
                is_best = True
                patience_counter = 0
            else:
                patience_counter += 1
        else:  # mode == "min"
            if monitor_metric < best_val_metric:
                best_val_metric = monitor_metric
                is_best = True
                patience_counter = 0
            else:
                patience_counter += 1

        if is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                cfg=cfg,
                filename='best_model.pth'
            )
            print(f"  ✓ Best model saved! ({cfg.training.checkpoint.monitor}={monitor_metric:.4f})")

        # 最後のエポックを常に保存
        if cfg.training.checkpoint.save_last:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                cfg=cfg,
                filename='last_model.pth'
            )

        # Early stopping
        if cfg.training.early_stopping.enabled:
            if patience_counter >= cfg.training.early_stopping.patience:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                print(f"  No improvement for {patience_counter} epochs")
                break

    # 学習完了
    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"Best {cfg.training.checkpoint.monitor}: {best_val_metric:.4f}")
    print(f"{'='*80}")

    if cfg.wandb.mode != "disabled":
        wandb.finish()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    cfg: DictConfig
) -> dict:
    """1エポックの学習"""
    model.train()

    total_loss = 0.0
    total_loss_class = 0.0
    total_loss_seg = 0.0

    all_preds_class = []
    all_targets_class = []

    pbar = tqdm(loader, desc=f"Train", leave=False)

    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        labels_class = batch['label_class'].to(device).float()

        # Forward
        pred_class, pred_seg = model(images)

        # Loss計算
        losses = criterion(pred_class, pred_seg, labels_class, masks)
        loss = losses['total']

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # メトリクス収集
        total_loss += loss.item()
        total_loss_class += losses['class']
        total_loss_seg += losses['seg']

        all_preds_class.extend(pred_class.detach().cpu().numpy())
        all_targets_class.extend(labels_class.detach().cpu().numpy())

        # プログレスバー更新
        pbar.set_postfix({
            'loss': loss.item(),
            'loss_cls': losses['class'],
            'loss_seg': losses['seg']
        })

    # エポック全体のメトリクス
    avg_loss = total_loss / len(loader)
    avg_loss_class = total_loss_class / len(loader)
    avg_loss_seg = total_loss_seg / len(loader)

    pr_auc = compute_pr_auc(
        np.array(all_preds_class),
        np.array(all_targets_class)
    )

    return {
        'train_loss': avg_loss,
        'train_loss_class': avg_loss_class,
        'train_loss_seg': avg_loss_seg,
        'train_pr_auc': pr_auc
    }


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    epoch: int
) -> dict:
    """検証"""
    model.eval()

    total_loss = 0.0
    total_loss_class = 0.0
    total_loss_seg = 0.0

    all_preds_class = []
    all_targets_class = []

    all_dice = []
    all_iou = []

    pbar = tqdm(loader, desc=f"Val", leave=False)

    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            labels_class = batch['label_class'].to(device).float()

            # Forward
            pred_class, pred_seg = model(images)

            # Loss計算
            losses = criterion(pred_class, pred_seg, labels_class, masks)
            loss = losses['total']

            # メトリクス収集
            total_loss += loss.item()
            total_loss_class += losses['class']
            total_loss_seg += losses['seg']

            all_preds_class.extend(pred_class.cpu().numpy())
            all_targets_class.extend(labels_class.cpu().numpy())

            # セグメンテーションメトリクス
            batch_metrics = compute_metrics_batch(pred_class, pred_seg, labels_class, masks)
            all_dice.append(batch_metrics['dice'])
            all_iou.append(batch_metrics['iou'])

            pbar.set_postfix({'loss': loss.item()})

    # エポック全体のメトリクス
    avg_loss = total_loss / len(loader)
    avg_loss_class = total_loss_class / len(loader)
    avg_loss_seg = total_loss_seg / len(loader)

    pr_auc = compute_pr_auc(
        np.array(all_preds_class),
        np.array(all_targets_class)
    )

    avg_dice = np.mean(all_dice)
    avg_iou = np.mean(all_iou)

    return {
        'val_loss': avg_loss,
        'val_loss_class': avg_loss_class,
        'val_loss_seg': avg_loss_seg,
        'val_pr_auc': pr_auc,
        'val_dice': avg_dice,
        'val_iou': avg_iou
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    cfg: DictConfig,
    filename: str
):
    """チェックポイント保存"""
    # cfg.checkpoint_dir は main() で設定済み
    checkpoint_dir = Path(cfg.checkpoint_dir)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': OmegaConf.to_container(cfg, resolve=True),
        # メタ情報を追加
        'axis': cfg.data.axis,
        'fold_id': cfg.split.fold_id,
        'model_name': cfg.model.encoder_name
    }

    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)

    print(f"  Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    main()
```

---

### **Phase 7: 実行準備**

#### 7.1 `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
virtual_env/
ENV/

# Hydra
outputs/
multirun/
.hydra/

# WandB
wandb/

# 出力
output/
*.pth
*.pt

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo
```

#### 7.2 環境構築

```bash
cd /mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/A/

# uv環境初期化
uv init
uv sync

# 仮想環境有効化
source .venv/bin/activate
```

#### 7.3 学習実行

**実行コマンド例**（方向別・fold別に自動管理）：

```bash
cd run/scripts

# Axial方向、Fold 0 (デフォルト)
uv run python train.py
# -> 出力先: A/output/axial/fold_0/
# -> WandB run名: axial/fold_0

# Axial方向、全Fold実行
uv run python train.py split.fold_id=0  # A/output/axial/fold_0/
uv run python train.py split.fold_id=1  # A/output/axial/fold_1/
uv run python train.py split.fold_id=2  # A/output/axial/fold_2/
uv run python train.py split.fold_id=3  # A/output/axial/fold_3/
uv run python train.py split.fold_id=4  # A/output/axial/fold_4/

# Coronal方向、Fold 0
uv run python train.py data=coronal split.fold_id=0
# -> 出力先: A/output/coronal/fold_0/
# -> WandB run名: coronal/fold_0

# Sagittal方向、Fold 1
uv run python train.py data=sagittal split.fold_id=1
# -> 出力先: A/output/sagittal/fold_1/
# -> WandB run名: sagittal/fold_1

# EfficientNetで実行
uv run python train.py model=multitask_unet_efficientnet
# -> A/output/axial/fold_0/ (モデル名は自動反映)

# 3方向×5fold = 15実験を順次実行
for axis in axial coronal sagittal; do
  for fold in 0 1 2 3 4; do
    uv run python train.py data=$axis split.fold_id=$fold
  done
done

# WandBをオフラインモードで実行 (デバッグ用)
uv run python train.py wandb.mode=offline

# 学習率を変更して実行
uv run python train.py optimizer.lr=0.0005
```

**生成されるディレクトリ構造**：
```
A/output/
├── axial/
│   ├── fold_0/
│   │   ├── checkpoints/
│   │   │   ├── best_model.pth
│   │   │   └── last_model.pth
│   │   ├── logs/
│   │   │   └── train_20250112_143000.log
│   │   └── config.yaml
│   ├── fold_1/
│   ├── fold_2/
│   ├── fold_3/
│   └── fold_4/
├── coronal/
│   ├── fold_0/
│   ├── fold_1/
│   ├── fold_2/
│   ├── fold_3/
│   └── fold_4/
└── sagittal/
    ├── fold_0/
    ├── fold_1/
    ├── fold_2/
    ├── fold_3/
    └── fold_4/
```

---

## 📊 実装優先順位とタイムライン

### **Week 1: プロジェクト基盤 + データモジュール**

**Day 1-2: プロジェクト構造作成**
- [ ] ディレクトリ構造作成
- [ ] `__init__.py` 配置
- [ ] `pyproject.toml` 作成
- [ ] `.gitignore` 作成

**Day 3-5: データモジュール実装**
- [ ] `src/utils/common.py` 実装
- [ ] `src/datamodule/dataset.py` 実装 (オンライン拡張 + get_labels()メソッド)
- [ ] `src/datamodule/sampler.py` 実装 (BalancedBatchSampler)
- [ ] `src/datamodule/dataloader.py` 実装 (BalancedBatchSampler使用)
- [ ] データローダーの動作確認 (サンプルデータで検証)
- [ ] バッチ内クラス比が1:1になることを確認

### **Week 2: モデル + 損失関数実装**

**Day 1-3: モデル実装**
- [ ] `src/model/attention_gate.py` 実装
- [ ] `src/model/multitask_unet.py` 実装
- [ ] モデルのforward pass検証 (ダミーデータで確認)

**Day 4-5: 損失関数・メトリクス実装**
- [ ] `src/modelmodule/multitask_loss.py` 実装
- [ ] `src/modelmodule/metrics.py` 実装
- [ ] 損失関数の動作確認

### **Week 3: 設定ファイル + 学習スクリプト**

**Day 1-2: Hydra設定ファイル**
- [ ] 全YAML設定ファイル作成
- [ ] Hydraの動作確認 (設定読み込みテスト)

**Day 3-5: 学習スクリプト実装**
- [ ] `run/scripts/train.py` 実装
- [ ] 学習ループの動作確認 (小規模データでテスト)

### **Week 4: デバッグ・学習実行**

**Day 1-2: デバッグ**
- [ ] エンドツーエンドでの動作確認
- [ ] メモリ使用量確認
- [ ] WandB連携確認

**Day 3-5: 学習実行**
- [ ] Axial方向で学習開始
- [ ] 学習曲線の確認
- [ ] ハイパーパラメータ調整

---

## ✅ 実装確認チェックリスト

### **データモジュール**
- [ ] Dataset: 256×256リサイズ確認
- [ ] Dataset: 3チャンネルHU Window変換確認
- [ ] Dataset: CT画像とマスクのパス対応確認
- [ ] Dataset: `get_labels()` メソッドが正しくラベルリストを返す
- [ ] Dataset: ±45度回転拡張確認（オンライン拡張）
- [ ] Sampler: BalancedBatchSamplerが正しく初期化される
- [ ] Sampler: 各バッチで骨折:非骨折 = 1:1になることを確認
- [ ] DataLoader: BalancedBatchSamplerを使用してバッチが生成される
- [ ] DataLoader: 患者レベル分割でデータリーケージなし

### **モデル**
- [ ] Model: forward()で (p_class, p_seg) が返る
- [ ] Model: p_class の shape が (B,) になる
- [ ] Model: p_seg の shape が (B, 1, 256, 256) になる
- [ ] Model: Attention Gateが正しく動作する
- [ ] Model: Encoder/Decoderのパラメータ取得メソッド動作確認

### **損失関数**
- [ ] Loss: w_class=1.0, w_seg=0.1 の重み付け確認
- [ ] Loss: Focal Loss の動作確認
- [ ] Loss: 総損失が適切に計算される

### **学習**
- [ ] Training: WandBにloss/PR-AUCがログされる
- [ ] Training: チェックポイントが保存される
- [ ] Training: Early stoppingが動作する
- [ ] Training: 差分学習率が適用される
- [ ] Training: メモリリーク・OOMが発生しない

---

## 🚀 次のステップ (Phase 8以降)

### **Phase 8: 評価スクリプト**
- `run/scripts/eval.py` 実装
- テストデータでの推論
- メトリクス算出

### **Phase 9: 3D統合実装**
- ステップ1: P_final = P_class × P_seg
- ステップ2: P_voxel = P_ax × P_co × P_sa
- 3D可視化

### **Phase 10: 実験・論文執筆**
- 3方向での実験実施
- アブレーションスタディ
- 結果の可視化
- 論文執筆

---

## 📚 参考資料

### **論文**
- Attention U-Net: https://arxiv.org/abs/1804.03999
- Focal Loss: https://arxiv.org/abs/1708.02002
- Multi-Task Learning: https://arxiv.org/abs/1706.05098

### **実装参考**
- segmentation_models_pytorch: https://github.com/qubvel/segmentation_models.pytorch
- Hydra: https://hydra.cc/docs/intro/

---

## 🔄 データ拡張法改善の概要

### **改善前: オーバーサンプリング方式**
- データセット全体で骨折スライスを3倍に複製
- エポック全体では均衡だが、**バッチ内では不均衡**
- あるバッチは全て非骨折、別のバッチは骨折が多いという偏り

### **改善後: バッチ内クラス均衡サンプリング + オンライン拡張**
- **BalancedBatchSampler** で各バッチ内で骨折:非骨折 = 1:1
- **オンライン拡張** で毎エポック異なる拡張パターン
- メモリ効率的（データセットを複製しない）
- 学習の安定性向上（バッチ間の損失のばらつきが減少）

### **利点まとめ**
✅ バッチ内クラス均衡: 各バッチで骨折:非骨折 = 1:1
✅ オンライン拡張: 毎エポック異なる拡張パターン
✅ メモリ効率: データセットを複製しない
✅ 学習の安定性: バッチごとの損失のばらつきが減少
✅ 汎化性能: 強いオンライン拡張により過学習を抑制

---

## 📁 方向別・Fold別結果管理の改善概要

### **改善のポイント**

#### **1. 階層的なディレクトリ構造**
```
A/output/
├── axial/fold_0/, axial/fold_1/, axial/fold_2/, axial/fold_3/, axial/fold_4/
├── coronal/fold_0/, coronal/fold_1/, coronal/fold_2/, coronal/fold_3/, coronal/fold_4/
└── sagittal/fold_0/, sagittal/fold_1/, sagittal/fold_2/, sagittal/fold_3/, sagittal/fold_4/
```

各ディレクトリ内:
- `checkpoints/`: モデルの重み
- `logs/`: 学習ログ（タイムスタンプ付き）
- `config.yaml`: 使用した設定

#### **2. 実験名の自動生成**
- フォーマット: `{axis}/fold_{fold_id}`
- 例: `axial/fold_0`, `coronal/fold_1`, `sagittal/fold_2`
- `data.axis` と `split.fold_id` から自動生成

#### **3. WandBの階層的管理**
- **Run名**: `{axis}/fold_{fold_id}`
- **Group**: `{axis}` (同じ方向の実験をグループ化)
- **Tags**: `[axis, fold_id, model_name, ...]`

#### **4. チェックポイントのメタ情報**
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': ...,
    'metrics': ...,
    # メタ情報
    'axis': 'axial',
    'fold_id': 0,
    'model_name': 'resnet18'
}
```

#### **5. 設定の自動保存**
- 各実験の `config.yaml` を自動保存
- 再現性の確保

### **利点まとめ**

✅ **自動化**: 実験名・ディレクトリ名を自動生成（手動設定不要）
✅ **階層的管理**: 方向別・fold別に明確に分離
✅ **検索性**: ディレクトリ構造で直感的に検索可能
✅ **WandB統合**: グループ・タグで階層的に管理
✅ **再現性**: 使用した設定をconfig.yamlとして保存
✅ **ログ管理**: 各実験のログファイルを個別に保存（タイムスタンプ付き）

### **結果の確認方法**

#### **ディレクトリ構造で確認**
```bash
# Axial方向の全foldを確認
ls A/output/axial/
# -> fold_0  fold_1  fold_2  fold_3  fold_4

# 特定foldのチェックポイントを確認
ls A/output/axial/fold_0/checkpoints/
# -> best_model.pth  last_model.pth

# ログファイルを確認
cat A/output/axial/fold_0/logs/train_*.log
```

#### **WandBで確認**
1. **Group by Axis**: 左サイドバーで "Group" → `group` を選択
2. **Filter by Tags**: `axial`, `fold_0` などでフィルタリング
3. **比較**: 同じ方向の異なるfoldを並べて比較

---

これで**A/フォルダで完全な学習パイプラインを一から構築**する準備が整いました!
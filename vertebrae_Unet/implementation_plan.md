# Attention U-Net 実装計画書

## プロジェクト概要

椎体骨折検出のためのAttention U-Netベースのセグメンテーションモデルの実装

**目的**: axial面スライス画像から骨折領域をピクセルレベルでセグメンテーション

---

## 実装フェーズ

### **Phase 1: 基礎実装 (最優先)** 🎯

Attention U-Net + 単一スライス学習の完全実装

#### 1.1 モデルアーキテクチャ

| ファイル | 内容 | 優先度 |
|---------|------|--------|
| `src/models/attention_gate.py` | Attention Gateモジュール実装 | 🔴 High |
| `src/models/attention_unet.py` | Attention U-Net本体 (Encoder-Decoder) | 🔴 High |
| `src/models/losses.py` | Dice Loss + BCE Loss実装 | 🔴 High |

**技術仕様**:
- **入力**: 単一axialスライス (H×W, 可変サイズ)
- **出力**: 骨折確率マップ (H×W)
- **アーキテクチャ**:
  - Encoder: 4段階ダウンサンプリング (conv→pool)
  - Decoder: 4段階アップサンプリング (upconv→concat→conv)
  - Attention Gate: 各skip connection前に適用
- **損失関数**: `α×Dice Loss + (1-α)×BCE Loss` (α=0.5)

---

#### 1.2 データローダー

| ファイル | 内容 | 優先度 |
|---------|------|--------|
| `src/datamodule/single_slice_dataset.py` | 単一スライスDataset/DataModule | 🔴 High |
| `src/datamodule/transforms.py` | Data Augmentation (optional) | 🟡 Medium |

**データフロー**:
```
CSV読み込み
  ↓
NIfTI画像読み込み (nibabel)
  ↓
HU値正規化 (-1000~3000 → [0, 1])
  ↓
Data Augmentation (回転, スケール, 輝度)
  ↓
Tensor変換 (1×H×W)
```

**必要な処理**:
- ✅ CSVからスライスパス・ラベル読み込み
- ✅ NIfTI画像読み込み (nibabel.load)
- ✅ HU値のクリッピング・正規化
- ✅ Augmentation (RandomRotation, RandomAffine, ColorJitter)
- ✅ Train/Val分割 (K-fold対応)

---

#### 1.3 学習モジュール

| ファイル | 内容 | 優先度 |
|---------|------|--------|
| `src/modelmodule/attention_unet_module.py` | PyTorch Lightning Module | 🔴 High |
| `src/utils/metrics.py` | 評価指標 (Dice, IoU, Precision, Recall) | 🔴 High |

**Lightning Moduleの実装要素**:
- `__init__`: モデル・損失関数・評価指標の初期化
- `forward`: 推論処理
- `training_step`: 学習ステップ (損失計算)
- `validation_step`: 検証ステップ (損失・Dice計算)
- `configure_optimizers`: Adam optimizer + CosineAnnealingLR

**評価指標**:
- **Dice係数**: `2×|X∩Y| / (|X|+|Y|)`
- **IoU**: `|X∩Y| / |X∪Y|`
- **Precision/Recall**: ピクセル単位の分類精度

---

#### 1.4 設定管理 (Hydra)

| ファイル | 内容 | 優先度 |
|---------|------|--------|
| `run/conf/config.yaml` | メイン設定 (defaults指定) | 🔴 High |
| `run/conf/train.yaml` | 学習ハイパーパラメータ | 🔴 High |
| `run/conf/model/attention_unet.yaml` | モデル設定 | 🔴 High |
| `run/conf/dataset/single_slice.yaml` | データセット設定 | 🔴 High |
| `run/conf/dir/local.yaml` | ディレクトリパス定義 | 🔴 High |

**設定項目**:

**train.yaml**:
```yaml
max_epochs: 100
batch_size: 16
learning_rate: 1e-4
num_workers: 4
accelerator: gpu
devices: 1
```

**model/attention_unet.yaml**:
```yaml
in_channels: 1
out_channels: 1
base_channels: 64
depth: 4
loss_alpha: 0.5  # Dice vs BCE weight
```

**dataset/single_slice.yaml**:
```yaml
csv_path: vertebrae_Unet/data/slice_train/axial/
train_ratio: 0.8
augmentation: true
normalize_hu: true
hu_min: -1000
hu_max: 3000
```

---

#### 1.5 学習スクリプト

| ファイル | 内容 | 優先度 |
|---------|------|--------|
| `run/scripts/train/train.py` | メイン学習スクリプト | 🔴 High |

**スクリプト構成**:
```python
# 1. Hydra設定読み込み
# 2. DataModuleインスタンス化
# 3. ModelModuleインスタンス化
# 4. Trainer設定 (callbacks, logger)
# 5. 学習実行 (trainer.fit)
# 6. ベストモデル保存
```

**実装するCallbacks**:
- `ModelCheckpoint`: ベストモデル保存 (Dice最大)
- `EarlyStopping`: 過学習防止
- `LearningRateMonitor`: LR記録
- `WandbLogger`: W&Bロギング

---

### **Phase 2: LSTM拡張** 🔄

5枚連続スライスによる時系列学習

#### 2.1 シーケンスデータセット

| ファイル | 内容 | 優先度 |
|---------|------|--------|
| `src/datamodule/sequence_dataset.py` | 5スライスシーケンスDataset | 🟡 Medium |
| `run/conf/dataset/sequence_5.yaml` | シーケンス設定 | 🟡 Medium |

**シーケンス構築**:
- 中央スライス `t` に対して `[t-2, t-1, t, t+1, t+2]` を取得
- 端部の処理: パディング or スキップ

---

#### 2.2 LSTM統合モデル

| ファイル | 内容 | 優先度 |
|---------|------|--------|
| `src/models/lstm_encoder.py` | U-Net + LSTM Encoder | 🟡 Medium |
| `src/modelmodule/unet_lstm_module.py` | LSTM版Lightning Module | 🟡 Medium |
| `run/conf/model/attention_unet_lstm.yaml` | LSTM設定 | 🟡 Medium |

**アーキテクチャ**:
```
5スライス (5×H×W)
  ↓
U-Net Encoder (各スライス独立)
  ↓
LSTM (時系列特徴統合)
  ↓
U-Net Decoder (中央スライスのみ)
  ↓
セグメンテーションマップ (H×W)
```

---

### **Phase 3: 推論・評価** 📊

学習済みモデルの推論・3D復元・評価

#### 3.1 推論パイプライン

| ファイル | 内容 | 優先度 |
|---------|------|--------|
| `run/scripts/inference/inference.py` | 2Dセグメンテーション推論 | 🟢 Low |
| `run/scripts/inference/reconstruct_3d.py` | 3D復元 (スライス統合) | 🟢 Low |
| `run/conf/inference.yaml` | 推論設定 | 🟢 Low |

**推論フロー**:
```
チェックポイント読み込み
  ↓
テストスライス読み込み
  ↓
モデル推論 (確率マップ出力)
  ↓
2D予測マスク保存 (.nii)
  ↓
3D復元 (椎体ごとに統合)
  ↓
3D評価指標計算
```

---

#### 3.2 可視化

| ファイル | 内容 | 優先度 |
|---------|------|--------|
| `run/scripts/visualization/visualize_heatmap.py` | ヒートマップ可視化 | 🟢 Low |
| `run/scripts/visualization/visualize_attention.py` | Attentionマップ可視化 | 🟢 Low |
| `run/scripts/visualization/visualize_3d.py` | 3Dレンダリング | 🟢 Low |
| `src/utils/visualization.py` | 可視化関数 | 🟢 Low |

**可視化内容**:
- **ヒートマップ**: CT画像 + 骨折確率マップ重畳
- **Attentionマップ**: 各層のAttention重み可視化
- **3Dレンダリング**: 骨折領域の3D表示

---

#### 3.3 評価

| ファイル | 内容 | 優先度 |
|---------|------|--------|
| `run/scripts/utils/evaluate_3d.py` | 3D評価指標計算 | 🟢 Low |
| `run/scripts/utils/combine_metrics.py` | 評価指標統合 | 🟢 Low |
| `src/utils/reconstruction.py` | 3D復元関数 | 🟢 Low |

**評価指標**:
- 2D評価: スライス単位のDice, IoU
- 3D評価: 椎体単位のDice, IoU
- 症例別・椎体別の統計

---

### **Phase 4: GAN拡張 (オプション)** 🚀

敵対的学習による精度向上

| ファイル | 内容 | 優先度 |
|---------|------|--------|
| `src/models/discriminator.py` | PatchGAN Discriminator | ⚪ Optional |
| `src/modelmodule/unet_gan_module.py` | GAN学習Module | ⚪ Optional |
| `run/conf/model/unet_gan.yaml` | GAN設定 | ⚪ Optional |

**GAN構成**:
- **Generator**: Attention U-Net
- **Discriminator**: PatchGAN (70×70)
- **損失関数**: `Dice + BCE + λ×Adversarial Loss`

---

## 実装スケジュール

### **Week 1-2: Phase 1基礎実装**

```
Day 1-2:  モデルアーキテクチャ (Attention U-Net, losses)
Day 3-4:  データローダー (single_slice_dataset)
Day 5-6:  学習モジュール (Lightning Module, metrics)
Day 7-8:  設定管理 (Hydra configs)
Day 9-10: 学習スクリプト (train.py)
Day 11-14: デバッグ・初期学習実験
```

### **Week 3: Phase 1検証**

```
- 学習曲線の確認
- Dice/IoU評価
- ハイパーパラメータチューニング
- クロスバリデーション
```

### **Week 4: Phase 2 LSTM拡張**

```
Day 1-3: シーケンスデータセット実装
Day 4-6: LSTM統合モデル実装
Day 7: LSTM学習実験
```

### **Week 5-6: Phase 3 推論・評価**

```
Week 5: 推論パイプライン・3D復元
Week 6: 可視化・評価スクリプト
```

### **Week 7: Phase 4 GAN (オプション)**

---

## 技術スタック

| カテゴリ | ツール/ライブラリ |
|---------|------------------|
| 深層学習 | PyTorch, PyTorch Lightning |
| 設定管理 | Hydra |
| 実験管理 | Weights & Biases |
| 医療画像 | nibabel, SimpleITK |
| データ処理 | NumPy, Pandas |
| 可視化 | Matplotlib, seaborn |

---

## 成功基準

### **Phase 1完了条件**

- ✅ Attention U-Netが学習可能
- ✅ 学習曲線が収束
- ✅ Validation Dice > 0.6
- ✅ W&Bにログが正常に記録
- ✅ チェックポイントが保存される

### **Phase 2完了条件**

- ✅ 5スライスシーケンスが正常に読み込める
- ✅ LSTM統合モデルが学習可能
- ✅ Validation Dice > Phase 1

### **Phase 3完了条件**

- ✅ テストデータで推論が動作
- ✅ 3D復元が正常に動作
- ✅ 可視化結果が妥当

---

## リスクと対策

| リスク | 対策 |
|--------|------|
| クラス不均衡 (骨折症例が少ない) | Weighted Loss, Focal Loss導入 |
| メモリ不足 | バッチサイズ削減, Gradient Accumulation |
| 過学習 | Dropout, Data Augmentation, Early Stopping |
| 学習が収束しない | LR調整, Loss関数見直し |
| 3D復元の精度が低い | アンサンブル, Post-processing |

---

## 参考文献

- **Attention U-Net**: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas" (2018)
- **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- **Dice Loss**: Milletari et al., "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation" (2016)

---

## 次のアクション

**Phase 1実装開始**:
1. `src/models/attention_gate.py` 実装
2. `src/models/attention_unet.py` 実装
3. `src/models/losses.py` 実装
4. `src/datamodule/single_slice_dataset.py` 実装
5. `src/modelmodule/attention_unet_module.py` 実装
6. Hydra設定ファイル作成
7. `run/scripts/train/train.py` 実装
8. デバッグ実行

---

**最終更新**: 2025-10-08
**ステータス**: Phase 1実装開始準備完了

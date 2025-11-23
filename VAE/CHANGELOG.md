# Changelog

## 2025-11-23 (2) - Data Augmentation機能追加

### 追加内容
z軸回りの回転とxy方向の平行移動をData Augmentationに追加。

### 追加機能

#### 1. z軸回りの回転
- **設定パラメータ**: `rotation_z`, `rotation_z_range`
- **回転範囲**: -15° ~ 15° (デフォルト)
- **実装**: `_rotate_volume_z()` メソッド
- **動作**: 各z軸スライスに対して2D回転を適用

#### 2. xy方向の平行移動
- **設定パラメータ**: `translation_xy`, `translation_xy_percent`
- **移動範囲**: ボリュームサイズの5% (デフォルト)
- **実装**: `_translate_volume_xy()` メソッド
- **動作**: 各z軸スライスに対して2D平行移動を適用

### 修正ファイル

**[run/conf/dataset/vae_data.yaml](run/conf/dataset/vae_data.yaml)**
```yaml
augmentation:
  rotation_z: true               # z軸回りの回転
  rotation_z_range: [-15, 15]    # 回転範囲 (度)
  translation_xy: true           # xy方向の平行移動
  translation_xy_percent: 0.05   # 移動範囲 (5%)
```

**[src/datamodule/dataset.py](src/datamodule/dataset.py)**
- `_apply_augmentation()`: z軸回転・xy平行移動の処理追加
- `_rotate_volume_z()`: 新規メソッド (z軸回転)
- `_translate_volume_xy()`: 新規メソッド (xy平行移動)

**ドキュメント**
- [README.md](README.md): Augmentation設定例を更新
- [context.md](context.md): Augmentation一覧を更新

### 技術詳細

両機能とも`torch.nn.functional.affine_grid`と`grid_sample`を使用してアフィン変換を実現:
- **補間**: bilinear
- **パディング**: zeros (範囲外は0埋め)
- **適用確率**: 50% (各Augmentationごと)

### 使用例

```bash
# デフォルト設定で実行
python train_vae.py fold_id=1

# 回転範囲を変更
python train_vae.py fold_id=1 dataset.augmentation.rotation_z_range='[-10,10]'

# 平行移動範囲を変更
python train_vae.py fold_id=1 dataset.augmentation.translation_xy_percent=0.1
```

---

## 2025-11-23 (1) - ディレクトリ名変更対応

### 変更内容
`run/conf/data/` → `run/conf/dataset/` へのディレクトリ名変更に伴う修正を実施。

### 修正ファイル一覧

#### 1. 設定ファイル

**[run/conf/config.yaml](run/conf/config.yaml)**
```diff
defaults:
  - model: vq_vae
- - data: vae_data
+ - dataset: vae_data
  - training: vae_training
```

**[run/conf/config_debug.yaml](run/conf/config_debug.yaml)**
```diff
defaults:
  - model: vq_vae
- - data: vae_data
+ - dataset: vae_data
  - training: vae_training
  - _self_

# Debug用のオーバーライド
training:
  max_epochs: 3
  early_stopping:
    patience: 5

-data:
+dataset:
  batch_size: 2
  num_workers: 2
```

#### 2. 学習スクリプト

**[run/scripts/train_vae.py](run/scripts/train_vae.py)**
```diff
# Usage例
-    python train_vae.py fold_id=1 training.max_epochs=5 data.batch_size=2
+    python train_vae.py fold_id=1 training.max_epochs=5 dataset.batch_size=2

# DataModule初期化
datamodule = VertebraeVAEDataModule(
-    data_dir=cfg.data.data_dir,
-    batch_size=cfg.data.batch_size,
-    num_workers=cfg.data.num_workers,
+    data_dir=cfg.dataset.data_dir,
+    batch_size=cfg.dataset.batch_size,
+    num_workers=cfg.dataset.num_workers,
    fold_id=fold_id,
-    augmentation=OmegaConf.to_container(cfg.data.augmentation, resolve=True),
-    pin_memory=cfg.data.pin_memory,
+    augmentation=OmegaConf.to_container(cfg.dataset.augmentation, resolve=True),
+    pin_memory=cfg.dataset.pin_memory,
)
```

#### 3. ドキュメント

**[README.md](README.md)**
```diff
# デバッグモード
-python train_vae.py fold_id=1 training.max_epochs=5 data.batch_size=2
+python train_vae.py fold_id=1 training.max_epochs=5 dataset.batch_size=2

# バッチサイズ変更
-python train_vae.py fold_id=1 data.batch_size=8
+python train_vae.py fold_id=1 dataset.batch_size=8

# データ設定パス
-### データ設定 ([run/conf/data/vae_data.yaml](run/conf/data/vae_data.yaml))
+### データ設定 ([run/conf/dataset/vae_data.yaml](run/conf/dataset/vae_data.yaml))

# トラブルシューティング - メモリ不足
-python train_vae.py fold_id=1 data.batch_size=2
+python train_vae.py fold_id=1 dataset.batch_size=2
```

**[context.md](context.md)**
```diff
#### Hydra設定構造
-4. ✓ [data/vae_data.yaml](run/conf/data/vae_data.yaml) - データ・Augmentation設定
+4. ✓ [dataset/vae_data.yaml](run/conf/dataset/vae_data.yaml) - データ・Augmentation設定
```

### 影響範囲

✅ **修正完了**
- Hydra設定ファイル (config.yaml, config_debug.yaml)
- 学習スクリプト (train_vae.py)
- ドキュメント (README.md, context.md)

❌ **影響なし**
- モデル実装 (src/models/)
- データモジュール実装 (src/datamodule/)
- Lightning Module実装 (src/training/)
- データ設定ファイル本体 (dataset/vae_data.yaml) - 内容変更なし

### 動作確認方法

```bash
cd /mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/VAE/run/scripts

# 設定ファイルの構文チェック
python -c "from omegaconf import OmegaConf; print(OmegaConf.load('../conf/config.yaml'))"

# デバッグモードで実行
python train_vae.py --config-name=config_debug fold_id=1
```

### 使用方法の変更点まとめ

**変更前:**
```bash
python train_vae.py fold_id=1 data.batch_size=4
```

**変更後:**
```bash
python train_vae.py fold_id=1 dataset.batch_size=4
```

**その他の使用例:**
```bash
# バッチサイズ変更
python train_vae.py fold_id=1 dataset.batch_size=8

# データディレクトリ変更
python train_vae.py fold_id=1 dataset.data_dir=/path/to/data

# Augmentation設定変更
python train_vae.py fold_id=1 dataset.augmentation.horizontal_flip=false
```

---

**注意:** ディレクトリ名は`dataset`に変更されましたが、設定ファイル内の変数名(`data_dir`, `batch_size`等)は変更されていません。Hydra経由でアクセスする際のプレフィックスのみが`cfg.data.*` → `cfg.dataset.*`に変更されました。

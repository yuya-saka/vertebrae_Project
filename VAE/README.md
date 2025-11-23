# 3D VQ-VAE for Vertebral Fracture Detection

## 概要

正常椎体データのみを用いて3D VQ-VAE (Vector Quantized Variational Autoencoder)を学習し、骨折検出のための再構成誤差マップを生成するプロジェクト。

## ディレクトリ構成

```
VAE/
├── src/
│   ├── models/              # VQ-VAEモデル実装
│   ├── datamodule/          # データローディング (fold分割対応)
│   └── training/            # Lightning Module
├── run/
│   ├── conf/                # Hydra設定ファイル
│   └── scripts/             # 学習スクリプト
├── outputs/                 # 学習結果保存先
├── PLAN.md                  # 研究計画
├── fold_plan.md             # Fold分割の詳細
└── data.md                  # データセット情報
```

## セットアップ

### 必要なパッケージ

```bash
pip install torch torchvision pytorch-lightning hydra-core wandb
pip install numpy nibabel
```

## 使用方法

### 1つのFoldで学習

```bash
cd /mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/VAE/run/scripts
python train_vae.py fold_id=1
```

### デバッグモード (少ないepoch)

```bash
python train_vae.py fold_id=1 training.max_epochs=3 dataset.batch_size=2 experiment.name=vqvae_debag
```

### 実験名を変更

```bash
python train_vae.py fold_id=1 experiment.name=vqvae_exp1
```

### 全Fold (1-5) で学習

```bash
for i in {1..5}; do python train_vae.py fold_id=$i; done
```

### モデルパラメータの変更

```bash
# コードブックサイズを変更
python train_vae.py fold_id=1 model.num_embeddings=1024

# 学習率を変更
python train_vae.py fold_id=1 training.learning_rate=5e-5

# バッチサイズを変更
python train_vae.py fold_id=1 dataset.batch_size=8
```

## 学習の監視

### WandB

学習中の指標はWandBに自動的にロギングされます:

- プロジェクト名: `vertebrae-vqvae`
- Run名: `{experiment_name}_fold{fold_id}`

監視する主な指標:
- `train/recon_loss`: 訓練再構成Loss
- `val/recon_loss`: 検証再構成Loss
- `train/vq_loss`: Vector Quantization Loss
- `train/perplexity`: コードブック使用多様性
- `train/codebook_usage`: コードブック使用率

### ローカルファイル

学習結果は以下に保存されます:
```
outputs/{experiment_name}/fold_{fold_id}/
├── config.yaml                    # 使用した設定
├── checkpoints/                   # モデルチェックポイント
│   ├── vqvae-epoch=XXX-val_total_loss=X.XXXX.ckpt
│   └── last.ckpt
└── wandb/                         # WandBログ
```

## データ要件

- データディレクトリ: `/mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/data/3d_data/train_vae/`
- フォーマット: `.npy` (128³ボリューム, float32, [0,1]正規化済み)
- ファイル名形式: `vol_{patient_id}_{vertebra}.npy` (例: `vol_1003_T4.npy`)

## Fold分割

5-Fold Cross Validationを使用:
- 各Fold: 6症例 (検証用)
- 学習データ: 24症例 (正常椎体 約180個)
- 患者レベルでの分割 (データリーケージ防止)

詳細は [fold_plan.md](fold_plan.md) を参照。

## 設定ファイルの編集

### モデル設定 ([run/conf/model/vq_vae.yaml](run/conf/model/vq_vae.yaml))

```yaml
latent_dim: 256                 # 潜在表現の次元
num_embeddings: 512             # コードブックサイズ
commitment_cost: 0.25           # Commitment Loss係数
dropout: 0.1                    # Dropout確率
```

### データ設定 ([run/conf/dataset/vae_data.yaml](run/conf/dataset/vae_data.yaml))

```yaml
batch_size: 4
num_workers: 4
augmentation:
  horizontal_flip: true      # 左右反転
  rotation_z: true           # z軸回りの回転 (-15°~15°)
  translation_xy: true       # xy方向の平行移動 (5%)
  scale: true                # スケーリング
  gaussian_noise: true       # ガウスノイズ
  brightness: true           # 輝度調整
  contrast: true             # コントラスト調整
  # ...
```

### 学習設定 ([run/conf/training/vae_training.yaml](run/conf/training/vae_training.yaml))

```yaml
max_epochs: 200
learning_rate: 1e-4
optimizer: adamw
early_stopping:
  patience: 20
```

## トラブルシューティング

### メモリ不足

```bash
# バッチサイズを減らす
python train_vae.py fold_id=1 dataset.batch_size=2

# Precision を32に変更
python train_vae.py fold_id=1 precision=32
```

### 学習が不安定

```bash
# 学習率を下げる
python train_vae.py fold_id=1 training.learning_rate=5e-5

# Commitment costを調整
python train_vae.py fold_id=1 model.commitment_cost=0.5
```

## 次のステップ

1. **再構成誤差マップの生成**: 学習済みVQVAEで全データの再構成誤差を計算
2. **骨折検出モデルの学習**: 再構成誤差マップ + 弱ラベルでU-Net等を学習
3. **評価**: Testデータ(8症例)で最終評価

## 参考資料

- [PLAN.md](PLAN.md): 研究全体の計画
- [fold_plan.md](fold_plan.md): Fold分割の詳細
- [data.md](data.md): データセット情報

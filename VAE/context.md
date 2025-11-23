# VAE プロジェクト 実装進捗

最終更新日: 2025-11-23

## 実装完了事項

### ✅ Phase 1: データ準備モジュール

#### 1. 3D Dataset実装 ([src/datamodule/dataset.py](src/datamodule/dataset.py))
- ✓ `.npy`形式の3Dボリューム(128³)読み込み
- ✓ 正常データのみフィルタリング機能
- ✓ 包括的な3D Data Augmentation実装:
  - 左右反転
  - z軸回りの回転 (-15°~15°)
  - xy方向の平行移動 (5%)
  - ランダムスケーリング (0.9-1.1)
  - ガウスノイズ付加
  - 輝度・コントラスト調整
- ✓ 患者IDベースのファイル管理

#### 2. DataModule実装 ([src/datamodule/dataloader.py](src/datamodule/dataloader.py))
- ✓ PyTorch Lightning DataModule
- ✓ 5-Fold Cross Validation対応
- ✓ fold_plan.mdに基づく患者レベル分割
- ✓ FOLD_DEFINITION辞書による厳密なFold管理
- ✓ データリーケージ防止機能

### ✅ Phase 2: モデル実装

#### 1. Vector Quantizer層 ([src/models/vector_quantizer.py](src/models/vector_quantizer.py))
- ✓ コードブック学習 (embedding_dim × num_embeddings)
- ✓ Commitment Loss実装
- ✓ Straight-Through Estimator (STE)
- ✓ Exponential Moving Average (EMA) オプション
- ✓ Perplexity計算 (コードブック使用多様性)
- ✓ コードブック使用率トラッキング

#### 2. 3D VQ-VAE本体 ([src/models/vq_vae_3d.py](src/models/vq_vae_3d.py))
- ✓ 3D Encoder実装:
  - 入力: (B, 1, 128, 128, 128)
  - 4層ダウンサンプリング: [32, 64, 128, 256]チャネル
  - BatchNorm + LeakyReLU + Dropout
  - 出力: (B, 256, 8, 8, 8) 潜在表現
- ✓ 3D Decoder実装:
  - Encoderの対称構造
  - 4層アップサンプリング
  - 最終層: Sigmoid活性化 ([0,1]範囲)
- ✓ 再構成誤差マップ生成機能 (骨折検出用)
- ✓ モデルビルダー関数

### ✅ Phase 3: 学習モジュール

#### Lightning Module ([src/training/lightning_module.py](src/training/lightning_module.py))
- ✓ 学習ループ実装:
  - 再構成Loss (L1/L2選択可能)
  - VQ Loss統合
  - 総合Loss = recon_loss + vq_loss
- ✓ オプティマイザ設定 (Adam/AdamW)
- ✓ スケジューラ設定:
  - CosineAnnealingLR
  - ReduceLROnPlateau
- ✓ WandBロギング:
  - train/val loss各種
  - Perplexity, コードブック使用率
  - 再構成画像サンプル (epoch毎)
- ✓ 勾配クリッピング

### ✅ Phase 4: 設定ファイル

#### Hydra設定構造
1. ✓ [config.yaml](run/conf/config.yaml) - メイン設定
2. ✓ [config_debug.yaml](run/conf/config_debug.yaml) - デバッグ用設定
3. ✓ [model/vq_vae.yaml](run/conf/model/vq_vae.yaml) - モデルハイパーパラメータ
4. ✓ [dataset/vae_data.yaml](run/conf/dataset/vae_data.yaml) - データ・Augmentation設定
5. ✓ [training/vae_training.yaml](run/conf/training/vae_training.yaml) - 学習設定

#### 設定のハイライト
```yaml
# モデル
latent_dim: 256
num_embeddings: 512
commitment_cost: 0.25

# 学習
max_epochs: 200
learning_rate: 1e-4
batch_size: 4
early_stopping_patience: 20

# Augmentation
- 左右反転、z軸回転(-15°~15°)、xy平行移動(5%)
- スケーリング、ガウスノイズ、輝度・コントラスト調整
```

### ✅ Phase 5: 学習スクリプト

#### [run/scripts/train_vae.py](run/scripts/train_vae.py)
- ✓ Hydra統合
- ✓ 5-Fold CV対応
- ✓ WandBロギング (fold別)
- ✓ Model Checkpoint & Early Stopping
- ✓ 設定の自動保存

#### 使用例
```bash
# 1つのFoldで学習
python train_vae.py fold_id=1

# デバッグモード
python train_vae.py --config-name=config_debug fold_id=1

# 全Fold実行
for i in {1..5}; do python train_vae.py fold_id=$i; done
```

### ✅ ドキュメント

1. ✓ [README.md](README.md) - プロジェクト概要、使用方法
2. ✓ [PLAN.md](PLAN.md) - 研究全体の計画 (既存)
3. ✓ [fold_plan.md](fold_plan.md) - Fold分割詳細 (既存)
4. ✓ [data.md](data.md) - データセット情報 (既存)
5. ✓ [context.md](context.md) - このファイル (実装進捗)
6. ✓ [test_installation.py](test_installation.py) - 動作確認スクリプト

## プロジェクト構造

```
VAE/
├── src/
│   ├── models/              # VQ-VAEモデル
│   │   ├── vector_quantizer.py
│   │   └── vq_vae_3d.py
│   ├── datamodule/          # データローディング
│   │   ├── dataset.py
│   │   └── dataloader.py
│   └── training/            # Lightning Module
│       └── lightning_module.py
├── run/
│   ├── conf/                # Hydra設定
│   │   ├── config.yaml
│   │   ├── config_debug.yaml
│   │   ├── model/vq_vae.yaml
│   │   ├── data/vae_data.yaml
│   │   └── training/vae_training.yaml
│   └── scripts/
│       └── train_vae.py     # 学習スクリプト
├── outputs/                 # 学習結果
└── ドキュメント各種
```

## 技術スタック

- **PyTorch**: 深層学習フレームワーク
- **PyTorch Lightning**: 学習ループ抽象化
- **Hydra**: 設定管理
- **WandB**: 実験トラッキング
- **NumPy**: 数値計算

## データフロー

1. **入力**: 正常椎体の3Dボリューム (128³, .npy, [0,1]正規化済み)
2. **Augmentation**: 回転、反転、スケーリング、ノイズ等
3. **Encoder**: (B, 1, 128³) → (B, 256, 8³)
4. **Vector Quantizer**: 潜在表現の離散化
5. **Decoder**: (B, 256, 8³) → (B, 1, 128³) 再構成
6. **Loss**: reconstruction + vq_loss

## Fold分割の詳細

- **Train**: 30症例 (5 Fold)
- **Test**: 8症例 (Hold-out、最終評価用)
- **Fold毎の学習データ**: 約180個の正常椎体
- **Fold毎の検証データ**: 約45個の正常椎体

各Foldの患者ID割り当ては[dataloader.py](src/datamodule/dataloader.py)の`FOLD_DEFINITION`参照。

## WandB指標

### 学習時
- `train/recon_loss`: 訓練再構成Loss (L1 or L2)
- `train/vq_loss`: Vector Quantization Loss
- `train/total_loss`: 総合Loss
- `train/perplexity`: コードブック使用多様性
- `train/codebook_usage`: コードブック使用率 (0-1)
- `train/learning_rate`: 学習率

### 検証時
- `val/recon_loss`: 検証再構成Loss
- `val/vq_loss`: 検証VQ Loss
- `val/total_loss`: 検証総合Loss
- `val/perplexity`: 検証Perplexity
- `val/original_slice_epochX`: 元画像の中央スライス
- `val/recon_slice_epochX`: 再構成画像の中央スライス
- `val/error_slice_epochX`: 再構成誤差マップ

## 次のステップ

### 🔄 実装予定 (優先度順)

1. **学習実行とハイパーパラメータ調整**
   - [ ] Fold 1でデバッグ実行
   - [ ] ハイパーパラメータチューニング
   - [ ] 全Fold (1-5) での学習

2. **再構成誤差マップ生成スクリプト**
   - [ ] 学習済みVQVAEで全データの再構成誤差を計算
   - [ ] `.npy`形式で保存 (骨折検出モデルの入力用)

3. **骨折検出モデル (Phase 2)**
   - [ ] 3D U-Netアーキテクチャ設計
   - [ ] 弱教師ありLoss関数実装 (L1, L2, L3, L4)
   - [ ] 再構成誤差マップ + 弱ラベルでの学習

4. **評価・可視化**
   - [ ] Hold-out Testデータでの最終評価
   - [ ] 再構成誤差の統計分析
   - [ ] 3D可視化ツール

## 既知の制約・注意事項

1. **メモリ使用量**
   - 3Dボリューム (128³) × バッチサイズ4 → 約8GB GPU RAM
   - メモリ不足時はbatch_size=2またはprecision=16推奨

2. **学習時間**
   - 1 epoch: 約5-10分 (GPU依存)
   - 全学習 (200 epochs): 約16-32時間
   - 5-Fold CV全体: 約80-160時間

3. **データ量の限界**
   - 各Foldの学習データ: 約180個 (小規模)
   - Augmentation強化が必須
   - Early Stopping推奨 (過学習防止)

4. **コードブック崩壊リスク**
   - num_embeddingsが大きすぎると一部しか使われない
   - Perplexity/Usage監視が重要
   - 初期値: 512 (経験的に適切)

## トラブルシューティング

### Q. VQ Lossが減少しない
- A. commitment_costを調整 (0.1-0.5)
- A. num_embeddingsを減らす (256-512)

### Q. 再構成が不鮮明
- A. latent_dimを増やす (256→512)
- A. hidden_dimsを深くする

### Q. Perplexityが低い (コードブック使用率低)
- A. num_embeddingsを減らす
- A. EMAを有効化 (use_ema=True)

## 参考文献・リンク

- [VQ-VAE論文](https://arxiv.org/abs/1711.00937)
- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)
- [Hydra Docs](https://hydra.cc/)

## 変更履歴

### 2025-11-23
- ✅ 初回実装完了
- ✅ 全モジュール・設定ファイル作成
- ✅ ドキュメント整備
- 🔄 学習実行待ち

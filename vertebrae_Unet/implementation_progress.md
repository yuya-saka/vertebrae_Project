# Attention U-Net 実装進捗管理

## 実装を進める度に、更新させる

**プロジェクト**: 椎体骨折セグメンテーション
**開始日**: 2025-10-08
**最終更新**: 2025-10-08

---

## 全体進捗サマリー

| Phase | タスク | ステータス | 進捗率 |
|-------|--------|-----------|--------|
| **Phase 0** | データ前処理 | ✅ 完了 | 100% |
| **Phase 1** | 基礎実装 | ⏳ 未着手 | 0% |
| **Phase 2** | LSTM拡張 | ⏳ 未着手 | 0% |
| **Phase 3** | 推論・評価 | ⏳ 未着手 | 0% |
| **Phase 4** | GAN拡張 | ⏳ 未着手 | 0% |

**現在のフェーズ**: Phase 1 (基礎実装)

---

## Phase 0: データ前処理 ✅

### 完了済みタスク

| タスク | ファイル | 完了日 | 備考 |
|--------|---------|--------|------|
| ✅ データ分割 | `data_preprocessing/data_pationing.py` | 完了 | Train:30, Test:8 |
| ✅ 椎体領域切り出し (Train) | `data_preprocessing/volume_cut/cut_train.py` | 完了 | T4-L5 (27-40) |
| ✅ 椎体領域切り出し (Test) | `data_preprocessing/volume_cut/cut_test.py` | 完了 | |
| ✅ Axialスライス作成 (Train) | `data_preprocessing/slice_data/slice_train_axial.py` | 完了 | |
| ✅ Axialスライス作成 (Test) | `data_preprocessing/slice_data/slice_test_axial.py` | 完了 | |

### データ準備状況

**訓練データ**: 30症例
- スライス画像: `vertebrae_Unet/data/slice_train/axial/`
- ラベルCSV: `fracture_labels_inp{症例番号}.csv`
- 椎体範囲: T4-L5 (番号27-40)

**テストデータ**: 8症例
- スライス画像: `vertebrae_Unet/data/slice_test/axial/`
- ラベルCSV: `fracture_labels_inp{症例番号}.csv`

**CSVフォーマット**:
```
FullPath, Vertebra, SliceIndex, Fracture_Label, Case, Axis, CT_H, CT_W, CT_D, InputCTPath
```

---

## Phase 1: 基礎実装 (Attention U-Net + 単一スライス)

**目標**: 単一axialスライスからの骨折セグメンテーション

### 1.1 モデルアーキテクチャ

| タスク | ファイル | ステータス | 担当 | 備考 |
|--------|---------|-----------|------|------|
| ⏳ Attention Gateモジュール | `src/models/attention_gate.py` | 未着手 | - | |
| ⏳ Attention U-Net本体 | `src/models/attention_unet.py` | 未着手 | - | Encoder-Decoder |
| ⏳ 損失関数 | `src/models/losses.py` | 未着手 | - | Dice + BCE |

**実装仕様**:
- 入力: 1×H×W (単一チャンネル)
- 出力: 1×H×W (骨折確率マップ)
- Depth: 4 (ダウンサンプリング段数)
- Base channels: 64

---

### 1.2 データローダー

| タスク | ファイル | ステータス | 担当 | 備考 |
|--------|---------|-----------|------|------|
| ⏳ 単一スライスDataset | `src/datamodule/single_slice_dataset.py` | 未着手 | - | CSV読み込み |
| ⏳ DataModule | `src/datamodule/single_slice_dataset.py` | 未着手 | - | Train/Val分割 |
| ⏳ Data Augmentation | `src/datamodule/transforms.py` | 未着手 | - | オプション |

**必要な処理**:
- [ ] CSV読み込み (pandas)
- [ ] NIfTI画像読み込み (nibabel)
- [ ] HU値正規化 (-1000~3000 → [0,1])
- [ ] Data Augmentation (回転, スケール, 輝度)
- [ ] Train/Val分割 (K-fold対応)

---

### 1.3 学習モジュール

| タスク | ファイル | ステータス | 担当 | 備考 |
|--------|---------|-----------|------|------|
| ⏳ Lightning Module | `src/modelmodule/attention_unet_module.py` | 未着手 | - | 学習・検証ループ |
| ⏳ 評価指標 | `src/utils/metrics.py` | 未着手 | - | Dice, IoU, Precision, Recall |

**Lightning Module実装項目**:
- [ ] `__init__`: モデル・損失関数初期化
- [ ] `forward`: 推論処理
- [ ] `training_step`: 学習ステップ
- [ ] `validation_step`: 検証ステップ
- [ ] `configure_optimizers`: Adam + CosineAnnealingLR

---

### 1.4 設定管理 (Hydra)

| タスク | ファイル | ステータス | 担当 | 備考 |
|--------|---------|-----------|------|------|
| ⏳ メイン設定 | `run/conf/config.yaml` | 未着手 | - | defaults指定 |
| ⏳ 学習設定 | `run/conf/train.yaml` | 未着手 | - | epoch, batch_size, lr |
| ⏳ モデル設定 | `run/conf/model/attention_unet.yaml` | 未着手 | - | channels, depth |
| ⏳ データセット設定 | `run/conf/dataset/single_slice.yaml` | 未着手 | - | CSV path, augmentation |
| ⏳ ディレクトリ設定 | `run/conf/dir/local.yaml` | 未着手 | - | パス定義 |

**設定項目**:
- [ ] 学習ハイパーパラメータ
- [ ] モデルアーキテクチャパラメータ
- [ ] データ拡張パラメータ
- [ ] ディレクトリパス

---

### 1.5 学習スクリプト

| タスク | ファイル | ステータス | 担当 | 備考 |
|--------|---------|-----------|------|------|
| ⏳ メイン学習スクリプト | `run/scripts/train/train.py` | 未着手 | - | Hydra統合 |
| ⏳ バッチ学習スクリプト | `run/scripts/train/run_train.py` | 未着手 | - | オプション |

**実装項目**:
- [ ] Hydra設定読み込み
- [ ] DataModuleインスタンス化
- [ ] ModelModuleインスタンス化
- [ ] Trainerセットアップ (callbacks, logger)
- [ ] 学習実行 (trainer.fit)
- [ ] ベストモデル保存

**Callbacks**:
- [ ] ModelCheckpoint (Dice最大で保存)
- [ ] EarlyStopping
- [ ] LearningRateMonitor
- [ ] WandbLogger

---

### Phase 1 進捗状況

**全体進捗**: 0% (0/10タスク完了)

**次のアクション**:
1. `src/models/attention_gate.py` 実装
2. `src/models/attention_unet.py` 実装
3. `src/models/losses.py` 実装

---

## Phase 2: LSTM拡張

**目標**: 5枚連続スライスによる時系列学習

### 2.1 シーケンスデータセット

| タスク | ファイル | ステータス | 担当 | 備考 |
|--------|---------|-----------|------|------|
| ⏳ シーケンスDataset | `src/datamodule/sequence_dataset.py` | 未着手 | - | [t-2, t-1, t, t+1, t+2] |
| ⏳ シーケンス設定 | `run/conf/dataset/sequence_5.yaml` | 未着手 | - | |

---

### 2.2 LSTM統合モデル

| タスク | ファイル | ステータス | 担当 | 備考 |
|--------|---------|-----------|------|------|
| ⏳ LSTM Encoder | `src/models/lstm_encoder.py` | 未着手 | - | U-Net + LSTM |
| ⏳ LSTM Module | `src/modelmodule/unet_lstm_module.py` | 未着手 | - | Lightning Module |
| ⏳ LSTM設定 | `run/conf/model/attention_unet_lstm.yaml` | 未着手 | - | |

**Phase 2 進捗**: 0% (Phase 1完了後に開始)

---

## Phase 3: 推論・評価

**目標**: 学習済みモデルの推論・3D復元・評価

### 3.1 推論パイプライン

| タスク | ファイル | ステータス | 担当 | 備考 |
|--------|---------|-----------|------|------|
| ⏳ 2D推論 | `run/scripts/inference/inference.py` | 未着手 | - | |
| ⏳ 3D復元 | `run/scripts/inference/reconstruct_3d.py` | 未着手 | - | スライス統合 |
| ⏳ 推論設定 | `run/conf/inference.yaml` | 未着手 | - | |

---

### 3.2 可視化

| タスク | ファイル | ステータス | 担当 | 備考 |
|--------|---------|-----------|------|------|
| ⏳ ヒートマップ可視化 | `run/scripts/visualization/visualize_heatmap.py` | 未着手 | - | |
| ⏳ Attention可視化 | `run/scripts/visualization/visualize_attention.py` | 未着手 | - | |
| ⏳ 3D可視化 | `run/scripts/visualization/visualize_3d.py` | 未着手 | - | |
| ⏳ 可視化関数 | `src/utils/visualization.py` | 未着手 | - | |

---

### 3.3 評価

| タスク | ファイル | ステータス | 担当 | 備考 |
|--------|---------|-----------|------|------|
| ⏳ 3D評価 | `run/scripts/utils/evaluate_3d.py` | 未着手 | - | |
| ⏳ 評価統合 | `run/scripts/utils/combine_metrics.py` | 未着手 | - | |
| ⏳ 3D復元関数 | `src/utils/reconstruction.py` | 未着手 | - | |

**Phase 3 進捗**: 0% (Phase 1完了後に開始)

---

## Phase 4: GAN拡張 (オプション)

**目標**: 敵対的学習による精度向上

| タスク | ファイル | ステータス | 担当 | 備考 |
|--------|---------|-----------|------|------|
| ⏳ Discriminator | `src/models/discriminator.py` | 未着手 | - | PatchGAN |
| ⏳ GAN Module | `src/modelmodule/unet_gan_module.py` | 未着手 | - | |
| ⏳ GAN設定 | `run/conf/model/unet_gan.yaml` | 未着手 | - | |

**Phase 4 進捗**: 0% (オプション実装)

---

## 実験記録

### Experiment Log

| 実験No | 日付 | モデル | データセット | Dice (Val) | IoU (Val) | 備考 |
|--------|------|--------|-------------|-----------|----------|------|
| - | - | - | - | - | - | 未実施 |

---

## 技術的課題・解決策

### 現在の課題

| 課題 | 優先度 | ステータス | 解決策 |
|------|--------|-----------|--------|
| - | - | - | - |

### 解決済み課題

| 課題 | 解決日 | 解決策 |
|------|--------|--------|
| - | - | - |

---

## メモ・備考

### 2025-10-08
- プロジェクト構造設計完了
- データ前処理完了 (axialスライス作成済み)
- Phase 1実装計画策定
- 次: モデル実装開始

---

## チェックリスト (Phase 1完了条件)

- [ ] Attention U-Netが正常に学習できる
- [ ] 学習曲線が収束する
- [ ] Validation Dice > 0.6
- [ ] W&Bに学習ログが記録される
- [ ] チェックポイントが保存される
- [ ] 推論が動作する

---

## 参考リンク

- [実装計画書](implementation_plan.md)
- [プロジェクト構造](project.md)
- [データ仕様](vertebrae.md)
- [README](README.md)

---

**凡例**:
- ✅ 完了
- 🚧 実装中
- ⏳ 未着手
- ⚠️ ブロック中
- ❌ 中止

**最終更新**: 2025-10-08
**次回更新予定**: Phase 1実装開始時

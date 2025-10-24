# YOLOv8 椎体骨折検出プロジェクト

YOLOv8を用いた椎体（胸椎・腰椎）骨折の自動検出システム

## 📋 プロジェクト概要

- **目的**: 3D-CT画像から椎体骨折を高精度に自動検出
- **手法**: YOLOv8 + LSTM（時系列統合）
- **データ**: 30症例、5-fold交差検証
- **特徴**: 3チャンネルHUウィンドウ処理（Bone/Soft Tissue/Wide Window）
- **実装**: シンプルなPyTorchによる学習ループ

## 🚀 セットアップ

### 1. 環境構築

**uv環境を使用する場合（推奨）:**
```bash
cd vertebrae_YOLO

# uvで依存関係をインストール
uv sync
```

**通常のPython環境を使用する場合:**
```bash
cd vertebrae_YOLO

# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 依存関係のインストール
pip install -r requirements.txt
```

### 2. データ準備

```bash
# YOLO形式データは既に変換済み
# data/yolo_format/images/axial/train/*.nii
# data/yolo_format/labels/axial/train/*.txt
```

## 📂 ディレクトリ構造

```
vertebrae_YOLO/
├── data_preparing/
│   └── convert_to_yolo.py            # マスク→YOLO形式変換（完了）
├── src/
│   ├── models/
│   │   ├── yolo_baseline.py          # YOLOv8ベースラインモデル
│   │   └── yolo_lstm.py              # YOLO+LSTMモデル（将来実装）
│   ├── dataset/
│   │   └── yolo_dataset.py           # PyTorch Dataset（3チャンネルHU処理）
│   └── utils/
│       ├── trainer.py                # 学習ユーティリティ
│       └── metrics.py                # 評価指標（mAP計算）
├── run/
│   ├── conf/
│   │   ├── config.yaml               # メイン設定
|   |   ├── hyp_custom.yaml           # データ拡張と、損失関数パラメタ設計
│   │   ├── model/
│   │   │   ├── yolo_baseline.yaml    # ベースラインモデル設定
│   │   │   └── yolo_lstm.yaml        # LSTM統合モデル設定
│   │   ├── data/yolo_data.yaml       # データ設定（HU設定含む）
│   │   ├── logging/wandb.yaml        # W&B設定
│   │   └── split/fold_*.yaml         # 5-fold分割設定（fold_0～4）
│   └── scripts/
│       ├── train/
│       │   └── train.py              # 学習スクリプト（PyTorch）
│       └── inference/
│           └── inference.py          # 推論スクリプト（将来実装）
├── notebook/
│   └── yolo_bbox_quality_analysis.py # BBox品質検証ノートブック
├── docs/
│   ├── WANDB_SETUP.md                # W&B設定ガイド
│   └── HU_WINDOW_GUIDE.md            # HUウィンドウ設定ガイド
├── output/
│   └── train/                        # チェックポイント・ログ保存先
├── requirements.txt                   # 依存関係（PyTorch）
├── pyproject.toml                     # uv設定
└── README.md
```

## 🎯 使用方法

### 学習の実行

**uv環境を使用する場合:**
```bash
cd vertebrae_YOLO/run/scripts/train

# Fold 0で学習（デフォルト）
uv run python train.py

# 特定のFoldで学習
uv run python train.py split=fold_1

# エポック数を変更
uv run python train.py training.max_epochs=50

# バッチサイズを変更
uv run python train.py data.batch_size=32
```

**通常のPython環境を使用する場合:**
```bash
cd vertebrae_YOLO/run/scripts/train

# Fold 0で学習（デフォルト）
python train.py

# 特定のFoldで学習
python train.py split=fold_1
```

### 5-fold交差検証の実行

```bash
# すべてのFoldで学習
for fold in 0 1 2 3 4; do
    uv run python train.py split=fold_${fold}
done
```

## ⚙️ 設定

### モデル設定 ([run/conf/model/yolo_baseline.yaml](run/conf/model/yolo_baseline.yaml))

- **variant**: YOLOv8のバリアント（yolov8n, yolov8s, yolov8m, etc.）
- **num_classes**: クラス数（骨折検出は1）
- **pretrained**: COCO事前学習済み重みを使用

### データ設定 ([run/conf/data/yolo_data.yaml](run/conf/data/yolo_data.yaml))

- **image_size**: 入力画像サイズ（256x256）
- **batch_size**: バッチサイズ（32）
- **num_workers**: DataLoaderのワーカー数（4）
- **hu_windows**: 3チャンネルHUウィンドウ設定（min/max形式）
  - Bone Window: min=400, max=1800
  - Soft Tissue Window: min=-100, max=300
  - Wide Window: min=-200, max=500

### 学習設定 ([run/conf/config.yaml](run/conf/config.yaml))

- **lr**: 学習率（0.001）
- **max_epochs**: 最大エポック数（100）
- **optimizer**: AdamW
- **weight_decay**: 重み減衰（0.0001）
- **scheduler**: CosineAnnealingLR
- **early_stopping_patience**: Early Stopping忍耐値（15）

## 📊 評価指標

- **mAP@0.5**: Mean Average Precision at IoU=0.5
- **mAP@0.5:0.95**: Mean Average Precision at IoU=0.5:0.95
- **椎体レベルF1**: 椎体単位の骨折検出精度

## 🔬 実験管理

### W&B（Weights & Biases）推奨

**セットアップ:**
```bash
# W&Bにログイン（初回のみ）
wandb login

# APIキーを入力（https://wandb.ai/authorize から取得）
```

**使用方法:**
```bash
# W&Bを使用して学習
uv run python train.py logging=wandb

# プロジェクト名をカスタマイズ
uv run python train.py logging=wandb logging.project_name=my_project

# 5-fold CV with W&B
for fold in 0 1 2 3 4; do
    uv run python train.py logging=wandb split=fold_${fold}
done
```

**詳細:** [W&B設定ガイド](docs/WANDB_SETUP.md)を参照

### TensorBoard（オプション）

```bash
# TensorBoardの起動
tensorboard --logdir output/train
```

## 📝 重要な注意事項

1. **患者レベル分割**: データリーケージ防止のため、同一患者のスライスは同じfoldに配置
2. **再現性**: シード固定（seed=42）による実験の再現性確保
3. **メモリ管理**: GPU メモリに応じてバッチサイズを調整（推奨: 16-32）
4. **医療データ**: 患者プライバシーの保護とデータ取り扱いの遵守
5. **実装状況**: 現在はベースライン実装を準備中（データ変換は完了）

## 🚧 実装ステータス

| Phase | タスク | 状態 | 備考 |
|-------|--------|------|------|
| Phase 1 | YOLO形式変換 | ✅ 完了 | [convert_to_yolo.py](data_preparing/convert_to_yolo.py) - 90,638ファイル生成 |
| Phase 1 | Dataset実装 | ✅ 完了 | [yolo_dataset.py](src/dataset/yolo_dataset.py) - 3チャンネルHU処理 |
| Phase 2 | YOLOモデル実装 | ✅ 完了 | [yolo_baseline.py](src/models/yolo_baseline.py) - Ultralytics YOLOv8 |
| Phase 2 | 学習ユーティリティ | ✅ 完了 | [trainer.py](src/utils/trainer.py) - カスタムトレーナー |
| Phase 2 | 学習スクリプト | ✅ 完了 | [train.py](run/scripts/train/train.py) - Hydra設定管理 |
| Phase 2 | ベースライン学習 | ⬜ 未着手 | 実装完了、学習実行待ち |
| Phase 3 | LSTM統合 | ⬜ 未着手 | 時系列統合（ベースライン学習後） |

状態: ✅ 完了 / 🔄 進行中 / ⬜ 未着手

**次のステップ**: Fold 0での学習実行と性能評価

## 🛠️ トラブルシューティング

### GPU メモリ不足

```bash
# バッチサイズを削減
uv run python train.py data.batch_size=8

# 画像サイズを削減
uv run python train.py data.image_size=128

# Gradient Accumulationを使用（実装後）
uv run python train.py training.accumulate_grad_batches=4
```

### Ultralyticsのエラー

```bash
# Ultralyticsの再インストール
pip uninstall ultralytics
pip install ultralytics>=8.0.0

# または uv環境の場合
uv pip install ultralytics>=8.0.0
```

### データローダーのエラー

```bash
# ワーカー数を削減（メモリ不足時）
uv run python train.py data.num_workers=0
```

## 📖 参考資料

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hydra Documentation](https://hydra.cc/)

## 📄 ライセンス

研究・教育目的での利用を想定しています。商用利用の場合は別途相談してください。

---

**開発者向けメモ**: 詳細な実装情報は[claude/YOLO/](../../claude/YOLO/)を参照してください。

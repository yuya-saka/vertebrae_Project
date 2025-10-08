# vertebrae_Unet プロジェクト構造設計

📁 プロジェクト構造
vertebrae_Unet/
├── README.md                    # プロジェクト概要
├── run/                         # 実行スクリプト・設定管理
│   ├── conf/                   # Hydra設定ファイル
│   │   ├── config.yaml         # メイン設定
│   │   ├── train.yaml          # 学習設定
│   │   ├── inference.yaml      # 推論設定
│   │   ├── constants.yaml      # 定数定義
│   │   ├── dir/
│   │   │   └── local.yaml      # ディレクトリパス
│   │   ├── model/
│   │   │   ├── attention_unet.yaml      # Attention U-Net設定
│   │   │   ├── attention_unet_lstm.yaml # U-Net+LSTM設定
│   │   │   └── unet_gan.yaml            # GAN設定
│   │   ├── dataset/
│   │   │   ├── sequence_5.yaml          # 5スライスシーケンス
│   │   │   └── single_slice.yaml        # 単一スライス
│   │   └── split/
│   │       └── train_test.yaml          # データ分割設定
│   └── scripts/                # 機能別スクリプト
│       ├── train/
│       │   ├── train.py                 # 単一学習
│       │   └── run_train.py             # バッチ学習
│       ├── inference/
│       │   ├── inference.py             # 推論
│       │   ├── reconstruct_3d.py        # 3D復元
│       │   └── run_inference.py         # バッチ推論
│       ├── visualization/
│       │   ├── visualize_heatmap.py     # ヒートマップ
│       │   ├── visualize_3d.py          # 3Dレンダリング
│       │   └── visualize_attention.py   # Attention可視化
│       └── utils/
│           ├── combine_metrics.py       # 評価指標統合
│           └── evaluate_3d.py           # 3D評価
├── src/                        # ソースコード
│   ├── datamodule/            # データローダー
│   │   ├── __init__.py
│   │   ├── sequence_dataset.py          # シーケンスデータセット
│   │   ├── single_slice_dataset.py      # 単一スライス
│   │   └── transforms.py                # データ拡張(オプション)
│   ├── modelmodule/           # モデルモジュール
│   │   ├── __init__.py
│   │   ├── attention_unet_module.py     # U-Netモジュール
│   │   └── unet_lstm_module.py          # U-Net+LSTMモジュール
│   ├── models/                # アーキテクチャ定義
│   │   ├── __init__.py
│   │   ├── attention_unet.py            # Attention U-Net
│   │   ├── attention_gate.py            # Attention Gate
│   │   ├── lstm_encoder.py              # LSTM統合
│   │   ├── discriminator.py             # GAN識別器(オプション)
│   │   └── losses.py                    # 損失関数
│   └── utils/                 # ユーティリティ
│       ├── __init__.py
│       ├── metrics.py                   # 評価指標(Dice, IoU)
│       ├── visualization.py             # 可視化関数
│       └── reconstruction.py            # 3D復元
├── data/                       # データ
│   ├── train/
│   ├── test/
│   ├── processed_train/
│   ├── processed_test/
│   ├── slice_train/
│   └── slice_test/
├── data_preprocessing/         # 前処理
│   ├── data_pationing.py
│   ├── volume_cut/
│   └── slice_data/
├── output/                     # 実験結果
│   ├── train/                 # 学習結果
│   │   ├── fold_0/
│   │   ├── fold_1/
│   │   └── ...
│   ├── inference/             # 推論結果
│   │   └── test/
│   ├── visualization/         # 可視化結果
│   │   ├── heatmaps/
│   │   ├── 3d_renders/
│   │   └── attention_maps/
│   ├── metrics/               # 評価指標
│   └── wandb/                 # W&Bログ
├── notebooks/                  # 実験ノートブック
│   ├── exploratory/           # 探索的分析
│   └── experiments/           # 実験記録
└── tests/                      # テストコード
    ├── test_dataset.py
    ├── test_model.py
    └── test_losses.py


## **スクリプト実行ガイド**

### **1. データ前処理スクリプト (vertebrae_Unet/data_preprocessing/)**

#### **1-1. データ分割 (data_pationing.py)**
**機能**: NIfTIファイルを訓練データ(24症例)とテストデータ(8症例)に分割

```bash
# 基本実行
uv run python vertebrae_Unet/data_preprocessing/data_pationing.py
```

**処理内容**:
- `input_nii/` から症例を読み込み
- ランダムに8症例をテストデータに割り当て
- `vertebrae_Unet/data/train/` と `vertebrae_Unet/data/test/` に分割コピー

**出力**:
- `vertebrae_Unet/data/train/` - 訓練データ(24症例)
- `vertebrae_Unet/data/test/` - テストデータ(8症例)

---

#### **1-2. 椎体領域切り出し (volume_cut/)**

**機能**: 各椎体(T4-L5)のバウンディングボックス領域を切り出し

**訓練データの切り出し**:
```bash
uv run python vertebrae_Unet/data_preprocessing/volume_cut/cut_train.py
```

**テストデータの切り出し**:
```bash
uv run python vertebrae_Unet/data_preprocessing/volume_cut/cut_test.py
```

**処理内容**:
- `cut_li*.txt` から切り出し座標を読み込み
- 各椎体領域をマージン付きで切り出し
- マルチプロセス並列処理で高速化

**出力**:
- `vertebrae_Unet/data/processed_train/inp{症例番号}/{椎体番号}/cut_*.nii`
- `vertebrae_Unet/data/processed_test/inp{症例番号}/{椎体番号}/cut_*.nii`

**ログ**: `./logs/nifti_cut_YYYYMMDD_HHMMSS.log`

---

#### **1-3. 2Dスライス画像作成 (slice_data/)**

**機能**: 3D椎体ボリュームから2D axial/coronalスライスを抽出

**訓練データのスライス作成 (Axial)**:
```bash
uv run python vertebrae_Unet/data_preprocessing/slice_data/slice_train_axial.py
```

**テストデータのスライス作成 (Axial)**:
```bash
uv run python vertebrae_Unet/data_preprocessing/slice_data/slice_test_axial.py
```

**処理内容**:
- 各椎体の全スライスを抽出
- 骨折ラベル情報をCSVに保存
- スライスごとの骨折有無を記録

**出力**:
- スライス画像: `vertebrae_Unet/data/slice_train/axial/inp{症例番号}/{椎体番号}/slice_*.nii`
- ラベルCSV: `vertebrae_Unet/data/slice_train/axial/inp{症例番号}/fracture_labels_inp{症例番号}.csv`

**CSVフォーマット**:
| 列名 | 説明 |
|------|------|
| FullPath | スライス画像の絶対パス |
| Vertebra | 椎体番号(27-40: T4-L5) |
| SliceIndex | スライス位置(0-N) |
| Fracture_Label | 骨折有無(0: なし, 1: あり) |
| Case | 症例番号 |
| Axis | 撮影方向(axial/coronal) |

**ログ**: `./logs/slice_extraction_YYYYMMDD_HHMMSS.log`

---

### **2. モデル学習スクリプト (vertebrae_Unet/run/scripts/train/)**

**現在実装中**

#### **2-1. 単一学習実行 (train.py)**
**機能**: Attention U-NetまたはU-Net+LSTMモデルの学習

```bash
# 基本実行
uv run python vertebrae_Unet/run/scripts/train/train.py

# モデル指定
uv run python vertebrae_Unet/run/scripts/train/train.py model=attention_unet

# データセット指定
uv run python vertebrae_Unet/run/scripts/train/train.py dataset=sequence_5

# デバッグモード
uv run python vertebrae_Unet/run/scripts/train/train.py debug=true
```

**設定ファイル**: `vertebrae_Unet/run/conf/train.yaml`

---

#### **2-2. バッチ学習実行 (run_train.py)**
**機能**: 複数の設定で一括学習

```bash
uv run python vertebrae_Unet/run/scripts/train/run_train.py --config vertebrae_Unet/run/conf/run_train.yaml
```

---

### **3. 推論スクリプト (vertebrae_Unet/run/scripts/inference/)**

**現在実装中**

#### **3-1. 基本推論 (inference.py)**
**機能**: 学習済みモデルで2Dセグメンテーション推論

```bash
# 基本実行
uv run python vertebrae_Unet/run/scripts/inference/inference.py

# CSV保存付き
uv run python vertebrae_Unet/run/scripts/inference/inference.py save_csv=true
```

**設定ファイル**: `vertebrae_Unet/run/conf/inference.yaml`

---

#### **3-2. 3D復元 (reconstruct_3d.py)**
**機能**: 2Dセグメンテーション結果を3D確率マップに統合

```bash
uv run python vertebrae_Unet/run/scripts/inference/reconstruct_3d.py
```

**処理内容**:
- 各スライスの予測結果を3Dボリュームに統合
- 確率マップの生成(投票方式または平均化)
- 3D Dice係数、IoUの計算

---

### **4. 可視化スクリプト (vertebrae_Unet/run/scripts/visualization/)**

**現在実装中**

#### **4-1. ヒートマップ可視化 (visualize_heatmap.py)**
**機能**: 骨折確率マップをCT画像に重畳表示

```bash
uv run python vertebrae_Unet/run/scripts/visualization/visualize_heatmap.py
```

---

#### **4-2. Attentionマップ可視化 (visualize_attention.py)**
**機能**: Attention Gateの注目領域を可視化

```bash
uv run python vertebrae_Unet/run/scripts/visualization/visualize_attention.py
```

---

#### **4-3. 3Dレンダリング (visualize_3d.py)**
**機能**: 3D骨折領域を立体表示

```bash
uv run python vertebrae_Unet/run/scripts/visualization/visualize_3d.py
```

---

### **5. 評価スクリプト (vertebrae_Unet/run/scripts/utils/)**

**現在実装中**

#### **5-1. 評価指標統合 (combine_metrics.py)**
**機能**: 各椎体・症例の評価指標を集約

```bash
# 単一実行
uv run python vertebrae_Unet/run/scripts/utils/combine_metrics.py

# マルチラン実行
uv run python vertebrae_Unet/run/scripts/utils/combine_metrics.py --multirun exp_no=001,002,003
```

**設定ファイル**: `vertebrae_Unet/run/conf/combine_metrics.yaml`

**出力**: `vertebrae_Unet/output/metrics/metrics_overall.csv`

---

#### **5-2. 3D評価 (evaluate_3d.py)**
**機能**: 3D復元結果の詳細評価

```bash
uv run python vertebrae_Unet/run/scripts/utils/evaluate_3d.py
```

**評価指標**:
- 3D Dice係数
- 3D IoU
- Precision/Recall
- 椎体別・症例別の統計

---

## **実行フロー例**

### **完全パイプライン実行**

```bash
# 1. データ前処理
uv run python vertebrae_Unet/data_preprocessing/data_pationing.py
uv run python vertebrae_Unet/data_preprocessing/volume_cut/cut_train.py
uv run python vertebrae_Unet/data_preprocessing/volume_cut/cut_test.py
uv run python vertebrae_Unet/data_preprocessing/slice_data/slice_train_axial.py
uv run python vertebrae_Unet/data_preprocessing/slice_data/slice_test_axial.py

# 2. モデル学習(実装後)
uv run python vertebrae_Unet/run/scripts/train/train.py

# 3. 推論実行(実装後)
uv run python vertebrae_Unet/run/scripts/inference/inference.py
uv run python vertebrae_Unet/run/scripts/inference/reconstruct_3d.py

# 4. 評価と可視化(実装後)
uv run python vertebrae_Unet/run/scripts/utils/evaluate_3d.py
uv run python vertebrae_Unet/run/scripts/visualization/visualize_heatmap.py
```

---

### **デバッグ実行フロー**

```bash
# 少数症例でテスト
uv run python vertebrae_Unet/run/scripts/train/train.py debug=true max_epochs=3

# 1症例のみ推論
uv run python vertebrae_Unet/run/scripts/inference/inference.py test_case=1010
```

---

## **出力ファイル構造**

```
vertebrae_Unet/
├── data/
│   ├── train/                    # 生データ(24症例)
│   ├── test/                     # 生データ(8症例)
│   ├── processed_train/          # 切り出し後(椎体別)
│   │   └── inp{症例番号}/{椎体番号}/cut_*.nii
│   ├── processed_test/           # 切り出し後(椎体別)
│   ├── slice_train/axial/        # スライス画像(訓練)
│   │   └── inp{症例番号}/{椎体番号}/slice_*.nii
│   └── slice_test/axial/         # スライス画像(テスト)
├── output/
│   ├── train/                    # 学習結果
│   │   └── {実験名}/
│   │       ├── checkpoints/      # モデル重み
│   │       └── logs/             # 学習ログ
│   ├── inference/                # 推論結果
│   │   └── {実験名}/
│   │       ├── predictions_2d/   # 2D予測マスク
│   │       ├── predictions_3d/   # 3D復元マップ
│   │       └── metrics.csv       # 評価指標
│   ├── visualization/            # 可視化結果
│   │   ├── heatmaps/             # ヒートマップ
│   │   ├── 3d_renders/           # 3Dレンダリング
│   │   └── attention_maps/       # Attentionマップ
│   └── wandb/                    # Weights & Biasログ
└── logs/                         # 実行ログ
    ├── nifti_cut_*.log
    └── slice_extraction_*.log
```

---

## **技術仕様**

### **データ仕様**
- **入力形式**: NIfTI (.nii, .nii.gz)
- **画像サイズ**: 可変(椎体別に切り出し)
- **HU範囲**: -1000~3000 (正規化必要)
- **椎体番号**: 27-40 (T4-L5)
- **症例数**: 32症例(訓練24, テスト8)

### **モデル仕様**
- **アーキテクチャ**: Attention U-Net + LSTM(オプション)
- **入力**: 単一スライス or 5スライスシーケンス
- **出力**: 骨折セグメンテーションマスク(H×W)
- **損失関数**: Dice Loss + BCE Loss + Adversarial Loss(オプション)

### **評価指標**
- **2D評価**: Dice係数, IoU, Precision, Recall
- **3D評価**: 3D Dice, 3D IoU, 椎体別精度

---

## **トラブルシューティング**

### **よくある問題**

#### **1. データパスエラー**
```bash
# パスが正しいか確認
ls vertebrae_Unet/data/train/
ls input_nii/
```

#### **2. メモリ不足**
```bash
# マルチプロセス数を減らす
# cut_train.py の max_workers を調整
```

#### **3. ログファイルが見つからない**
```bash
# ログディレクトリを確認
ls ./logs/
```

---

## **プロジェクトステータス**

### **実装済み**
- ✅ データ分割 (data_pationing.py)
- ✅ 椎体領域切り出し (volume_cut/)
- ✅ Axialスライス作成 (slice_data/)

### **実装中**
- 🚧 Attention U-Netモデル
- 🚧 U-Net + LSTMモデル
- 🚧 学習スクリプト

### **未実装**
- ⏳ 推論スクリプト
- ⏳ 3D復元スクリプト
- ⏳ 可視化スクリプト
- ⏳ 評価スクリプト

---

## **参考資料**

### **関連プロジェクト**
- [Sakaguchi_file](Sakaguchi_file/): ResNet18分類ベースライン
- [prior_YOLO_file](prior_YOLO_file/): YOLO検出アプローチ

### **技術スタック**
- PyTorch Lightning - 学習フレームワーク
- Hydra - 設定管理
- Weights & Biases - 実験管理
- nibabel - NIfTI処理
- NumPy, Pandas - データ処理

### **プロジェクト構造**
詳細は [vertebrae_Unet/project.md](vertebrae_Unet/project.md) を参照


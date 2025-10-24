# スラブ画像 + Focal Loss学習済みYOLOv8モデル 推論ガイド

このドキュメントでは、スラブRGB画像（R=1-slice, G=15-slice, B=31-slice）とFocal Lossで学習したYOLOv8モデルを使用した推論の方法を説明します。

## 目次
- [概要](#概要)
- [環境構築](#環境構築)
- [使用方法](#使用方法)
  - [基本的な使用](#基本的な使用)
  - [単一画像推論](#単一画像推論)
  - [複数しきい値評価](#複数しきい値評価)
  - [Pythonスクリプトからの使用](#pythonスクリプトからの使用)
  - [Jupyterノートブックでの使用](#jupyterノートブックでの使用)
- [出力ファイル](#出力ファイル)
- [パラメータ説明](#パラメータ説明)
- [トラブルシューティング](#トラブルシューティング)

---

## 概要

### 学習済みモデル情報
- **モデルパス**: `/mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/Sakaguchi_file/slab_dataset_creation/focal_loss_training_results/slab_focal_loss_threshold_0.13/weights/best.pt`
- **モデルアーキテクチャ**: YOLOv8s (small)
- **学習手法**: Focal Loss (α=9.65, γ=2.5)
- **入力画像**: スラブRGB画像
  - Rチャンネル: 1-slice slab (高解像度詳細構造)
  - Gチャンネル: 15-slice slab (中解像度構造情報)
  - Bチャンネル: 31-slice slab (低ノイズ広域構造)
- **学習日**: 2025年7月24日
- **性能指標** (閾値=0.1):
  - mAP@0.5: 0.687
  - mAP@0.5:0.95: 0.313
  - Precision: 0.893
  - Recall: 0.500

### 主な機能
- ✅ 単一画像での推論
- ✅ データセット全体での推論
- ✅ 複数しきい値での性能評価
- ✅ 詳細な評価メトリクス計算
- ✅ 結果の可視化（混同行列、ROC曲線、予測サンプル）
- ✅ JSON/CSV形式での結果保存

---

## 環境構築

### 必要な依存関係

```bash
# 仮想環境の作成（推奨）
cd /mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/vertebrae_YOLO
python -m venv venv
source venv/bin/activate

# 依存関係のインストール
pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install matplotlib seaborn
pip install pandas numpy
pip install scikit-learn
pip install tqdm
pip install PyYAML
pip install jupyter notebook  # Jupyterノートブック使用時
```

---

## 使用方法

### 基本的な使用

テストデータセット全体で推論を実行し、評価結果を出力します。

```bash
cd /mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/vertebrae_YOLO/src

python inference/slab_focal_loss_inference.py \
    --model_path /path/to/best.pt \
    --data_config /path/to/slab_dataset.yaml \
    --data_split test \
    --conf_threshold 0.1
```

**具体例**:
```bash
python inference/slab_focal_loss_inference.py \
    --model_path /mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/Sakaguchi_file/slab_dataset_creation/focal_loss_training_results/slab_focal_loss_threshold_0.13/weights/best.pt \
    --data_config /mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/Sakaguchi_file/slab_dataset_creation/slab_dataset/slab_dataset.yaml \
    --data_split test \
    --conf_threshold 0.1
```

**出力例**:
```
モデル読み込み中: /path/to/best.pt
✓ モデル読み込み完了
出力ディレクトリ: runs/inference/slab_focal_loss_20251021_120000
...
============================================================
評価結果
============================================================
Accuracy:    0.9234
Precision:   0.8930
Recall:      0.5000
F1-Score:    0.6400
Sensitivity: 0.5000
Specificity: 0.9580
AUC:         0.8765
============================================================
```

---

### 単一画像推論

1枚の画像に対して推論を実行します。可視化結果が自動的に保存されます。

```bash
python inference/slab_focal_loss_inference.py \
    --model_path /path/to/best.pt \
    --single_image /path/to/test_image.png \
    --conf_threshold 0.1
```

**出力例**:
```
推論実行: /path/to/test_image.png
  検出数: 2
  最大信頼度: 0.842
  可視化保存: runs/inference/slab_focal_loss_*/test_image_prediction.png
```

---

### 複数しきい値評価

複数の信頼度閾値で性能を比較評価します。最適な閾値を見つけるのに有用です。

```bash
python inference/slab_focal_loss_inference.py \
    --model_path /path/to/best.pt \
    --data_config /path/to/slab_dataset.yaml \
    --data_split test \
    --thresholds 0.01 0.05 0.1 0.25 0.5
```

**出力例**:
```
============================================================
しきい値別評価結果:
============================================================
   threshold  accuracy  precision    recall  f1_score       auc
        0.01    0.8542     0.6234    0.8500    0.7189    0.8765
        0.05    0.9012     0.7845    0.6800    0.7289    0.8765
        0.10    0.9234     0.8930    0.5000    0.6400    0.8765
        0.25    0.9456     0.9234    0.3500    0.5089    0.8765
        0.50    0.9512     0.9500    0.2000    0.3289    0.8765
```

しきい値比較グラフも自動生成されます：
- `threshold_comparison.png`: 各メトリクスの変化を可視化
- `threshold_comparison.csv`: 結果をCSV形式で保存

---

### Pythonスクリプトからの使用

推論クラスを直接インポートして使用できます。

```python
from pathlib import Path
from inference.slab_focal_loss_inference import SlabFocalLossPredictor

# 推論器を初期化
predictor = SlabFocalLossPredictor(
    model_path="/path/to/best.pt",
    config_path="/path/to/slab_dataset.yaml",
    output_dir="./my_inference_results"
)

# 単一画像推論
result = predictor.predict_single_image(
    image_path="/path/to/image.png",
    conf_threshold=0.1,
    save_visualization=True
)

print(f"骨折検出: {result['has_fracture']}")
print(f"検出数: {result['num_detections']}")
print(f"信頼度: {result['confidences']}")

# データセット評価
predictions, metrics = predictor.run_evaluation(
    data_split='test',
    conf_threshold=0.1,
    visualize=True
)

print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")

# 複数しきい値評価
df_results = predictor.evaluate_multiple_thresholds(
    data_split='test',
    thresholds=[0.01, 0.05, 0.1, 0.25]
)
print(df_results)
```

---

### Jupyterノートブックでの使用

インタラクティブな分析には、提供されているノートブックを使用します。

```bash
cd /mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/vertebrae_YOLO/notebook
jupyter notebook slab_focal_loss_inference.ipynb
```

**ノートブックの内容**:
1. 環境セットアップ
2. パス設定
3. データセット確認
4. 推論器初期化
5. 単一画像推論テスト
6. テストデータセット全体評価
7. 複数しきい値評価
8. 結果の詳細分析
9. メトリクス確認

すべてのセルを順番に実行することで、完全な推論・評価プロセスを体験できます。

---

## 出力ファイル

推論実行後、以下のファイルが `output_dir` に生成されます。

### ディレクトリ構造
```
runs/inference/slab_focal_loss_TIMESTAMP/
├── predictions.json              # 全予測結果（JSON形式）
├── metrics.json                  # 評価メトリクス（JSON形式）
├── results.csv                   # 予測結果サマリー（CSV形式）
├── confusion_matrix.png          # 混同行列
├── roc_curve.png                 # ROC曲線
├── sample_predictions.png        # サンプル予測結果の可視化
├── threshold_comparison.png      # しきい値比較グラフ（複数しきい値評価時）
└── threshold_comparison.csv      # しきい値比較結果（複数しきい値評価時）
```

### ファイル詳細

#### `predictions.json`
各画像の詳細な予測結果。
```json
[
  {
    "image_path": "/path/to/image1.png",
    "boxes": [[x1, y1, x2, y2], ...],
    "confidences": [0.85, 0.72, ...],
    "classes": [0, 0, ...],
    "has_fracture": true,
    "num_detections": 2
  },
  ...
]
```

#### `metrics.json`
評価メトリクスとモデル情報。
```json
{
  "model_path": "/path/to/best.pt",
  "confidence_threshold": 0.1,
  "timestamp": "2025-10-21T12:00:00",
  "metrics": {
    "accuracy": 0.9234,
    "precision": 0.8930,
    "recall": 0.5000,
    "f1_score": 0.6400,
    "sensitivity": 0.5000,
    "specificity": 0.9580,
    "auc": 0.8765,
    "confusion_matrix": {
      "tn": 8100,
      "fp": 150,
      "fn": 182,
      "tp": 182
    },
    "total_samples": 8614,
    "positive_samples": 364,
    "negative_samples": 8250
  }
}
```

#### `results.csv`
画像ごとの予測サマリー。
```csv
image_path,has_fracture,num_detections,max_confidence
/path/to/image1.png,True,2,0.8500
/path/to/image2.png,False,0,0.0000
/path/to/image3.png,True,1,0.7200
...
```

#### `threshold_comparison.csv`
複数しきい値評価の結果。
```csv
threshold,accuracy,precision,recall,f1_score,sensitivity,specificity,auc
0.01,0.8542,0.6234,0.8500,0.7189,0.8500,0.8523,0.8765
0.05,0.9012,0.7845,0.6800,0.7289,0.6800,0.9100,0.8765
0.10,0.9234,0.8930,0.5000,0.6400,0.5000,0.9580,0.8765
0.25,0.9456,0.9234,0.3500,0.5089,0.3500,0.9789,0.8765
```

---

## パラメータ説明

### コマンドライン引数

| パラメータ | 必須 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--model_path` | ✅ | - | 学習済みモデルファイル（.pt）のパス |
| `--data_config` | - | - | データセット設定ファイル（.yaml）のパス |
| `--data_split` | - | `test` | 評価するデータ分割（`train`, `val`, `test`） |
| `--conf_threshold` | - | `0.1` | 信頼度閾値（0.0～1.0） |
| `--thresholds` | - | - | 複数しきい値評価用の閾値リスト（例: `0.01 0.05 0.1`） |
| `--single_image` | - | - | 単一画像推論用の画像パス |
| `--output_dir` | - | `runs/inference/slab_focal_loss_*` | 結果出力ディレクトリ |
| `--no_visualize` | - | `False` | 可視化をスキップ |

### Pythonクラスパラメータ

#### `SlabFocalLossPredictor.__init__()`
```python
def __init__(
    model_path: str,           # 学習済みモデルパス（必須）
    config_path: str = None,   # データセット設定パス（オプション）
    output_dir: str = None     # 出力ディレクトリ（オプション）
)
```

#### `predict_single_image()`
```python
def predict_single_image(
    image_path: str,                # 画像パス
    conf_threshold: float = 0.1,    # 信頼度閾値
    save_visualization: bool = True # 可視化を保存するか
) -> Dict
```

#### `run_evaluation()`
```python
def run_evaluation(
    data_split: str = 'test',      # データ分割
    conf_threshold: float = 0.1,   # 信頼度閾値
    visualize: bool = True         # 可視化を行うか
) -> Tuple[List[Dict], Dict]
```

#### `evaluate_multiple_thresholds()`
```python
def evaluate_multiple_thresholds(
    data_split: str = 'test',           # データ分割
    thresholds: List[float] = [0.01, 0.05, 0.1, 0.25]  # 評価する閾値リスト
) -> pd.DataFrame
```

---

## トラブルシューティング

### よくある問題と解決策

#### 1. モデルファイルが見つからない
```
FileNotFoundError: モデルファイルが見つかりません: /path/to/best.pt
```

**解決策**:
- モデルパスが正しいか確認してください
- ファイルの存在を確認: `ls -la /path/to/best.pt`

#### 2. CUDA out of memory エラー
```
RuntimeError: CUDA out of memory
```

**解決策**:
- バッチサイズを小さくする（データセット評価時）
- 他のGPUプロセスを終了する
- CPUで実行: `CUDA_VISIBLE_DEVICES=-1 python ...`

#### 3. データセット設定ファイルが見つからない
```
設定ファイル読み込みエラー: [Errno 2] No such file or directory
```

**解決策**:
- `--data_config` パスが正しいか確認
- データセット設定ファイルの内容を確認:
  ```bash
  cat /path/to/slab_dataset.yaml
  ```

#### 4. 画像が見つからない
```
データセットパスが存在しません: /path/to/images
```

**解決策**:
- データセット設定ファイル（yaml）の`path`が正しいか確認
- 画像ディレクトリの存在確認:
  ```bash
  ls -la /path/to/dataset/test/images/
  ```

#### 5. 依存関係エラー
```
ModuleNotFoundError: No module named 'ultralytics'
```

**解決策**:
```bash
pip install ultralytics
```

---

## 評価メトリクスの解釈

### 主要メトリクス

- **Accuracy（正解率）**: 全予測のうち、正しく予測できた割合
- **Precision（適合率）**: 骨折と予測したもののうち、実際に骨折だった割合
- **Recall（再現率）**: 実際の骨折のうち、正しく検出できた割合
- **F1-Score**: PrecisionとRecallの調和平均（バランス指標）
- **Sensitivity（感度）**: Recallと同じ（骨折の検出率）
- **Specificity（特異度）**: 正常を正常と判定できた割合
- **AUC**: ROC曲線の下の面積（モデルの識別能力）

### しきい値の選択

- **低い閾値（0.01～0.05）**: Recall重視（見逃しを減らす）
- **中程度の閾値（0.1～0.25）**: バランス重視（F1-Score最大化）
- **高い閾値（0.5～）**: Precision重視（誤検出を減らす）

**推奨**: 臨床応用では見逃しを避けるため、Recall重視の低めの閾値が望ましい。

---

## サポート

問題が発生した場合：
1. このドキュメントのトラブルシューティングを確認
2. ログファイル（`slab_focal_loss_inference.log`）を確認
3. GitHubのIssueを作成

---

## 更新履歴

- **2025-10-21**: 初版作成
  - 基本的な推論機能
  - 複数しきい値評価
  - 可視化機能
  - Jupyterノートブック

---

## ライセンス

このプロジェクトは研究目的で使用されます。

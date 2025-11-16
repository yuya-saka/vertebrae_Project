## 🔄 データセット変更に伴う重要な更新 (2025年1月)

### **🎯 新しいデータセット形式**

**変更内容:**
- **旧:** NIFTI形式 (`.nii`)、HU値、可変解像度
- **新:** PNG形式 (`.png`)、8-bit RGB、正規化済み、統一解像度

**新データセットの特徴:**

```yaml
# データ構造
data/dataset/
  ├── Path/                      # CSVファイル格納
  │   ├── segmentation_dataset_axial.csv
  │   ├── segmentation_dataset_coron.csv
  │   └── segmentation_dataset_sagit.csv
  ├── slice_image/              # 入力画像 (PNG)
  └── slice_image_ans/          # マスク画像 (PNG)

# CSV列構成
image_path, mask_path, patient_id, vertebra_id, orientation, has_fracture

# patient_id形式
"AI1003" (文字列、旧: 数値 1003)
```

---

### **✅ データセット更新済みコード**

#### **dataset.py** (完全書き換え済み)

**主な変更点:**
- ✅ NIFTI読み込み削除 → PNG読み込み (OpenCV)
- ✅ HU Window処理削除 (PNG画像は正規化済み)
- ✅ 3チャンネル入力: RGBをそのまま使用
- ✅ patient_id形式変更対応 (`AI{id}`)
- ✅ 新CSV列名対応 (`has_fracture`, `image_path`, `mask_path`)

```python
# 新しいデータセット初期化
dataset = MultiTaskDataset(
    csv_file="data/dataset/Path/segmentation_dataset_axial.csv",
    project_root="/path/to/project",  # CSV内パスの基準
    patient_ids=[1003, 1015, ...],    # 数値IDを自動変換
    image_size=(256, 256),
    augmentation={...},
    is_training=True
)
```

#### **dataloader.py** (完全書き換え済み)

**主な変更点:**
- ✅ 単一CSVファイルから患者IDでフィルタリング
- ✅ 複数CSVファイル検索ロジック削除

---

### **✅ 設定ファイル更新済み**

#### **constants.yaml**

```yaml
# 新しいデータパス
dataset_dir: "${data_dir}/dataset"
dataset_path_dir: "${dataset_dir}/Path"
slice_image_dir: "${dataset_dir}/slice_image"
slice_image_ans_dir: "${dataset_dir}/slice_image_ans"

# patient_id: 数値のまま管理（コード内で"AI{id}"に変換）
train_patient_ids: [1003, 1015, 1017, ...]
test_patient_ids: [1010, 1012, 1016, ...]
```

#### **data_direction/{axial|coronal|sagittal}.yaml**

```yaml
# 新しいCSVファイル指定
csv_file: "${dataset_path_dir}/segmentation_dataset_axial.csv"
project_root_for_csv: "${project_root}"

# PNG対応フラグ
use_png: true

# HU Window設定を無効化
hu_windows: null  # PNG画像は正規化済み
```

---

### **✅ 動作確認済み**

テストスクリプト (`A/test_dataloader.py`) で確認完了:

```bash
✓ DataLoaders created successfully!
  - Train batches: 857
  - Val batches: 170
  - Train samples: 37,648 (24 patients)
  - Val samples: 10,860 (6 patients)

✓ Batch structure confirmed:
  - Image shape: (32, 3, 256, 256) ✓
  - Mask shape: (32, 1, 256, 256) ✓
  - Class balance: 16 fracture / 16 non-fracture ✓
```

---

## 📋 学習開始前に調整すべき設定

### **✅ 1. GPU・メモリ関連設定**

**現在の環境: RTX A6000 (49GB) × 3**

#### 推奨設定（`train.yaml`）:

```yaml
training:
  batch_size: 16        # ✅ OK (RTX A6000なら32-64も可能)
  num_workers: 4        # ⚠️ 要調整 → 8-12推奨
  accumulation_steps: 1 # ✅ OK
```

**調整案:**

- `num_workers: 8` に変更（データロード高速化）
- `batch_size`はまず16で試し、GPUメモリに余裕があれば32に増やす

---

### **⚠️ 2. データパス設定の修正が必要**

**問題:** CSVファイル内のパスが古いプロジェクトパス(`vertebrae_Unet`)を参照

```csv
FullPath=/mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/vertebrae_Unet/data/...
```

**現在の実装では:**

- `dataset.py`が`row['FullPath']`をそのまま使用
- パスが存在しないとエラーになる可能性

**対策（2つの選択肢）:**

#### **選択肢A: データセットコードを修正（推奨）**

`src/datamodule/dataset.py`の`__getitem__`を以下のように修正:

```python
# 修正前
image_path = row['FullPath']

# 修正後（新パスを構築）
case_id = f"inp{row['Case']}"
vertebra = str(row['Vertebra'])
slice_idx = row['SliceIndex']
axis = row['Axis']
image_path = self.image_base_dir / case_id / vertebra / f"slice_{slice_idx:03d}.nii"
```

#### **選択肢B: シンボリックリンクを作成**

```bash
ln -s /mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka \
      /mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/vertebrae_Unet
```

---

### **⚠️ 3. WandB設定**

**`config.yaml`:**

```yaml
wandb:
  entity: null  # ⚠️ あなたのWandBユーザー名を設定
  mode: "online"  # または "offline" (デバッグ用)
```

**調整:**

```yaml
wandb:
  entity: "your-wandb-username"  # ← 要設定
  mode: "offline"  # 初回テストは offline 推奨
```

---

### **✅ 4. 学習ハイパーパラメータ（現在の設定は良好）**

#### **学習率設定:**

```yaml
optimizer:
  lr: 0.001                    # ✅ 適切
  encoder_lr_factor: 0.1       # ✅ エンコーダは 0.0001
  use_differential_lr: true    # ✅ ImageNet pretrainedなので必須
```

#### **損失の重み:**

```yaml
loss:
  w_class: 1.0   # ✅ 分類が主タスク
  w_seg: 0.1     # ✅ セグは補助タスク（適切）
```

**注意点:**

- `w_seg`が大きすぎると骨折検出精度が下がる可能性
- 必要に応じて`0.05`や`0.2`も試す

---

### **⚠️ 5. Early Stopping設定**

**現在の設定:**

```yaml
early_stopping:
  monitor: "val_loss"
  mode: "min"

checkpoint:
  monitor: "val_pr_auc"  # ← これが主要評価指標
  mode: "max"
```

**問題:** モニター指標が不一致

 

**推奨修正:**

```yaml
early_stopping:
  enabled: true
  patience: 15
  monitor: "val_pr_auc"  # ← チェックポイントと統一
  mode: "max"            # ← max に変更

checkpoint:
  monitor: "val_pr_auc"
  mode: "max"
```

---

### **✅ 6. データ拡張設定（現在の設定は適切）**

```yaml
augmentation:
  rotation_degrees: 45       # ✅ 強い拡張
  translation_pixels: 20     # ✅ 適切
  scale_range: [0.8, 1.2]    # ✅ 適切
  horizontal_flip_prob: 0.5  # ✅ 適切
```

---

### **📝 7. テスト実行用の設定**

初回テストには以下のオーバーライドを推奨:

```bash
# 1エポックのみ、少数データでテスト
uv run python train.py \
  training.max_epochs=1 \
  training.batch_size=4 \
  wandb.mode=offline
```

---

## 🔧 **必須修正項目まとめ**

### **最優先:**

1. **データパス問題の解決** (選択肢AまたはB)
2. **WandB entity設定** (`your-wandb-username`)
3. **Early Stopping指標の統一** (`val_pr_auc`)

### **推奨:**

4. **num_workers増加** (4 → 8)
5. **初回はofflineモード** でテスト

---

## 📄 修正版設定ファイル

必要であれば、以下の修正版を作成しましょうか？

1. `src/datamodule/dataset.py` (パス構築修正)
2. `run/conf/config.yaml` (WandB設定)
3. `run/conf/train.yaml` (Early Stopping修正)

どの修正を実施しますか？
---

## 📝 データセット更新後の注意点 (2025年1月追記)

### **✅ 完了済みの修正**

1. ✅ **PNG データセット対応完了**
   - dataset.py: NIFTI → PNG読み込み
   - HU Window処理削除
   - 新CSV構造対応

2. ✅ **設定ファイル更新完了**
   - constants.yaml: 新データパス追加
   - data_direction/*.yaml: CSV指定、HU Window削除

3. ✅ **動作確認完了**
   - test_dataloader.py で検証済み
   - Train/Val分割正常動作
   - クラスバランス確認済み

### **⚠️ 学習前の最終チェック**

```bash
# 1. テストスクリプトで動作確認
cd A
uv run python test_dataloader.py

# 2. 学習開始 (Axial, Fold 0)
cd run/scripts
uv run python train.py wandb.mode=offline  # 初回はオフライン推奨

# 3. 正常動作確認後、本番学習
uv run python train.py
```

### **推奨設定変更**

```yaml
# train.yaml
training:
  batch_size: 32       # PNG画像は軽量なので32推奨
  num_workers: 8       # データロード高速化

# config.yaml
wandb:
  entity: "your-username"  # ← 要設定
  mode: "offline"          # 初回テストはオフライン
```


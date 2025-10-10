## **開発状況**

### **現在の実装状況サマリー**

**✅ Phase 0: データ準備（完了）**
- データ分割（train: 30症例, test: 8症例）
- 椎体領域切り出し（T4-L5）
- 2D Axialスライス作成とラベルCSV生成

**🚧 Phase 1: U-Netベースライン構築（進行中）**
- 目標: 単一スライス入力のAttention U-Netで学習・評価完了

---

## **Phase 1 実装計画: U-Netベースライン構築**

### **Step 1: Hydra設定ファイル構築** 📝
**状態: ⏳ 未着手**

#### タスク一覧
- [ ] `run/conf/config.yaml` - メイン設定ファイル
- [ ] `run/conf/constants.yaml` - 定数定義（HU範囲、画像サイズ等）
- [ ] `run/conf/dir/local.yaml` - ディレクトリパス設定
- [ ] `run/conf/train.yaml` - 学習ハイパーパラメータ
- [ ] `run/conf/split/fold_0.yaml` - フォールド分割定義
- [ ] `run/conf/model/unet.yaml` - U-Netモデル設定

#### 技術仕様
- HU範囲: 0-1800（constants.yamlで定義）
- 画像サイズ: アスペクト比維持、ゼロパディングで統一
- フォールド数: 5-fold cross validation
- W&B設定: プロジェクト名、エンティティ名

---

### **Step 2: Attention U-Netアーキテクチャ実装** 🏗️
**状態: ⏳ 未着手**

#### タスク一覧
- [ ] `src/models/__init__.py` - パッケージ初期化
- [ ] `src/models/attention_gate.py` - Attention Gate実装
  - [ ] AttentionGateクラス
  - [ ] テストコード（unit test）
- [ ] `src/models/attention_unet.py` - Attention U-Net本体
  - [ ] エンコーダ実装（5層）
  - [ ] デコーダ実装（5層 + Attention Gate）
  - [ ] ボトルネック実装
  - [ ] forward関数
  - [ ] テストコード（入出力形状確認）

#### 技術仕様
- **入力**: (B, 1, H, W) - バッチ、チャンネル、高さ、幅
- **出力**: (B, 1, H, W) - セグメンテーションマスク
- **エンコーダ**: Conv3x3 → BatchNorm → ReLU → MaxPool
- **デコーダ**: ConvTranspose → Attention Gate → Concat → Conv
- **Attention Gate**: スキップ接続に適用

---

### **Step 3: データローダー実装** 📦
**状態: ⏳ 未着手**

#### タスク一覧
- [ ] `src/datamodule/__init__.py` - パッケージ更新
- [ ] `src/datamodule/dataset.py` - PyTorch Dataset
  - [ ] VertebralDatasetクラス
  - [ ] CSV読み込み機能
  - [ ] NIfTI画像読み込み
  - [ ] HU正規化（0-1800 → 0-1）
  - [ ] ゼロパディング実装
  - [ ] Data Augmentation（回転、スケール、輝度）
- [ ] `src/datamodule/dataloader.py` - DataLoader構築
  - [ ] 患者レベル分割（データリーケージ防止）
  - [ ] train/val split機能
  - [ ] バッチサイズ設定

#### 技術仕様
- **正規化**: `(HU - 0) / (1800 - 0)` → [0, 1]
- **パディング**: アスペクト比維持、最大サイズに統一
- **Augmentation**:
  - 回転: ±15度
  - スケール: 0.9-1.1倍
  - 輝度: ±10%
- **分割**: 患者単位で80% train, 20% val

---

### **Step 4: モデルモジュール実装** ⚙️
**状態: ⏳ 未着手**

#### タスク一覧
- [ ] `src/modelmodule/__init__.py` - パッケージ初期化
- [ ] `src/modelmodule/model_module.py` - LightningModule
  - [ ] `__init__` - モデル、損失関数、評価指標初期化
  - [ ] `forward` - 順伝播
  - [ ] `training_step` - 学習ステップ
  - [ ] `validation_step` - 検証ステップ
  - [ ] `configure_optimizers` - Optimizer/Scheduler設定
  - [ ] `_calculate_metrics` - 評価指標計算（Dice, IoU, Precision, Recall）

#### 技術仕様
- **損失関数**: Dice Loss + Binary Cross Entropy（重み: 0.5:0.5）
- **Optimizer**: AdamW（lr=1e-4, weight_decay=1e-5）
- **Scheduler**: ReduceLROnPlateau（patience=5, factor=0.5）
- **評価指標**:
  - Dice係数
  - IoU (Intersection over Union)
  - Precision
  - Recall

---

### **Step 5: 学習スクリプト実装** 🚀
**状態: ⏳ 未着手**

#### タスク一覧
- [ ] `run/scripts/train/train.py` - 単一fold学習
  - [ ] Hydra設定読み込み
  - [ ] データローダー初期化
  - [ ] モデル初期化
  - [ ] W&Bロガー設定
  - [ ] ModelCheckpoint設定（best/lastモデル保存）
  - [ ] EarlyStopping設定（patience=10）
  - [ ] Trainer設定
  - [ ] 学習実行
  - [ ] 結果保存

#### 技術仕様
- **出力先**: `output/train/{実験名}/axial/fold_0/`
- **保存内容**:
  - `best_model.ckpt` - 最良モデル
  - `last_model.ckpt` - 最終エポックモデル
  - `metrics.csv` - エポック毎の評価指標
  - `config.yaml` - 使用した設定ファイル

---

### **Step 6: 推論・評価スクリプト実装** 📊
**状態: ⏳ 未着手**

#### タスク一覧
- [ ] `run/scripts/inference/inference.py` - 2D推論
  - [ ] チェックポイント読み込み
  - [ ] テストデータ推論
  - [ ] 予測マスク保存
  - [ ] 評価指標計算
- [ ] `run/scripts/utils/evaluate_2d.py` - 2D評価
  - [ ] 症例別評価指標計算
  - [ ] 椎体別評価指標計算
  - [ ] 統計量計算（平均、標準偏差）
  - [ ] 結果CSV出力

#### 技術仕様
- **出力先**: `output/inference/{実験名}/axial/fold_0/`
- **保存内容**:
  - `predictions/` - 予測マスク（NIfTI形式）
  - `metrics_per_case.csv` - 症例別評価指標
  - `metrics_per_vertebra.csv` - 椎体別評価指標
  - `metrics_summary.csv` - 統計サマリー

---

## **実装スケジュール（推奨）**

### Week 1
- **Day 1-2**: Step 1 - Hydra設定ファイル構築
- **Day 3-5**: Step 2 - Attention U-Net実装 + テスト
- **Day 6-7**: Step 3 - データローダー実装 + テスト

### Week 2
- **Day 1-3**: Step 4 - モデルモジュール実装 + テスト
- **Day 4-5**: Step 5 - 学習スクリプト実装
- **Day 6-7**: デバッグ実行（少数エポック）

### Week 3
- **Day 1-2**: Step 6 - 推論・評価スクリプト実装
- **Day 3-7**: 本格実験開始 + 結果分析

---

## **重要な技術的考慮事項**

### 1. データリーケージ防止
- ✅ **患者レベル分割**: 同一患者の異なる椎体が train/val 両方に含まれないようにする
- ✅ **フォールド分割**: 患者IDをキーにK-fold split実行

### 2. 画像サイズ統一
- ✅ **アスペクト比維持**: オリジナルの縦横比を保持
- ✅ **ゼロパディング**: 最大サイズに合わせてパディング
- ✅ **設定ファイル管理**: `constants.yaml`で柔軟に変更可能

### 3. HU値正規化
- ✅ **範囲**: 0-1800 → 0-1の線形変換
- ✅ **定義場所**: `constants.yaml`

### 4. W&B統合
- ✅ **自動記録**: Loss、Dice、IoU、Precision、Recall
- ✅ **モデル保存**: W&B Artifactsにチェックポイント保存
- ✅ **可視化**: 予測マスク、Attentionマップ（Phase 2）

---

## **Next Steps（次にやること）**

**現在の優先タスク**: Step 1 - Hydra設定ファイル構築

1. `run/conf/config.yaml` を作成
2. `run/conf/constants.yaml` でHU範囲・画像サイズ定義
3. `run/conf/dir/local.yaml` でパス設定
4. その他設定ファイル整備

**開始コマンド**:
```bash
# Step 1開始後
uv run python vertebrae_Unet/run/scripts/train/train.py --cfg job
# → 設定ファイル確認
```


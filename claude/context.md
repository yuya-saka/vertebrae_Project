## **開発状況**

### **現在の実装状況サマリー**

**✅ Phase 0: データ準備（完了）**
- データ分割（train: 30症例, test: 8症例）
- 椎体領域切り出し（T4-L5）
- 2D Axialスライス作成とラベルCSV生成
- マスク画像スライス作成（train/test両方）

**✅ Phase 0.5: 探索的データ分析（完了）** - 2025/10/10 → 2025/10/12
- ✅ 画像サイズ分布の可視化完了
- ✅ HU値分布の分析完了
- ✅ CT画像とマスクのオーバーレイ確認完了
- ✅ HU値ウィンドウイング探索完了
- ✅ **データ分布分析完了** - 2025/10/12
  - クラス分布: Train 10.83%, Test 9.39%
  - 骨折ピクセル比率: 平均 0.616%
  - 椎体別骨折分布確認
- ✅ **データ拡張戦略確定** - 2025/10/12
  - オンライン拡張・クラス比率均衡化（50%目標）
  - 椎体単位でスライス間連続性保持

**✅ Phase 1: U-Netベースライン構築（実装完了）** - 2025/10/12
- 目標: 単一スライス入力のAttention U-Netで学習・評価完了
- ✅ Hydra設定ファイル構築完了
- ✅ Dataset/DataLoader実装完了（3チャンネル入力、オンライン拡張対応）
- ✅ Attention Gate実装完了
- ✅ Attention U-Net実装完了
- ✅ LightningModule実装完了（損失関数、評価指標含む）
- ✅ 学習スクリプト実装完了

**✅ Phase 1.5: 学習設定の最適化（実装完了）** - 2025/10/14
- ✅ PRAUC評価指標の実装（`metrics.py`）
- ✅ 最適閾値探索の修正（`model_module.py`）
  - validation_stepでprobabilitiesをCPUに保存（GPUメモリ効率化）
  - on_validation_epoch_endでPRAUC計算とF1最大化による閾値最適化
  - best_thresholdを`self.threshold`と`self.hparams.threshold`に保存
  - DDP（分散学習）対応の全プロセス集約機能追加
- ✅ 学習設定ファイル更新（`train.yaml`）
  - モニタリング指標をval_optimal_f1 → val_praucに変更
  - EarlyStoppingとModelCheckpointをval_praucベースに変更
- 次: セットアップテスト → 学習実行

---

## **Phase 0.5: 探索的データ分析の結果**

### **実施済み分析**

#### 1. 画像サイズ分析
- **スクリプト**: `notebook/exploratory_visualization.py`
- **結果**:
  - Height範囲: 確認済み
  - Width範囲: 確認済み
  - 椎体ごとのサイズ分布: 確認済み
- **結論**: 全画像リサイズ

#### 2. HU値ウィンドウイング分析
- **スクリプト**: `notebook/hu_value_exploration.py`
- **出力先**: `notebook/exploratory_image/`
- **検証項目**:
  - ✅ 6種類のHU値ウィンドウ設定の視覚的比較
  - ✅ 複数サンプルでの推奨ウィンドウ検証
  - ✅ 5種類の正規化方法の比較
  - ✅ 骨折領域vs非骨折領域のHU値統計比較

#### 3. データ分布分析 - 2025/10/12
- **スクリプト**: `notebook/data_distribution_analysis.py`
- **結果**:
  - Train: 45,815スライス（骨折 10.83%）
  - Test: 13,691スライス（骨折 9.39%）
  - 骨折ピクセル比率: 平均 0.616% (Train), 0.553% (Test)
  - 椎体別骨折率: V30-V32が高い（20-25%）、V27/V40が低い（1-3%）
- **結論**:
  - 深刻なクラス不均衡（骨折 10.8% vs 非骨折 89.2%）
  - 骨折領域が極小（画像の0.6%）→ Attention機構必須
  - 骨折スライスのオーバーサンプリング必要（約9倍）

#### 4. 分析結果サマリー

**推奨HU値ウィンドウ**:
- **第1推奨**: [0, 1800] - 骨組織と骨折の視認性バランスが最良
- **第2推奨**: [-200, 300] - やや軟部組織も含む
- **代替案**: [200, 1200] - 高密度骨組織に特化
  これらを3チャンネル入力にする->すべて香料する必要があるから

**推奨正規化方法**:
```python
normalized = np.clip(HU, 0, 1800) / 1800  # [0, 1]に正規化
```

**確定データ前処理パイプライン** ✅:
1. NIfTI画像読み込み
2. HU値を[0, 1800],[-200, 300],[200, 1200]にクリップ
3. [0, 1]に正規化
4. 全画像256×256にリサイズ
5. Data Augmentation適用（オンライン、椎体単位）
   - 回転: ±15度
   - 平行移動: ±10ピクセル
   - スケーリング: 0.95-1.05倍
   - 水平反転: 適用
   - 輝度: ±50 HU
   - コントラスト: 0.95-1.05倍
6. クラス比率均衡化（骨折スライス50%目標）

---


---

## **Phase 1: U-Netベースライン構築の実装詳細**

### **実装完了コンポーネント** - 2025/10/12

#### 1. Hydra設定ファイル ✅
- **場所**: `run/conf/`
- **ファイル**:
  - `config.yaml` - メイン設定
  - `constants.yaml` - 定数定義（HU範囲、画像サイズ等）
  - `dir/local.yaml` - ディレクトリパス
  - `train.yaml` - 学習ハイパーパラメータ
  - `model/attention_unet.yaml` - モデル設定
  - `split/fold_0.yaml` - フォールド分割
- **特徴**:
  - 3チャンネルHU入力: [0,1800], [-200,300], [200,1200]
  - 画像サイズ: 256×256
  - オーバーサンプリング: 9倍（骨折スライス50%目標）

#### 2. Dataset/DataLoader ✅
- **場所**: `src/datamodule/`
- **ファイル**:
  - `dataset.py` - VertebralFractureDataset
  - `dataloader.py` - VertebralFractureDataModule
- **機能**:
  - 3チャンネルHU入力生成
  - オンラインデータ拡張（回転、平行移動、スケール、反転、輝度、コントラスト）
  - 骨折スライスのオーバーサンプリング
  - 患者レベルでのtrain/val分割（データリーケージ防止）
  - 5-fold cross validation対応

#### 3. モデル実装 ✅
- **場所**: `src/model/`
- **ファイル**:
  - `attention_gate.py` - Attention Gate実装
  - `attention_unet.py` - Attention U-Net実装
- **アーキテクチャ**:
  - エンコーダ: 4層（64→128→256→512）
  - ボトルネック: 1024チャンネル
  - デコーダ: 4層（Attention Gate付き）
  - 入力: (B, 3, 256, 256)
  - 出力: (B, 1, 256, 256)
- **特徴**:
  - Attention機構でスキップ接続を重み付け
  - BatchNorm + Dropout
  - Kaiming初期化

#### 4. LightningModule ✅
- **場所**: `src/modelmodule/`
- **ファイル**:
  - `model_module.py` - SegmentationModule
  - `losses.py` - DiceLoss, CombinedLoss
  - `metrics.py` - Dice, IoU, Precision, Recall, F1
- **損失関数**: Dice Loss (0.5) + BCE Loss (0.5)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau (monitor=val_dice)
- **評価指標**: Dice, IoU, Precision, Recall, F1

#### 5. 学習スクリプト ✅
- **場所**: `run/scripts/train/train.py`
- **機能**:
  - Hydra設定読み込み
  - DataModule/Model初期化
  - W&Bロガー設定
  - ModelCheckpoint（best/lastモデル保存）
  - EarlyStopping（patience=10）
  - 学習実行

---

### **次のステップ**

1. **セットアップテスト**:
   ```bash
   cd vertebrae_Unet
   python run/scripts/test_setup.py
   ```

2. **学習開始**:
   ```bash
   cd vertebrae_Unet
   python run/scripts/train/train.py
   ```

3. **パラメータ変更例**:
   ```bash
   python run/scripts/train/train.py \
     experiment.name=test_run \
     training.batch_size=8 \
     training.max_epochs=50
   ```

---

## **Phase 1.5: 学習設定の最適化 詳細** - 2025/10/14

### **実施した修正内容**

#### 1. PRAUC評価指標の追加 ✅
**ファイル**: `src/modelmodule/metrics.py`

**実装内容**:
- `pr_auc_score()` 関数を追加
- Precision-Recallカーブの面積（PRAUC）を計算
- クラス不均衡タスク（骨折10.8%）に適した評価指標
- sklearn.metrics.aucを使用したトラペゾイド法による計算

**技術仕様**:
```python
def pr_auc_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    # Sigmoid適用 → CPU移動 → numpy変換
    # 予測スコアで降順ソート
    # TP/FP累積カウント → Precision/Recall計算
    # AUC計算（台形則）
```

#### 2. 最適閾値探索の修正 ✅
**ファイル**: `src/modelmodule/model_module.py`

**修正内容**:
- **validation_step**:
  - `torch.sigmoid(predictions)` でprobabilitiesを計算
  - `.detach().cpu()` でCPUに移動（GPUメモリ効率化）
  - logitsではなくprobabilitiesを保存

- **on_validation_epoch_end**:
  - 全プロセス集約（DDP対応）: `self.all_gather()`
  - PRAUC計算: `pr_auc_score(all_probabilities, all_targets)`
  - 閾値探索: F1スコア最大化で最適閾値決定
  - **重要**: `self.threshold = best_threshold` でインスタンス変数更新
  - **重要**: `self.hparams.threshold = best_threshold` でhparams保存
  - ログ: `val_prauc`, `val_optimal_threshold`, `val_optimal_*`

**修正理由**:
- **問題1**: 閾値が更新されていなかった → `self.threshold`に反映
- **問題2**: F1よりPRAUCが適切 → PRAUC計算追加
- **問題3**: GPUメモリ不足リスク → CPU移動
- **問題4**: DDP未対応 → all_gather実装

#### 3. 学習設定ファイルの更新 ✅
**ファイル**: `run/conf/train.yaml`

**変更内容**:
```yaml
# Before
scheduler:
  monitor: val_optimal_f1
checkpoint:
  monitor: val_optimal_f1
  filename: 'epoch={epoch:02d}-val_optimal_f1={val_optimal_f1:.4f}'
early_stopping:
  monitor: val_loss

# After
scheduler:
  monitor: val_prauc  # ← PRAUC最大化
checkpoint:
  monitor: val_prauc  # ← PRAUC最大化
  filename: 'epoch={epoch:02d}-val_prauc={val_prauc:.4f}'
early_stopping:
  monitor: val_prauc  # ← PRAUC最大化
```

### **期待される効果**

1. **クラス不均衡への対応**
   - PRAUCは不均衡データ（骨折10.8%）に適した評価指標
   - F1スコアよりも少数クラスの検出性能を適切に評価

2. **閾値の動的最適化**
   - エポック毎に最適閾値を更新
   - 学習進行に応じた閾値調整

3. **メモリ効率の向上**
   - GPUメモリに保持せずCPUで閾値探索
   - 大規模バッチでの学習が可能

4. **分散学習対応**
   - 複数GPUでの学習時も正確な評価
   - all_gatherで全プロセスのデータを集約

### **技術的考慮事項**

**PRAUC vs F1の使い分け**:
- **PRAUC**: 閾値非依存、全体的な性能評価（モデル選択に使用）
- **F1**: 閾値依存、閾値最適化に使用（各閾値候補でF1計算）
- 本実装: PRAUCでモデル評価、F1で最適閾値探索

**メモリ管理**:
- validation_step: CPU保存（`detach().cpu()`）
- on_validation_epoch_end: numpy変換（PRAUC計算）
- threshold_candidates: 50点（デフォルト）

**DDP動作**:
- `self.trainer.world_size > 1` でDDP検出
- `all_gather()` で全プロセス集約
- `reshape()` で正規化

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
- HU範囲: 3チャンネル入力（constants.yamlで定義）
- 画像サイズ: 256×256リサイズ
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
- **入力**: (B, 3, H, W) - バッチ、チャンネル、高さ、幅
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


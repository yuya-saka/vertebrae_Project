## 学習アーキテクチャや設計方法の検討

### 2025/10/18 - YOLO+LSTM実装計画

#### 全体実装フェーズ

**Phase 1: データセット作成とデータローダー実装** (優先度: 最高)

1. **YOLO形式データセット作成**
   - 目的: スライス画像→YOLO形式アノテーション変換
   - 入力: `data/slice_train/axial/`, `data/slice_train/axial_mask/`
   - 出力: YOLO形式(`images/`, `labels/`)
   - 実装内容:
     - マスク画像から骨折領域のバウンディングボックス抽出
     - YOLO形式テキストファイル生成 (`<class> <x_center> <y_center> <width> <height>`)
     - 椎体ごとのクラス分類 (T4~L5: 14クラス) or 骨折/非骨折 (1クラス)
     - train/val split (患者レベル分割、5-fold CV対応)

2. **データローダー実装 (2種類必要)**

   **【YOLO学習用データローダー】**
   - 目的: スライスレベルの骨折検出器を学習
   - 1サンプル = 1枚のスライス画像（3チャンネル入力）
   - 全症例・全椎体の全スライスを独立サンプルとして扱う
   - データ拡張: 回転、反転、明度調整、ランダムクロップ
   - バッチ構成: [Batch_size, C, H, W] (例: [32, 3, 256, 256])
     - C=3: Bone Window, Soft Tissue Window, Wide Windowの3チャンネル

   **【LSTM学習用データローダー】**
   - 目的: 椎体レベルの時系列統合を学習
   - 1サンプル = 1椎体の全スライス (実データ: 50~100枚の可変長)
   - 固定長シーケンスに調整:
     - 長い場合: 中心部優先サンプリング or 均等間隔サンプリング
     - 短い場合: 後方パディング (最終スライス複製 or ゼロパディング)
   - **スライディングウィンドウは使用しない** (椎体全体を1サンプルとする)
   - バッチ構成: [Batch_size, N_slices, C, H, W] (例: [4, 70, 3, 256, 256])
     - C=3: 各スライスが3チャンネル（Bone, Soft Tissue, Wide Window）

   **【共通事項】**
   - 患者レベル分割の厳守: 同一患者のスライスが train/val に跨がらない
   - 両データローダーで**同じ患者分割**を使用 (5-fold CV対応)

**Phase 2: YOLO+LSTMアーキテクチャ設計** (優先度: 最高)

1. **アーキテクチャ選択: YOLOv8 + LSTM 2段階学習アプローチ（確定）**

   **学習・推論フロー:**
   ```
   ============================================
   【学習フェーズ1: YOLO事前学習】
   ============================================
   目的: スライスレベルの骨折検出器を作成

   入力: 全スライス画像 (全症例×全椎体×全スライス) を独立サンプル扱い
        例: 38症例 × 14椎体/症例 × 70スライス/椎体 ≈ 37,000サンプル
     ↓
   YOLOv8 (事前学習済み重み初期化: ImageNet/COCO)
     ↓
   学習タスク: BBox検出 (YOLO標準損失)
   - BBox Regression Loss (IoU Loss)
   - Classification Loss (Focal Loss)
   - Objectness Loss
     ↓
   出力: 学習済みYOLOモデル (checkpoint保存)
        → この段階でスライスレベル評価 (ベースライン性能確認)

   ============================================
   【学習フェーズ2: LSTM学習】
   ============================================
   目的: 椎体レベルの骨折分類器を作成

   YOLOの重み: **固定** (requires_grad=False)
              または低学習率でファインチューニング

   入力: 1椎体の全スライス [N_slices, 1, H, W]
        例: N_slices ≈ 50~100 (可変長 → 固定長70に調整)
     ↓
   学習済みYOLOで特徴抽出 (順伝播のみ)
     各スライス → 特徴ベクトル [256-dim]
     ↓
   時系列特徴: [N_slices, 256]
     ↓
   LSTM (Hidden_dim=256, Layers=2, Bidirectional=False)
     ↓
   最終隠れ状態: [256]
     ↓
   分類器 (Linear): [256] → [2] (骨折/非骨折)
     ↓
   学習タスク: 椎体レベルのクロスエントロピー損失
     ↓
   出力: YOLO+LSTMモデル全体 (checkpoint保存)

   ============================================
   【推論フェーズ】
   ============================================
   入力: テストデータの1椎体の全スライス
     ↓
   YOLO (固定) → 特徴抽出 [N, 256]
     ↓
   LSTM → 時系列統合 [256]
     ↓
   分類器 → 最終予測 (椎体レベル: 骨折確率)
   ```

   **この設計の利点:**
   - **少数データ対応**: YOLOは事前学習済み→汎化性能確保、LSTMは軽量な時系列統合のみ
   - **段階的検証**: YOLOベースライン（LSTM無し）→LSTM追加で効果を独立評価可能
   - **解釈性**: 各スライスのYOLO検出結果を明示的に確認可能（医療AI要件）
   - **計算効率**: BBox情報のみLSTMに渡す→ConvLSTMの4D畳み込みより軽量
   - **実装柔軟性**: 既存YOLOv8ライブラリを活用、カスタム実装の負担軽減

   **ConvLSTMを採用しない理由:**

   | 項目 | CNN + LSTM (採用) | ConvLSTM (不採用) |
   |------|-------------------|-------------------|
   | 学習パラメータ数 | 少（YOLOは事前学習済み） | 多（4D畳み込み全体を学習） |
   | 過学習リスク（38症例） | 低 | **高（データ不足）** |
   | 解釈性 | 高（各段階で検証可能） | 低（統合ブラックボックス） |
   | 実装難易度 | 中（既存ライブラリ活用） | 高（カスタム実装必要） |
   | GPU メモリ効率 | 良（数値データ処理） | **悪（4D特徴マップ処理）** |
   | アブレーション実験 | 容易（YOLO/LSTM独立評価） | 困難（統合モデル） |

2. **バックボーン選択肢と比較**

   YOLOv8はデフォルトのCSPDarknet以外にも、様々なバックボーンへの置き換えが可能です。
   38症例という少数データ環境では、事前学習済みバックボーンの選択が性能を大きく左右します。

   **使用可能なバックボーン:**

   | バックボーン | パラメータ数 | 事前学習 | 医療画像実績 | 38症例での推奨度 |
   |------------|-------------|---------|------------|----------------|
   | **EfficientNet-B0** | 5.3M | ImageNet | 高 | ⭐⭐⭐ |
   | **EfficientNet-B1** | 7.8M | ImageNet | 高 | ⭐⭐⭐ |
   | **ResNet-50** | 25.6M | ImageNet | 高 | ⭐⭐⭐ |
   | **CSPDarknet (デフォルト)** | 7.2M (n) | COCO | 中 | ⭐⭐ |
   | **ResNet-18** | 11.7M | ImageNet | 中 | ⭐⭐ |
   | **MobileNetV3** | 5.5M | ImageNet | 低 | ⭐ |
   | **EfficientNet-B3以上** | 12M+ | ImageNet | 高 | ⭐ (過学習リスク大) |

   **推奨バックボーンと理由:**

   1. **EfficientNet-B0/B1（最推奨）**
      - パラメータ効率が極めて良く、少数データに最適
      - 複合スケーリング手法により高精度と軽量性を両立
      - 医療画像（CT/MRI）での骨折検出研究で実績多数
      - 事前学習済み重みが豊富（ImageNet, COCO等）

   2. **ResNet-50（次点推奨）**
      - 深層学習の標準的バックボーン、汎化性能が高い
      - 残差接続により勾配消失を防ぎ、安定した学習が可能
      - PyTorch/TorchVisionの事前学習済み重みが充実
      - 医療AI分野で最も実績が多い

   3. **CSPDarknet（ベースライン）**
      - YOLOv8のデフォルト、まず性能確認用として使用
      - COCO検出タスクに最適化されているが、医療画像では未知数
      - EfficientNet/ResNetとの比較実験で効果を検証

   **実装方法:**

   ```yaml
   # run/conf/model/yolo_custom_backbone.yaml
   model:
     backbone:
       type: efficientnet_b1  # or resnet50, cspdarknet
       pretrained: true
       freeze_layers: [0, 1, 2]  # 初期3層を凍結（少数データ対策）
       out_indices: [2, 3, 4]    # 特徴マップ出力層の選択

     neck:
       type: YOLOv8PAFPN
       in_channels: [40, 112, 320]  # EfficientNet-B1の出力チャンネル
       # ResNet-50の場合: [512, 1024, 2048]
       # CSPDarknetの場合: YOLOv8デフォルト設定
   ```

   **実装上の注意点:**

   1. **チャンネル数の整合性**: バックボーンの出力チャンネルとNeck層の入力チャンネルを一致させる
   2. **事前学習重みの活用**: `pretrained=true` で必ずImageNet等の重みをロード
   3. **層の凍結（Transfer Learning）**: 少数データでは初期層を凍結し、後半層のみファインチューニング
   4. **バッチ正規化の調整**: 凍結層のBatchNormは `eval()` モードに固定

   **アブレーション実験計画:**

   段階的にバックボーンを比較し、最適な選択肢を探索します。

   | 実験ID | バックボーン | 凍結層 | 目的 |
   |--------|------------|--------|------|
   | exp_001 | CSPDarknet (yolov8n) | なし | ベースライン性能確認 |
   | exp_002 | EfficientNet-B0 | [0,1,2] | 軽量バックボーンの効果検証 |
   | exp_003 | EfficientNet-B1 | [0,1,2] | B0との性能比較 |
   | exp_004 | ResNet-50 | [0,1,2,3] | 標準バックボーンとの比較 |
   | exp_005 | 最良モデル | 最適化 | 凍結層数のハイパラ調整 |

   **評価指標**: mAP@0.5, mAP@0.5:0.95, 推論速度、GPU メモリ使用量

3. **損失関数設計**
   - YOLOv8標準損失: BBox loss + Classification loss + Objectness loss
   - 不均衡対策: Focal Loss / 重み付けサンプリング
   - 椎体一括学習: 全椎体を独立サンプルとして扱う

**Phase 3: 学習パイプライン実装** (優先度: 高)

1. **PyTorchベースの学習ループ + Hydra構成**
   ```yaml
   model:
     # バックボーン設定（カスタマイズ可能）
     backbone:
       type: cspdarknet  # efficientnet_b0, efficientnet_b1, resnet50
       variant: yolov8n  # yolov8s, yolov8m (CSPDarknetの場合)
       pretrained: true
       freeze_layers: []  # 例: [0, 1, 2] で初期3層を凍結

     # LSTM設定
     lstm:
       hidden_dim: 256
       num_layers: 2
       bidirectional: false
       dropout: 0.3

     # 分類設定
     num_classes: 2  # 骨折/非骨折 (二値分類)
     # num_classes: 14  # 椎体別分類 (T4~L5) - 将来的に拡張

   data:
     # 入力画像設定
     input_channels: 3  # Bone Window, Soft Tissue Window, Wide Window
     hu_windows:
       bone:
         center: 1100
         width: 1400
       soft_tissue:
         center: 100
         width: 400
       wide:
         center: 150
         width: 700

     # YOLO学習時のデータ設定
     yolo_training:
       batch_size: 32  # スライス単位のバッチ
       num_workers: 4
       image_size: [256, 256]
       augmentation: true
       sampling_balance: true  # 骨折/非骨折のバランスサンプリング

     # LSTM学習時のデータ設定
     lstm_training:
       batch_size: 4   # 椎体単位のバッチ (メモリ消費大)
       num_workers: 2
       max_slices_per_vertebra: 70  # 椎体あたり最大スライス数
       sampling_strategy: center_crop  # center_crop, uniform_sample, full
       padding_mode: replicate  # replicate, zero
       image_size: [256, 256]

   training:
     # YOLO学習設定
     yolo_phase:
       epochs: 100
       optimizer: AdamW
       lr: 0.001
       weight_decay: 0.0001
       scheduler: CosineAnnealingLR
       early_stopping_patience: 15

     # LSTM学習設定
     lstm_phase:
       epochs: 50
       freeze_yolo: true  # YOLOの重みを固定
       optimizer: AdamW
       yolo_lr: 0.00001  # freeze_yolo=false時のYOLO学習率
       lstm_lr: 0.001    # LSTMと分類器の学習率
       weight_decay: 0.0001
       scheduler: ReduceLROnPlateau
       early_stopping_patience: 10

   # アブレーション実験用バックボーン比較リスト
   experiment:
     backbones_to_compare:
       - cspdarknet_yolov8n  # ベースライン
       - efficientnet_b0
       - efficientnet_b1
       - resnet50
   ```

2. **学習スクリプト**
   - 5-fold Cross Validation: `run/scripts/train/train.py`
   - シンプルなPyTorchの学習ループ
   - Checkpointing: Best model保存 (mAP基準)
   - Logging: W&B / TensorBoard
   - Early Stopping、LR Schedulerを自前実装

**Phase 4: 推論・評価パイプライン** (優先度: 高)

1. **2D推論**
   - スライスごとの検出結果保存
   - 信頼度スコアと共にBBox座標を記録

2. **3D統合 (マルチオリエンテーション対応)**
   - 入力: axial/sagittal/coronalの各方向の検出結果
   - 統合手法の比較:
     1. 閾値ベース: 複数方向で検出された領域をAND/OR統合
     2. 信頼度スコア重み付け: スコアが高い検出を優先
     3. Non-Maximum Suppression (NMS): 重複検出の除去
     4. クラスタリング: DBSCAN等で近接検出をグループ化

3. **評価指標**
   - 2D評価: mAP@0.5, mAP@0.5:0.95, Precision, Recall
   - 3D評価: 症例レベルAUC, 椎体レベルF1スコア
   - 統計解析: 5-fold平均と標準偏差

**Phase 5: 実験・改善** (優先度: 中)

1. **アブレーション実験**
   - LSTM有無の比較
   - 連続スライス数の最適化 (N=3, 5, 7, 10)
   

2. **ハイパーパラメータ最適化**
   - Optuna等でLSTM隠れ層数、学習率を探索

#### 推奨ディレクトリ構造 (更新版)

```
vertebrae_YOLO/
├── data_preparing/
│   └── convert_to_yolo.py      # マスク→YOLO変換(data/に保存する)
├── src/
│   ├── models/
│   │   ├── yolo_lstm.py        # YOLO+LSTMモデル
│   │   └── yolo_baseline.py   # ベースライン (LSTM無し)
│   ├── dataset/                # データセットクラス
│   │   └── yolo_dataset.py     # PyTorch Dataset
│   └── utils/
│       ├── trainer.py          # 学習ユーティリティ
│       └── metrics.py          # 評価指標計算
├── run/
│   ├── conf/
│   │   ├── config.yaml
│   │   ├── train.yaml
│   │   ├── model/yolo_lstm.yaml
│   │   └── split/fold_0~4.yaml
│   └── scripts/
│       ├── train.py            # シンプルなPyTorch学習ループ
│       ├── inference.py
│       └── reconstruct_3d.py
└── output/
    ├── train/{exp_name}/
    └── inference/{exp_name}/
```

#### 重要な実装上の注意点

1. **患者レベル分割の徹底**: データリーケージ防止
2. **LSTM入力設計**: パディング/トランケーションの戦略
3. **メモリ管理**: バッチサイズと連続スライス数のトレードオフ
4. **不均衡データ対策**: 骨折/非骨折の極端な偏り
5. **再現性**: シード固定 + Hydraでの設定管理

#### 次のステップ (2段階学習の実装順序)

**【Phase A: YOLO単体の学習と評価】**

1. **Step A-1**: マスク画像からYOLO形式アノテーション作成
   - 入力: `data/slice_train/axial_mask/`
   - 出力: `data/yolo_dataset/images/`, `data/yolo_dataset/labels/`
   - マルチインスタンス対応 (1スライスに複数BBox)

2. **Step A-2**: YOLO学習用DataLoader実装
   - スライス単位のデータセット
   - 患者レベル分割 (5-fold CV)
   - データ拡張 (回転、反転、明度調整)

3. **Step A-3**: YOLOv8ベースライン学習
   - バックボーン: CSPDarknet (yolov8n) から開始
   - BBox検出タスク (YOLO標準損失)
   - Checkpoint保存

4. **Step A-4**: YOLO単体での推論と評価
   - 各スライスの検出結果を保存 (BBox + 信頼度スコア)
   - スライスレベル評価: mAP@0.5, mAP@0.5:0.95
   - 椎体レベル評価: 多数決による骨折判定 (ベースライン性能確認)
   - **目標**: この段階でベースライン性能を確立

**【Phase B: LSTM統合と最終評価】**

5. **Step B-1**: LSTM学習用椎体DataLoader実装
   - 椎体単位のデータセット (1サンプル = 1椎体の全スライス)
   - 可変長 → 固定長変換 (max_slices=70)
   - サンプリング戦略とパディング実装

6. **Step B-2**: YOLO+LSTMモデル統合
   - 学習済みYOLOの重みをロード
   - LSTM層と分類器を追加
   - YOLO重みの固定設定

7. **Step B-3**: LSTM学習 (YOLO重みは固定)
   - 椎体レベルのクロスエントロピー損失
   - LSTMと分類器のみを学習
   - Checkpoint保存

8. **Step B-4**: エンドツーエンド推論と評価
   - 椎体レベル評価: Accuracy, F1, AUC
   - YOLOのみ vs YOLO+LSTM の性能比較

9. **Step B-5**: アブレーション実験と最適化
   - スライス数の最適化 (max_slices=50, 70, 100)
   - サンプリング戦略の比較 (center_crop vs uniform_sample)
   - バックボーン比較 (EfficientNet-B0/B1, ResNet-50)
   - エンドツーエンドファインチューニング (freeze_yolo=false)

**【Phase C: 推論・3D統合 (将来実装)】**

10. **Step C-1**: 推論・3D統合スクリプト実装
11. **Step C-2**: 評価指標計算と可視化

#### 検討が必要な設計判断

1. **椎体クラス分類**: 14クラス (T4~L5個別) vs 2クラス (骨折/非骨折)
   - **推奨**: まず2クラス（骨折/非骨折）で実装→精度確認後に14クラスへ拡張
   - 理由: 38症例で14クラスはデータ不足、まず骨折検出精度を確立

2. **椎体あたりスライス数の調整方法**
   - **実データ**: 1椎体あたり50~100スライス (可変長)
   - **固定長化**: `max_slices_per_vertebra=70` を推奨 (平均的な椎体サイズ)
   - **長い椎体の処理 (N > max_slices)**:
     - `center_crop`: 中心部優先サンプリング (骨折は椎体中心に多い)
     - `uniform_sample`: 均等間隔サンプリング (全体の情報を保持)
   - **短い椎体の処理 (N < max_slices)**:
     - `replicate`: 後方パディング (最終スライス複製)
     - `zero`: ゼロパディング
   - **アブレーション実験**: max_slices=50, 70, 100 で比較

3. **マルチオリエンテーション**: 最初からaxial/sagittal/coronal全方向 vs axialのみで先行実装
   - **推奨**: axialのみで先行実装→3D統合手法確立後に他方向追加
   - 理由: 段階的検証、計算コスト削減

**確定事項:**
- **アーキテクチャ**: YOLOv8 + LSTM 2段階アプローチ（ConvLSTMは不採用）

**推奨実装順序**: Step 1 (YOLO形式変換) から開始

---

## 設計決定の変更履歴

### 2025/10/20 - 3チャンネルHUウィンドウ入力の採用

**入力データ設計の変更:**
- **決定**: 3つの異なるHU値ウィンドウで処理した画像を3チャンネル（RGB）として入力
- **チャンネル構成**:
  - R (赤): Bone Window (WW=1400, WL=1100) - 骨構造の可視化
  - G (緑): Soft Tissue Window (WW=400, WL=100) - 軟部組織の可視化
  - B (青): Wide Window (WW=700, WL=150) - 全体のバランス
- **理由**:
  - 骨組織と軟部組織の情報を同時に活用可能
  - ImageNet事前学習済みバックボーン（RGB 3チャンネル）との整合性
  - 医療画像解析における標準的手法で検出精度向上が期待できる
- **実装への影響**:
  - `convert_to_yolo.py`の`normalize_and_pad_image()`を修正
  - データローダーのバッチ形状: [B, 3, H, W] (従来: [B, 1, H, W])
  - YOLOv8モデルはデフォルトで3チャンネル対応のため変更不要

### 2025/10/19 - アーキテクチャ決定とバックボーン選択肢の追加

**アーキテクチャ決定:**
- **決定**: YOLOv8 + LSTM 2段階アプローチを採用（ConvLSTMは不採用）
- **理由**:
  - 38症例という少数データでの過学習リスク軽減
  - 事前学習済みYOLOv8の活用による汎化性能確保
  - 医療AI要件（解釈性・検証容易性）への対応
  - 段階的実装とアブレーション実験の実現可能性
  - GPU メモリ効率とパラメータ削減

**バックボーン選択肢の追加:**
- **追加内容**: YOLOv8バックボーンのカスタマイズ可能性を明記
- **推奨バックボーン**: EfficientNet-B0/B1, ResNet-50（少数データに最適）
- **実装計画**: CSPDarknetベースライン→EfficientNet/ResNet比較実験
- **Transfer Learning戦略**: 事前学習済み重み活用 + 初期層凍結

### 2025/10/18 - YOLO+LSTM実装計画策定
- 全体実装フェーズ（Phase 1-5）を定義
- ディレクトリ構造とデータフロー設計

### 2025/10/17
- LSTMによる時系列統合の検討開始

### 2025/10/16
- Attention機構の検討（→LSTM優先のため保留）
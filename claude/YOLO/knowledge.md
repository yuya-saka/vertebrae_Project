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

2. **連続スライス用データローダー**
   - LSTM用の時系列データ構造:
     - 1サンプル = 連続N枚のスライス (例: N=5~10枚)
     - スライディングウィンドウで連続性を保持
     - データ拡張: 回転、反転、明度調整
   - 患者レベル分割の厳守: 同一患者のスライスが train/val に跨がらない

**Phase 2: YOLO+LSTMアーキテクチャ設計** (優先度: 最高)

1. **アーキテクチャ選択: YOLOv8 + ConvLSTM融合モデル**
   ```
   入力: 連続スライス [N, C, H, W]
     ↓
   YOLOバックボーン (特徴抽出)
     ↓ [N, Feature_dim, H', W']
   ConvLSTM層 (時系列特徴学習)
     ↓ [N, Hidden_dim, H', W']
   YOLO Detection Head (BBox予測)
     ↓
   出力: [cls, x, y, w, h, conf]
   ```
   - 代替案: YOLOv8の各スライス検出 → LSTM後処理で時系列統合

2. **損失関数設計**
   - YOLOv8標準損失: BBox loss + Classification loss + Objectness loss
   - 不均衡対策: Focal Loss / 重み付けサンプリング
   - 椎体一括学習: 全椎体を独立サンプルとして扱う

**Phase 3: 学習パイプライン実装** (優先度: 高)

1. **PyTorch Lightning + Hydra構成**
   ```yaml
   model:
     backbone: yolov8n  # or yolov8s
     lstm_hidden: 256
     lstm_layers: 2
     num_classes: 1  # or 1 (binary)

   data:
     batch_size: 8
     num_workers: 4
     sequence_length: 7  # 連続スライス数

   training:
     epochs: 100
     optimizer: AdamW
     lr: 0.001
     scheduler: CosineAnnealingLR
   ```

2. **学習スクリプト**
   - 5-fold Cross Validation: `run/scripts/train/train.py`
   - Checkpointing: Best model保存 (mAP基準)
   - Logging: W&B / TensorBoard

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
   - 椎体一括学習 vs 症例単位学習

2. **ハイパーパラメータ最適化**
   - Optuna等でLSTM隠れ層数、学習率を探索

#### 推奨ディレクトリ構造 (更新版)

```
vertebrae_YOLO/
├── data_preparation/
│   ├── convert_to_yolo.py      # マスク→YOLO変換
│   └── create_dataset.py       # Dataset/DataLoader定義
├── src/
│   ├── models/
│   │   ├── yolo_lstm.py        # YOLO+LSTMモデル
│   │   └── yolo_baseline.py   # ベースライン (LSTM無し)
│   ├── datamodule/
│   │   └── vertebrae_datamodule.py
│   └── modelmodule/
│       └── yolo_module.py      # Lightning Module
├── run/
│   ├── conf/
│   │   ├── config.yaml
│   │   ├── train.yaml
│   │   ├── model/yolo_lstm.yaml
│   │   └── split/fold_0~4.yaml
│   └── scripts/
│       ├── train/train.py
│       ├── inference/inference.py
│       └── inference/reconstruct_3d.py
└── output/
    ├── datasets/yolo_format/   # YOLO変換後データ
    ├── train/{exp_name}/
    └── inference/{exp_name}/
```

#### 重要な実装上の注意点

1. **患者レベル分割の徹底**: データリーケージ防止
2. **LSTM入力設計**: パディング/トランケーションの戦略
3. **メモリ管理**: バッチサイズと連続スライス数のトレードオフ
4. **不均衡データ対策**: 骨折/非骨折の極端な偏り
5. **再現性**: シード固定 + Hydraでの設定管理

#### 次のステップ (推奨実装順序)

1. **Step 1**: マスク画像からYOLO形式アノテーション作成
2. **Step 2**: 連続スライス用DataLoader実装
3. **Step 3**: YOLOv8 + ConvLSTMモデル実装
4. **Step 4**: PyTorch Lightning学習パイプライン構築
5. **Step 5**: 推論・3D統合スクリプト実装
6. **Step 6**: 評価指標計算と可視化

#### 検討が必要な設計判断

1. **椎体クラス分類**: 14クラス (T4~L5個別) vs 2クラス (骨折/非骨折)
2. **連続スライス数**: N=5, 7, 10のどれから始めるか
3. **YOLO統合方法**: YOLOバックボーン内にLSTM統合 vs YOLO後処理でLSTM適用
4. **マルチオリエンテーション**: 最初からaxial/sagittal/coronal全方向 vs axialのみで先行実装

**推奨**: Step 1 (YOLO形式変換) から開始

---

### 2025/10/16
- YOLOにもattention機構つけたらいいのでは？

### 2025/10/17 
- YOLOにあとでLSTMか？
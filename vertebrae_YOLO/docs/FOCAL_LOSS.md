# Focal Loss 実装ガイド

## 📋 概要

Focal Lossは、クラス不均衡問題に対処するための損失関数です（Lin et al., 2017）。
本プロジェクトでは、骨折検出タスクにおける**骨折なし（多数派）vs 骨折あり（少数派）**の不均衡を解決するため、YOLOv8の分類損失にFocal Lossを導入しました。

### Focal Lossの数式

```
FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
```

**パラメータ:**
- **γ (gamma)**: Focusing parameter（易しいサンプルの重み削減）
- **α (alpha)**: Balancing factor（クラス不均衡補正）

---

## 🎯 期待される効果

### 1. 易しいサンプルの重み削減
- **背景領域（非骨折）**: 明確に骨折がない領域は損失が小さくなる
- **難しい骨折**: 微細な骨折線や境界が不明瞭な骨折に注力

### 2. クラス不均衡の補正
- **骨折あり（少数派）**: 重みを増加（alpha=0.3）
- **骨折なし（多数派）**: 重みを削減

### 3. 検出精度の向上
- **偽陰性（骨折の見逃し）**: 削減
- **Recall**: 向上
- **mAP@0.5**: 向上

---

## 🔧 実装ファイル

### 1. カスタム損失関数: `src/utils/custom_loss.py`

**FocalLossクラス:**
- Binary Cross Entropyをベースに、Focal Loss modulating factorを適用
- gamma, alphaパラメータで易しいサンプルの重みを制御

**CustomDetectionLossクラス:**
- YOLOv8の`v8DetectionLoss`を継承
- 分類損失のみをFocal Lossに置き換え
- BBox回帰損失とDFL損失は変更なし

### 2. トレーナー統合: `src/utils/trainer.py`

**CustomDetectionTrainer.get_model():**
- モデルロード時にFocal Lossを適用
- `use_focal_loss=true`の場合のみ有効化
- gamma, alphaパラメータをconfig.yamlから読み込み

**Trainer.fit():**
- Focal Loss設定をoverridesに追加
- CustomDetectionTrainerに渡す

### 3. 設定ファイル

**config.yaml:**
```yaml
training:
  use_focal_loss: true       # Focal Lossを使用
  focal_gamma: 2.0           # Focusing parameter
  focal_alpha: 0.3           # Balancing factor
```

**hyp_custom.yaml:**
```yaml
focal_gamma: 2.0    # Focusing parameter
focal_alpha: 0.3    # Balancing factor
```

---

## 🚀 使用方法

### Focal Lossを有効にする（デフォルト）

```bash
cd vertebrae_YOLO/run/scripts/train

# Focal Loss有効で学習（デフォルト設定）
uv run python train.py
```

学習開始時に以下のメッセージが表示されます:

```
============================================================
🔥 Applying Focal Loss to YOLOv8 Detection Loss
============================================================
  Focal Loss Parameters:
    - gamma (focusing parameter): 2.0
    - alpha (balancing factor): 0.3
  Expected Effects:
    ✓ Down-weights easy examples (non-fracture backgrounds)
    ✓ Focuses on hard examples (subtle fractures)
    ✓ Addresses class imbalance
============================================================

✅ CustomDetectionLoss initialized with Focal Loss
   - gamma (focusing parameter): 2.0
   - alpha (balancing factor): 0.3
   - Effect: Down-weights easy examples, focuses on hard negatives
```

### Focal Lossを無効にする

```bash
# config.yamlを編集
training:
  use_focal_loss: false  # Focal Lossを無効化
```

または、コマンドラインでオーバーライド:

```bash
uv run python train.py training.use_focal_loss=false
```

無効化時のメッセージ:

```
ℹ️  Using default YOLOv8 loss (BCEWithLogitsLoss)
   Set use_focal_loss=true in config.yaml to enable Focal Loss
```

### パラメータの調整

```bash
# gamma, alphaをカスタマイズ
uv run python train.py training.focal_gamma=1.5 training.focal_alpha=0.25
```

---

## 📊 パラメータの推奨値

### gamma (Focusing Parameter)

| 値 | 効果 | 推奨シーン |
|----|------|-----------|
| 1.5 | 軽度の重み削減 | クラス不均衡が軽微な場合 |
| **2.0** | **標準的な重み削減**（推奨） | **医療画像の骨折検出** |
| 2.5 | 強めの重み削減 | クラス不均衡が極端な場合 |

### alpha (Balancing Factor)

| 値 | 効果 | クラス比率 |
|----|------|-----------|
| 0.25 | 標準的な補正 | 骨折25%:非骨折75% |
| **0.3** | **やや強めの補正**（推奨） | **骨折30%:非骨折70%** |
| 0.4 | 強めの補正 | 骨折40%:非骨折60% |
| 0.5 | 最大の補正 | 骨折50%:非骨折50% |

**注意:** alphaはデータセットの実際のクラス比率に応じて調整してください。

---

## 🧪 アブレーション実験

Focal Lossの効果を検証するため、以下の実験を推奨します:

### 実験1: Focal Loss有無の比較

```bash
# Baseline: Focal Lossなし
uv run python train.py training.use_focal_loss=false \
  logging.experiment_name="baseline_no_focal"

# Focal Loss: あり（デフォルト設定）
uv run python train.py training.use_focal_loss=true \
  logging.experiment_name="focal_gamma2.0_alpha0.3"
```

### 実験2: gammaの最適化

```bash
# gamma=1.5
uv run python train.py training.focal_gamma=1.5 \
  logging.experiment_name="focal_gamma1.5"

# gamma=2.0（推奨）
uv run python train.py training.focal_gamma=2.0 \
  logging.experiment_name="focal_gamma2.0"

# gamma=2.5
uv run python train.py training.focal_gamma=2.5 \
  logging.experiment_name="focal_gamma2.5"
```

### 実験3: alphaの最適化

```bash
# alpha=0.25
uv run python train.py training.focal_alpha=0.25 \
  logging.experiment_name="focal_alpha0.25"

# alpha=0.3（推奨）
uv run python train.py training.focal_alpha=0.3 \
  logging.experiment_name="focal_alpha0.3"

# alpha=0.4
uv run python train.py training.focal_alpha=0.4 \
  logging.experiment_name="focal_alpha0.4"
```

---

## 📈 評価指標

Focal Lossの効果を確認するため、以下の指標を比較してください:

### 学習曲線（W&B/TensorBoard）
- **val_loss**: 安定した減少が期待される
- **train_loss vs val_loss**: 過学習の兆候を確認

### 検出精度
- **mAP@0.5**: 全体的な検出精度
- **mAP@0.5:0.95**: IoU閾値を変えた場合の精度
- **Precision**: 偽陽性（誤検出）の削減
- **Recall**: 偽陰性（骨折の見逃し）の削減 ← **Focal Lossの主要効果**

### 期待される改善
| 指標 | 期待される変化 |
|------|---------------|
| val_loss | ↓ 安定した減少 |
| Recall | ↑ 骨折の見逃し削減 |
| mAP@0.5 | ↑ 全体的な検出精度向上 |
| Precision | → または ↑ 偽陽性の削減 |

---

## 🔬 技術的な詳細

### Focal Lossの動作原理

**1. Binary Cross Entropy (BCE):**
```
BCE = -[y log(p) + (1-y) log(1-p)]
```

**2. Focal Loss Modulating Factor:**
```
p_t = y * p + (1-y) * (1-p)
modulating_factor = (1 - p_t)^gamma
```

**3. Alpha Balancing:**
```
alpha_t = y * alpha + (1-y) * (1-alpha)
```

**4. Focal Loss:**
```
FL = alpha_t * modulating_factor * BCE
```

### 易しいサンプルの重み削減の例

| p_t（正しいクラスの確率） | gamma=2.0での重み |
|-------------------------|------------------|
| 0.9（易しい） | (1-0.9)^2 = 0.01 |
| 0.7（中程度） | (1-0.7)^2 = 0.09 |
| 0.5（難しい） | (1-0.5)^2 = 0.25 |
| 0.3（非常に難しい） | (1-0.3)^2 = 0.49 |

→ 易しいサンプル（p_t=0.9）の重みは、難しいサンプル（p_t=0.3）の**約1/50**に削減される

---

## 📚 参考文献

1. **Lin et al., 2017**: "Focal Loss for Dense Object Detection"
   - 原論文: https://arxiv.org/abs/1708.02002
   - クラス不均衡問題に対処するFocal Lossを提案

2. **YOLOv8 Documentation**:
   - Ultralytics公式ドキュメント: https://docs.ultralytics.com/

3. **医療画像におけるFocal Lossの応用**:
   - 骨折検出、病変検出、異常検出などで広く使用

---

## 🐛 トラブルシューティング

### Focal Lossが適用されない

**症状:** 学習開始時に「Using default YOLOv8 loss」と表示される

**解決策:**
1. `config.yaml`の`use_focal_loss: true`を確認
2. `trainer.py`のimport文を確認（`from vertebrae_YOLO.src.utils.custom_loss import CustomDetectionLoss`）
3. `custom_loss.py`が正しく配置されているか確認

### gamma, alphaが反映されない

**症状:** カスタム値を設定したが、デフォルト値（2.0, 0.3）が使用される

**解決策:**
1. `config.yaml`の`focal_gamma`, `focal_alpha`を確認
2. コマンドラインオーバーライドの構文を確認（`training.focal_gamma=1.5`）
3. W&B/TensorBoardのログでパラメータ値を確認

### val_lossが改善しない

**症状:** Focal Lossを使用してもval_lossが下がらない

**解決策:**
1. **gamma, alphaを調整:** デフォルト値が最適とは限らない
2. **学習率を下げる:** `lr0=0.0005`に設定
3. **patienceを短縮:** `patience=10`に設定（過学習防止）
4. **データ拡張を確認:** 過度な変換が学習を妨げていないか
5. **5-fold CVで評価:** 単一foldの結果に依存しない

---

## ✅ まとめ

Focal Lossは以下の条件で特に有効です:

- ✅ **クラス不均衡が顕著**（骨折なし >> 骨折あり）
- ✅ **偽陰性（見逃し）が問題**（Recallを向上させたい）
- ✅ **易しいサンプルが損失を支配**（背景領域が多い）

**次のステップ:**
1. Fold 0で学習を実行
2. W&B/TensorBoardでval_loss曲線を確認
3. mAP, Precision, Recallを評価
4. 必要に応じてgamma, alphaを調整
5. 5-fold CVで最終評価

**学習コマンド:**
```bash
cd vertebrae_YOLO/run/scripts/train
uv run python train.py  # Focal Loss有効（デフォルト）
```

---

**実装日**: 2025/10/20
**実装者**: Claude Code
**バージョン**: v1.0

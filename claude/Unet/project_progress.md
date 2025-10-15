## 基本的に自分用の整理

### 2025/10/7
- Unetを使った実装に決定
- まず、リポジトリを整えるところから
  - trainとteatにデータわける
     ->テストデータに割り当てられた症例数: 8 症例、訓練データに割り当てられた症例数: 30 症例
     ->成功
### 2025/10/8
- ==ディレクトリ構造をしっかり考える必要がある==。ちゃんと役割を考えないと、膨大な設計になり管理できず4ぬから
- 設定を明示的にする。
	- HUは0から1800
	- 画像スケールをそろえる必要あり、cut_li.txtで作成した椎体ボリュームの大きさに差があることが判明
	  そのため、アスペクト比は変えずに、ゼロパディング等で調整する
	- wandbの設定
	- ちゃんとfoldを分けて実装するようにする
- LSTM学習のためには、椎体部位間で自然につながるようにしないといけない、
  そのため、データ切り出し時点でやり方を変える必要あり
### 2025/10/10
- まず、unetのみでのmodelを作り、学習、評価まで行うことを目標とする
     - そのため、まず、入力するためのデータ準備と、Unetのアーキテクチャと学習設定を考える
        - チャネル数どうするか
     - 画像可視化でデータ入力策検討
        - 256×256でリサイズによる正規化ありかも
        - 3チャンネルで、HU幅を変えて入力する
            候補：
            - 0,1800
            - 200,1200
            - -160, 240
            - -300, 800
        - スライス間比率1:9、骨折ピクセルの全体に対する比率　0.6%
     - 学習設定
        - データ拡張行う
          - 学習時に生成する（オンライン）拡張
          - スライス画像間のクラス比率を同じになるよう拡張する
          - スライス間連続性の保持
        - Focal lossやTversky loss

### 2025/10/12
- **データ分布分析完了**
  - ✅ Train/Testのクラス分布調査完了
    - Train: 骨折スライス 10.83% (4,963 / 45,815枚)
    - Test: 骨折スライス 9.39% (1,286 / 13,691枚)
  - ✅ 骨折マスクのピクセル比率調査完了
    - Train: 平均 0.616% (149.5ピクセル/枚)
    - Test: 平均 0.553% (142.0ピクセル/枚)
  - ✅ 椎体ごとの骨折分布確認
    - V30-V32で骨折が多い（20-25%）
    - V27, V40で骨折が少ない（1-3%）
  - スクリプト: `notebook/data_distribution_analysis.py`

- **データ拡張戦略の決定**
  - ✅ オンライン拡張の方針決定
  - ✅ クラス比率均衡化の方針決定
    - 目標: 骨折スライス 50% （現状10.8% → 50%）
    - 骨折スライスを約9倍にオーバーサンプリング
  - ✅ スライス間連続性保持の実装方針決定
    - 椎体単位で同じ変換パラメータを適用
    - 回転・平行移動・スケーリング・反転は全スライス統一
    - 輝度・ノイズは独立適用可能
  - 実装: `src/datamodule/augmentation.py`

- **Phase 1実装完了** - 2025/10/12
  - ✅ Hydra設定ファイル構築完了
    - HU範囲、画像サイズ、拡張パラメータの定義
  - ✅ データセット実装完了（Dataset/DataLoaderクラス）
    - 3チャンネルHU入力生成
    - オンライン拡張機能の統合
    - 骨折スライスのオーバーサンプリング（9倍）
    - 患者レベルでのtrain/val分割
  - ✅ Attention U-Netモデル実装完了
    - Attention Gate実装
    - U-Net本体実装（depth=4, init_features=64）
  - ✅ LightningModule実装完了
    - 損失関数（focal Tversky loss）
    - 評価指標（Dice, IoU, Precision, Recall, F1）
  - ✅ 学習スクリプト実装完了

- **次のステップ**
  - [ ] セットアップテストの実行
  - [ ] 学習の実行（fold 0から開始）
  - [ ] 学習結果の評価と分析


### 2025/10/14 - model_module.py の問題点と修正事項

---

## 🔴 緊急修正事項（学習実行中に発見）

### **問題点1: 閾値最適化のロジックで変数名が混乱している** 🔴
**ファイル**: `vertebrae_Unet/src/modelmodule/model_module.py:197-216`

**現状のコード**:
```python
best_prauc = 0.0  # ❌
best_metrics = None

for threshold in self.threshold_candidates:
    metrics = calculate_all_metrics(probs_gpu, targets_gpu, threshold)
    f1 = metrics['f1']

    if f1 > best_prauc:  # ❌ 
        best_prauc = f1
        best_threshold = threshold
        best_metrics = metrics
```

**問題**:
- PRAUCで最適化しているのに、f1を使ってしまっている
- PRAUCで適切に最適化するようにする


### **問題点2: Checkpoint filenameに重複した文字列** 🔴
**ファイル**: `vertebrae_Unet/run/conf/train.yaml:91`

**現状の出力**:
```
epoch=epoch=00-val_prauc=val_prauc=0.0366.ckpt
```

**問題**:
- `epoch=`と`val_prauc=`が二重に表示されている
- PyTorch Lightningが自動的にプレフィックスを付けている

**現在の設定**:
```yaml
filename: 'epoch={epoch:02d}-val_prauc={val_prauc:.4f}'
```

**修正案1（シンプル）**:
```yaml
filename: 'ep{epoch:02d}_prauc{val_prauc:.4f}'
```

**修正案2（可読性重視）**:
```yaml
filename: '{epoch:02d}-prauc_{val_prauc:.4f}'
```

**期待される出力**:
```
ep00_prauc0.0366.ckpt
# または
00-prauc_0.0366.ckpt
```

---

### **問題点3: 閾値探索範囲が狭すぎる** 🟡
**ファイル**: `vertebrae_Unet/run/conf/train.yaml:94-98`

**現状の設定**:
```yaml
threshold_optimization:
  enabled: true
  min_threshold: 0.01   # ❌ 下限が高すぎる
  max_threshold: 0.90   # 上限は適切
  num_candidates: 50    # 候補数は適切
```

**問題**:
- 極端なクラス不均衡（骨折領域0.6%）の場合、0.01以下の閾値が最適になる可能性が高い
- 現在のログでは`val_optimal_threshold=0.010`が選ばれている（下限値）
- より低い閾値が最適である可能性を探索できていない

**理論的根拠**:
- 骨折ピクセル比率: 平均0.616% (99.4%が非骨折)
- PRAUC最大化では、Precision-Recallのバランスが重要
- 極端な不均衡では、低い閾値でRecallを上げることがF1向上につながる
- 数値例（骨折1000px, 非骨折100万pxの場合）:
  - 閾値0.5: Precision=0.909, Recall=0.100, F1=0.181
  - 閾値0.1: Precision=0.500, Recall=0.500, F1=0.500
  - 閾値0.01: Precision=0.138, Recall=0.800, **F1=0.235**
  - 閾値0.001: Precision=0.019, Recall=0.950, F1=0.037

**修正案**:
```yaml
threshold_optimization:
  enabled: true
  min_threshold: 0.001  # ✅ 0.001まで下げる（1/1000）
  max_threshold: 0.95   # 0.95まで上げる
  num_candidates: 95    # より細かく探索
```

**探索範囲の例**:
```
[0.001, 0.011, 0.021, 0.031, ..., 0.941, 0.950]
約0.01刻みで95点探索
```

---

### **問題点4: GPU移動の効率が悪い** 🟢（最適化）
**ファイル**: `vertebrae_Unet/src/modelmodule/model_module.py:197-216`

**現状のコード**:
```python
for threshold in self.threshold_candidates:
    # 毎回GPU移動している
    if torch.cuda.is_available():
        probs_gpu = all_probabilities.cuda()
        targets_gpu = all_targets.cuda()
    else:
        probs_gpu = all_probabilities
        targets_gpu = all_targets

    metrics = calculate_all_metrics(probs_gpu, targets_gpu, threshold)
    # ...
```

**問題**:
- 50回のループで毎回GPU移動が発生
- CPUメモリからGPUメモリへの転送オーバーヘッド


---

### **問題点5: 予測確率分布のログが不足** 🟡（診断用）
**ファイル**: `vertebrae_Unet/src/modelmodule/model_module.py:132-168`

**現状**:
- モデルの予測確率分布が分からない
- 閾値0.01が妥当なのか、モデルの出力が低すぎるのか判断できない


---

## 📊 診断が必要な項目

### **PRAUC が非常に低い (0.06前後)** 🔍
**ログより**:
```
Epoch 8: val_prauc=0.0632
Epoch 12: val_prauc=0.0636
```

**問題**:
- PRAUC 0.06 はほぼランダム予測に近い（理想は1.0）
- Epoch 14でも改善が見られない

**考えられる原因**:
1. **データローダーのバグ** - マスクとCT画像の対応が間違っている
2. **損失関数の設定が不適切** - Focal Tversky Lossのパラメータ
3. **学習率の問題** - 高すぎるor低すぎる
4. **データ拡張が強すぎる** - モデルが学習できない
5. **モデルアーキテクチャの問題** - Attention機構が機能していない

**診断手順**:
1. ✅ **予測確率分布を確認** - 上記のログを追加
2. ✅ **データローダーの検証** - バッチを1つ可視化して確認
   ```python
   # テストスクリプト
   batch = next(iter(train_dataloader))
   print(f"Image shape: {batch['image'].shape}")
   print(f"Mask shape: {batch['mask'].shape}")
   print(f"Mask sum: {batch['mask'].sum()}")  # 0でないことを確認

   # 可視化
   import matplotlib.pyplot as plt
   plt.subplot(1, 2, 1)
   plt.imshow(batch['image'][0, 0].cpu(), cmap='gray')
   plt.subplot(1, 2, 2)
   plt.imshow(batch['mask'][0, 0].cpu(), cmap='hot')
   plt.savefig('debug_batch.png')
   ```

3. ✅ **損失関数のデバッグ** - 損失値が減少しているか確認
4. ✅ **学習率スケジューラの確認** - W&Bで学習率の推移を確認

---

- TESTデータにおける骨折ゼロの椎体
以下の6つの椎体で骨折マスクが全くありませんでした：
  V35: 0 / 981 (0.0%)
  V36: 0 / 1,020 (0.0%)
  V37: 0 / 1,064 (0.0%)
  V38: 0 / 1,087 (0.0%)
  V39: 0 / 994 (0.0%)
  V40: 0 / 913 (0.0%)
- TRAINデータ
  - すべての椎体（V27-V40）に少なくとも1枚以上の骨折マスクが存在
  - 最も少ないのはV27とV40で約2.9%
  重要な含意
  - データの偏り: TESTセットでV35-V40（下部腰椎）の骨折が全くないことは、モデル評価に影響する可能性があります
  - 評価指標への影響: これらの椎体では常にTrue Negativeになるため、特異度（Specificity）は高くなりますが、感度（Recall）の計
    算が部分的にしかできません
  - モデル学習への影響: TRAINデータでは全椎体に骨折があるため、学習自体は問題ありませんが、TESTでの汎化性能評価が偏る可能性が
    あります


### 2025/10/15 
- データローダー修正

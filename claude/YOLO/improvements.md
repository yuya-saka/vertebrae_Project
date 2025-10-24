## データ前処理最適化

### YOLOのbbox作り方（マルチインスタンス対応）

#### データセットの特性（調査結果 2025/10/19）
- 骨折マスクは値0～6で構成（0=背景, 1～6=各骨折インスタンス）
- 各非ゼロ値は異なる骨折領域を示す（専門医による個別アノテーション）
- **骨折分布:**
  - 単一骨折: 96椎体 (70.6%)
  - 複数骨折: 40椎体 (29.4%)
    - 2個: 24椎体
    - 3個: 10椎体
    - 4個: 5椎体
    - 5個: 1椎体
- 1椎体あたり最大5つの独立した骨折が存在

#### BBox作成アルゴリズム

**入力:** マスク画像（NIfTI形式、値0～6）
**出力:** YOLO形式テキストファイル（1画像に複数行可）

**処理フロー:**
```python
for mask_value in range(1, 7):  # 値1～6をループ
    binary_mask = (mask_data == mask_value)

    if not binary_mask.any():
        continue  # この値が存在しない場合スキップ

    # BBox座標抽出
    y_coords, x_coords = np.where(binary_mask)
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()

    # YOLO形式に正規化 [0, 1]
    x_center = (x_min + x_max) / 2 / width
    y_center = (y_min + y_max) / 2 / height
    bbox_width = (x_max - x_min) / width
    bbox_height = (y_max - y_min) / height

    # 品質チェック（後述）
    if is_valid_bbox(bbox_width, bbox_height, area):
        # YOLO形式で保存: <class> <x_center> <y_center> <width> <height>
        write_line(f"0 {x_center} {y_center} {bbox_width} {bbox_height}")
```

**YOLO形式例（1スライスに2つの骨折）:**
```
# case1003_vertebra31_slice119.txt
0 0.573 0.384 0.089 0.052  # 骨折インスタンス1
0 0.536 0.531 0.065 0.048  # 骨折インスタンス2
```

#### BBox品質管理

**フィルタリング条件:**
1. **最小面積閾値:** BBox面積 < 50px² は除外（ノイズ除去）
2. **妥当性チェック:**
   - 幅・高さが0でない
   - 正規化座標が [0, 1] の範囲内
   - アスペクト比が極端でない（例: 1:20以上は要確認）
3. **画像端処理:** BBoxが画像端で切れている場合は保持（骨折の一部）

**可視化検証:**
- ランダムサンプリング（各ケース10%）で目視確認
- 複数BBoxが正しく分離されているか確認
- マスク値と BBox の対応を色分け表示

#### 実装上の注意点

1. **マスク値の連続性:** 値1, 3のみ存在（値2なし）のケースあり → 全値をループ
2. **スライス間の一貫性:** 同じ骨折インスタンスが連続スライスで同じ値とは限らない
   → LSTM段階で追跡・統合が必要
3. **クラスID:** 初期実装は全て class=0（骨折/非骨折の二値分類）
   → 将来的に椎体別分類（14クラス）に拡張可能

#### データ統計レポート

変換後に以下の統計を出力:
- 総BBox数 / 総スライス数
- スライスあたりBBox数の分布（1個, 2個, ...）
- BBoxサイズの分布（min, max, median）
- 極小/極大BBoxのリスト（要目視確認）

### 2025/10/20: マルチウィンドウHU値による3チャンネル入力の採用（min/max形式に更新）

#### 背景と動機
CTスキャン画像は広範囲のHU値（-1000～3000）を持ち、組織の種類によって異なる値を示します：
- 骨組織: 高HU値（+300～+3000）
- 軟部組織: 中HU値（-100～+300）
- 空気: 低HU値（-1000～-100）

単一のウィンドウ設定では、特定の組織のコントラストを強調すると他の組織の情報が失われる問題がありました。

#### 採用する方針
**3つの異なるHU値ウィンドウで処理した画像を3チャンネル（RGB）として入力**

| チャンネル | ウィンドウ名 | HU範囲 | 目的 | 強調される組織 |
|-----------|------------|--------|------|---------------|
| R (赤) | Bone Window | +400～+1800 | 骨構造の可視化 | 椎体の骨梁構造、皮質骨 |
| G (緑) | Soft Tissue Window | -100～+300 | 軟部組織の可視化 | 筋肉、靭帯、神経 |
| B (青) | Wide Window | -200～+500 | 全体のバランス | 骨と軟部組織の境界 |

#### HU値設定方式の変更（2025/10/20更新）

**変更内容**: center/width形式 → **min/max形式**

**理由**:
- より直感的（可視化したいHU範囲を直接指定）
- 計算不要（center - width/2 の計算が不要）
- カスタマイズが容易

**新しい設定形式**:
```yaml
hu_windows:
  bone:
    min: 400    # 最小HU値
    max: 1800   # 最大HU値
  soft_tissue:
    min: -100
    max: 300
  wide:
    min: -200
    max: 500
```

**旧形式（参考）**:
```yaml
# 旧: center/width形式（現在は使用しない）
hu_windows:
  bone:
    center: 1100  # min = 1100 - 1400/2 = 400
    width: 1400   # max = 1100 + 1400/2 = 1800
```

#### 実装方法

```python
def apply_hu_window(image: np.ndarray, hu_min: int, hu_max: int) -> np.ndarray:
    """
    HU値にウィンドウ処理を適用して[0, 1]に正規化

    Args:
        image: CT画像（HU値、-1000～3000）
        hu_min: ウィンドウの最小HU値
        hu_max: ウィンドウの最大HU値

    Returns:
        正規化された画像 [0, 1]
    """
    # ウィンドウ範囲外をクリップ
    windowed = np.clip(image, hu_min, hu_max)

    # [0, 1]に正規化
    normalized = (windowed - hu_min) / (hu_max - hu_min)

    return normalized

def create_3channel_input(ct_image: np.ndarray) -> np.ndarray:
    """
    単一チャンネルCT画像から3チャンネル入力を作成

    Args:
        ct_image: CTスライス画像（HU値、[H, W]）

    Returns:
        3チャンネル画像 [3, H, W]、各チャンネルは[0, 1]に正規化済み
    """
    # Bone Window (min=400, max=1800)
    bone_channel = apply_hu_window(ct_image, hu_min=400, hu_max=1800)

    # Soft Tissue Window (min=-100, max=300)
    soft_channel = apply_hu_window(ct_image, hu_min=-100, hu_max=300)

    # Wide Window (min=-200, max=500)
    wide_channel = apply_hu_window(ct_image, hu_min=-200, hu_max=500)

    # 3チャンネルに統合
    rgb_image = np.stack([bone_channel, soft_channel, wide_channel], axis=0)  # [3, H, W]

    return rgb_image
```

#### メリット

1. **多様な組織情報の保持**
   - 骨組織と軟部組織の情報を同時に利用可能
   - 骨折部位（骨）と周囲の炎症・浮腫（軟部組織）を統合的に評価

2. **事前学習済みモデルの活用**
   - ImageNetで学習されたバックボーン（EfficientNet, ResNet）はRGB 3チャンネル入力を想定
   - チャンネル数を合わせることで事前学習の恩恵を最大化

3. **医療画像解析の実績**
   - 多くの医療AI研究で採用されている標準的手法
   - 単一ウィンドウより高い検出精度が報告されている

#### 実装への影響

**変更が必要な箇所:**
- [convert_to_yolo.py](../../vertebrae_YOLO/data_preparing/convert_to_yolo.py): `normalize_and_pad_image()`関数
  - 現在: 単一チャンネル [H, W] → [1, H, W]
  - 変更後: 3チャンネル [H, W] → [3, H, W]（3つのHUウィンドウ適用）

**変更不要な箇所:**
- YOLOv8モデル: デフォルトで3チャンネル入力に対応
- DataLoader: チャンネル次元は自動的に処理される
- LSTM部分: YOLOの特徴抽出後は影響なし

#### ウィンドウパラメータの調整可能性

初期実装後、以下をアブレーション実験で最適化：
- ウィンドウ幅（Window Width）
- ウィンドウレベル（Window Level/Center）
- チャンネルの組み合わせ（Bone + Soft + Wide以外の組み合わせ）

---

### 2025/10/20: Phase 1-2 実装完了（データ準備・学習環境整備）

#### 実装完了した機能

**1. データセット・モデル実装:**
- [yolo_dataset.py](../../vertebrae_YOLO/src/dataset/yolo_dataset.py): 3チャンネルHUウィンドウ処理対応のPyTorch Dataset
- [yolo_baseline.py](../../vertebrae_YOLO/src/models/yolo_baseline.py): Ultralytics YOLOv8ラッパーモデル

**2. カスタムトレーナー実装:**
- [trainer.py](../../vertebrae_YOLO/src/utils/trainer.py):
  - `CustomYOLOv8Dataset`: 骨折なしサンプルのデータ拡張を確率的に無効化
  - `CustomDetectionTrainer`: Ultralytics DetectionTrainerを継承し、カスタムデータセットを使用
  - NIFTI→PNG変換の自動実行機能

**3. 学習スクリプト・設定:**
- [train.py](../../vertebrae_YOLO/run/scripts/train/train.py): Hydra設定管理、5-fold CV対応
- [config.yaml](../../vertebrae_YOLO/run/conf/config.yaml): メイン設定
- [hyp_custom.yaml](../../vertebrae_YOLO/run/conf/hyp_custom.yaml): データ拡張・損失関数パラメータ
- [split/fold_*.yaml](../../vertebrae_YOLO/run/conf/split/): 5-fold分割設定（30症例を5分割）

#### データ拡張の工夫（不均衡対策）

**問題:** 骨折なしサンプルが多数派で、データ拡張により過剰に増幅される懸念

**解決策:** CustomYOLOv8Datasetでラベルの有無を判定し、データ拡張を動的に制御
```python
def __getitem__(self, index):
    labels = self.get_image_and_label(index)
    has_labels = 'cls' in labels and len(labels['cls']) > 0

    if not has_labels:
        # 骨折なしサンプル: 確率的augmentationを一時的に無効化
        for t in self.transforms.transforms:
            if hasattr(t, 'p'):
                t.p = 0.0  # RandomHSV, RandomFlipなどを無効化
        labels = self.transforms(labels)
        # 確率を元に戻す
    else:
        # 骨折ありサンプル: 通常通りaugmentationを適用
        labels = self.transforms(labels)
```

#### 実装状況まとめ

- ✅ データ変換: 90,638ファイル（画像+ラベル）生成完了
- ✅ Dataset・モデル: 3チャンネルHU処理、YOLOv8ラッパー実装完了
- ✅ 学習環境: カスタムトレーナー、Hydra設定、W&Bログ対応完了
- ⬜ 学習実行: 実装完了、学習実行待ち

#### 次のアクション

1. Fold 0で学習実行（`uv run python train.py`）
2. W&B/TensorBoardで学習曲線とmAP評価
3. 5-fold交差検証の実行
4. ベースライン性能の確立

---

### 2025/10/16
- 切り出した画像データは大きくても256×256の範囲内
- 画像データはリサイズせず、256×256に合わせる、小さい画像は、ゼロパディングで穴埋めで合わせる、これをYOLOの学習パラメータとする

### 2025/10/19
- **骨折マスクのマルチインスタンス特性を発見**
  - マスク値1～6が異なる骨折インスタンスを示す
  - 29.4%の椎体に複数の独立した骨折が存在
  - 各マスク値ごとに独立BBoxを生成する実装方針を決定
  - 連結成分解析は不要（既にインスタンス分離済み）
  - 調査対象: 全trainデータセットの骨折椎体136個

## 失敗した実装とその原因分析、修正点

### 2025/10/20: BBox座標ずれとアフィン行列未適用の問題

#### 問題1: BBox座標のずれ（座標系の不一致）

**症状:**
- 生成されたBBoxが実際の骨折領域からずれる
- 可視化時に明らかな位置のミスマッチが発生

**根本原因:**
処理フローにおける座標系の不一致。以前の実装では以下の順序で処理していた:
```python
# [誤った処理フロー]
1. オリジナルサイズのマスクからBBox座標を計算
2. 画像のみを256x256にリサイズ・パディング
3. 異なるサイズの座標系を無理やり組み合わせる
→ 結果: 座標系の不一致によりBBoxがずれる
```

**修正内容:**
処理順序を根本的に変更し、座標系を統一:
```python
# [正しい処理フロー]
1. 画像とマスクの両方を先に256x256にリサイズ・パディング
   - 画像: normalize_and_pad_image() 使用
   - マスク: normalize_and_pad_mask() 使用（最近傍補間）
2. 変形後のマスクからBBox座標を計算
→ 結果: 画像とBBoxが同じ座標系で一貫性を保つ
```

**実装の詳細:**
- [convert_to_yolo.py:232-274](vertebrae_YOLO/data_preparing/convert_to_yolo.py#L232-L274): マスク専用の `normalize_and_pad_mask()` 関数を新規追加
  - `order=0`（最近傍補間）によりマスク値0～6の整数性を保持
  - 画像と全く同じリサイズ・パディングロジックを適用
- [convert_to_yolo.py:311-316](vertebrae_YOLO/data_preparing/convert_to_yolo.py#L311-L316): `convert_case()` 関数内の処理順序変更
  - 画像とマスクを先に変形してから、BBox抽出を実行

#### 問題2: 画像の傾き（アフィン行列の未適用）

**症状:**
- CTスキャン時の体の傾きや撮影角度により、画像データ自体が傾いて読み込まれる
- 解剖学的に不正確な向きで画像が表示される

**根本原因:**
NIfTI形式ファイルにはアフィン行列（位置・回転・傾き情報）が含まれているが、以前の実装では:
```python
# [誤った実装]
data = np.asarray(nii.dataobj)  # アフィン行列を無視
data = np.fliplr(data)           # 手動で左右反転
```
この方法では、アフィン行列に含まれる幾何学的変換情報を全く考慮していなかった。

**修正内容:**
nibabelの`get_fdata()`を使用してアフィン行列を自動適用:
```python
# [正しい実装]
data = nii.get_fdata(dtype=np.float32)  # アフィン行列を自動適用
# 手動の np.fliplr() を削除
```

**実装の詳細:**
- [convert_to_yolo.py:89-101](vertebrae_YOLO/data_preparing/convert_to_yolo.py#L89-L101): `load_nifti_slice()` 関数の修正
  - `np.asarray(nii.dataobj)` → `nii.get_fdata(dtype=np.float32)` に変更
  - `get_fdata()`はアフィン行列に基づきデータを自動で再配向（傾き補正）
  - 手動の`np.fliplr()`を削除（不要になった）
- [convert_to_yolo.py:305-306](vertebrae_YOLO/data_preparing/convert_to_yolo.py#L305-L306): マスク読み込み時の整数性担保
  - `get_fdata()`は浮動小数点を返すため、`np.round().astype(np.int32)`で整数に戻す
  - 補間による微小な誤差（例: 2.0000001 → 2）を補正

#### 副次的な修正

**HU値（CT値）の保持:**
- [convert_to_yolo.py:188-229](vertebrae_YOLO/data_preparing/convert_to_yolo.py#L188-L229): `normalize_and_pad_image()` のリサイズ処理改善
  - `PIL.Image`（uint8変換が必要）から`scipy.ndimage.zoom`（float型のまま処理）に変更
  - CT値の範囲（-1000～3000 HU）を正確に保持
  - パディング値を`0`から`image.min()`に変更（空気のHU値で埋める）

**影響:**
- BBoxと画像の完全な位置一致を実現
- 解剖学的に正しい向きでの画像処理
- YOLO学習時の座標精度向上

# HU値ウィンドウ設定ガイド

YOLOv8椎体骨折検出プロジェクトにおけるHU値ウィンドウの設定方法

---

## 📋 目次

1. [HU値とは](#hu値とは)
2. [ウィンドウ設定方法](#ウィンドウ設定方法)
3. [推奨設定値](#推奨設定値)
4. [カスタマイズ方法](#カスタマイズ方法)
5. [設定例](#設定例)

---

## HU値とは

**HU値（Hounsfield Unit）**は、CT画像における組織の密度を表す単位です。

### 主な組織のHU値範囲

| 組織 | HU値範囲 |
|------|---------|
| 空気 | -1000 |
| 脂肪 | -100 ~ -50 |
| 水 | 0 |
| 軟部組織（筋肉・臓器） | +40 ~ +80 |
| 血液 | +30 ~ +45 |
| 骨皮質 | +400 ~ +1000 |
| 骨梁（海綿骨） | +300 ~ +400 |
| 高密度骨 | +1000 ~ +3000 |

---

## ウィンドウ設定方法

### 新方式: min/max形式（現在の実装）

**最小値と最大値を直接指定**する方式に変更しました。

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

**メリット:**
- 直感的で分かりやすい
- 可視化したいHU範囲を直接指定
- center/width計算が不要

### 旧方式: center/width形式（参考）

従来の医療画像ビューアーで使われる方式：

```yaml
# 参考: 旧方式（現在は使用しない）
hu_windows:
  bone:
    center: 1100  # ウィンドウ中心
    width: 1400   # ウィンドウ幅
```

**変換式:**
- `min = center - width / 2`
- `max = center + width / 2`

例: center=1100, width=1400 の場合
- min = 1100 - 700 = **400**
- max = 1100 + 700 = **1800**

---

## 推奨設定値

### デフォルト設定（骨折検出用）

```yaml
hu_windows:
  bone:
    min: 400    # 骨梁～高密度骨を強調
    max: 1800
  soft_tissue:
    min: -100   # 脂肪～筋肉・臓器を可視化
    max: 300
  wide:
    min: -200   # 全体のバランス（骨と軟部組織の境界）
    max: 500
```

### 各チャンネルの役割

#### 1. Bone Window（骨ウィンドウ）
- **目的**: 骨構造の可視化
- **min: 400, max: 1800**
- **強調される組織**: 骨皮質、骨梁、骨折線
- **用途**: 骨折の有無、骨折線の確認

#### 2. Soft Tissue Window（軟部組織ウィンドウ）
- **目的**: 軟部組織の可視化
- **min: -100, max: 300**
- **強調される組織**: 筋肉、靭帯、血管、浮腫
- **用途**: 骨折周囲の軟部組織損傷、炎症の確認

#### 3. Wide Window（広域ウィンドウ）
- **目的**: 全体のバランス
- **min: -200, max: 500**
- **強調される組織**: 骨と軟部組織の境界
- **用途**: 解剖学的位置関係の把握

---

## カスタマイズ方法

### 方法1: 設定ファイルを編集

[run/conf/data/yolo_data.yaml](../run/conf/data/yolo_data.yaml) を編集：

```yaml
data:
  hu_windows:
    bone:
      min: 500    # より高密度の骨のみ強調
      max: 2000
    soft_tissue:
      min: -50    # 脂肪を除外
      max: 250
    wide:
      min: -100
      max: 600
```

### 方法2: コマンドラインでオーバーライド

```bash
# Bone Windowのみ変更
python train.py \
    data.hu_windows.bone.min=500 \
    data.hu_windows.bone.max=2000

# 複数チャンネルを変更
python train.py \
    data.hu_windows.bone.min=500 \
    data.hu_windows.bone.max=2000 \
    data.hu_windows.soft_tissue.min=-50 \
    data.hu_windows.soft_tissue.max=250
```

### 方法3: 新しい設定ファイルを作成

カスタム設定ファイルを作成：

```yaml
# run/conf/data/yolo_data_custom.yaml
data:
  # ... 他の設定 ...

  hu_windows:
    bone:
      min: 600    # カスタム設定
      max: 2500
    soft_tissue:
      min: 0
      max: 200
    wide:
      min: -100
      max: 800
```

使用時：

```bash
python train.py data=yolo_data_custom
```

---

## 設定例

### 例1: 高密度骨のみ強調（骨粗鬆症検出用）

```yaml
hu_windows:
  bone:
    min: 600    # より高密度の骨のみ
    max: 2500
  soft_tissue:
    min: -100
    max: 300
  wide:
    min: -200
    max: 500
```

### 例2: 軟部組織を強調（靭帯損傷検出用）

```yaml
hu_windows:
  bone:
    min: 400
    max: 1800
  soft_tissue:
    min: -50    # 脂肪を除外、筋肉・靭帯を強調
    max: 200
  wide:
    min: -100
    max: 400
```

### 例3: 広範囲可視化（全体確認用）

```yaml
hu_windows:
  bone:
    min: 200    # 低密度骨も含む
    max: 2000
  soft_tissue:
    min: -200   # 脂肪も含む
    max: 400
  wide:
    min: -300   # 空気近くも可視化
    max: 800
```

---

## HU値の可視化確認

設定したHU値ウィンドウが適切かを確認するには、以下のスクリプトを使用：

```python
# 簡易確認スクリプト
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def visualize_hu_windows(nii_path, hu_windows):
    """HU値ウィンドウの効果を可視化"""
    # CT画像読み込み
    nii = nib.load(nii_path)
    image = nii.get_fdata()[:, :, 0]  # 1スライス

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # オリジナル
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original (HU値そのまま)')

    # Bone Window
    bone = np.clip(image, hu_windows['bone']['min'], hu_windows['bone']['max'])
    bone = (bone - hu_windows['bone']['min']) / (hu_windows['bone']['max'] - hu_windows['bone']['min'])
    axes[1].imshow(bone, cmap='gray')
    axes[1].set_title(f"Bone Window ({hu_windows['bone']['min']}～{hu_windows['bone']['max']})")

    # Soft Tissue Window
    soft = np.clip(image, hu_windows['soft_tissue']['min'], hu_windows['soft_tissue']['max'])
    soft = (soft - hu_windows['soft_tissue']['min']) / (hu_windows['soft_tissue']['max'] - hu_windows['soft_tissue']['min'])
    axes[2].imshow(soft, cmap='gray')
    axes[2].set_title(f"Soft Tissue Window ({hu_windows['soft_tissue']['min']}～{hu_windows['soft_tissue']['max']})")

    # Wide Window
    wide = np.clip(image, hu_windows['wide']['min'], hu_windows['wide']['max'])
    wide = (wide - hu_windows['wide']['min']) / (hu_windows['wide']['max'] - hu_windows['wide']['min'])
    axes[3].imshow(wide, cmap='gray')
    axes[3].set_title(f"Wide Window ({hu_windows['wide']['min']}～{hu_windows['wide']['max']})")

    plt.tight_layout()
    plt.savefig('hu_window_visualization.png')
    print("Saved: hu_window_visualization.png")

# 使用例
hu_windows = {
    'bone': {'min': 400, 'max': 1800},
    'soft_tissue': {'min': -100, 'max': 300},
    'wide': {'min': -200, 'max': 500},
}

visualize_hu_windows('data/yolo_format/images/axial/train/inp1003_27_slice_050.nii', hu_windows)
```

---

## トラブルシューティング

### Q1: 画像が真っ白/真っ黒になる

**原因**: HU範囲が狭すぎる、または画像の実際のHU値範囲外

**解決策**:
```yaml
# 範囲を広げる
hu_windows:
  bone:
    min: 200    # 最小値を下げる
    max: 2500   # 最大値を上げる
```

### Q2: コントラストが低い

**原因**: HU範囲が広すぎる

**解決策**:
```yaml
# 範囲を狭める（目的の組織に特化）
hu_windows:
  bone:
    min: 600    # 最小値を上げる
    max: 1500   # 最大値を下げる
```

### Q3: どの値が適切か分からない

**推奨手順**:
1. まずデフォルト値で学習
2. 可視化スクリプトで確認
3. 必要に応じて調整
4. アブレーション実験で比較

---

## 参考文献

- Hounsfield Scale: https://radiopaedia.org/articles/hounsfield-unit
- CT Window Settings: 医療画像処理の標準的手法

---

**最終更新**: 2025/10/20

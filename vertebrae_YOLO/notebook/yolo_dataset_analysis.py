"""
YOLO Dataset Quality Analysis

このスクリプトは、変換されたYOLO形式データセットの品質を検証するための探索的分析を行います。

検証項目:
1. データセット構造の確認
2. 画像とラベルファイルの対応チェック
3. BBoxアノテーションの統計分析
4. 画像の可視化とBBox描画
5. 異常値・エッジケースの検出
6. 画像品質のチェック
"""

# %%
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# 日本語フォント設定（必要に応じて）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 設定
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# %% [markdown]
# ## 1. データセット構造の確認

# %%
# パス設定
BASE_DIR = Path('../../data/yolo_format')
VIEW = 'axial'
SPLIT = 'train'  # 'train', 'val', 'test'

images_dir = BASE_DIR / 'images' / VIEW / SPLIT
labels_dir = BASE_DIR / 'labels' / VIEW / SPLIT

print(f"Images directory: {images_dir}")
print(f"Labels directory: {labels_dir}")
print(f"Images exist: {images_dir.exists()}")
print(f"Labels exist: {labels_dir.exists()}")

# %%
# ファイル数の確認
image_files = sorted(list(images_dir.glob('*.nii')))
label_files = sorted(list(labels_dir.glob('*.txt')))

print(f"Total images: {len(image_files)}")
print(f"Total labels: {len(label_files)}")
print(f"\nSample files:")
for i in range(min(3, len(image_files))):
    print(f"  Image: {image_files[i].name}")
    print(f"  Label: {label_files[i].name}")

# %% [markdown]
# ## 2. 画像とラベルファイルの対応チェック

# %%
# ファイル名の対応確認
image_stems = {f.stem for f in image_files}
label_stems = {f.stem for f in label_files}

# 対応しているか確認
missing_labels = image_stems - label_stems
missing_images = label_stems - image_stems

print(f"Images without labels: {len(missing_labels)}")
if missing_labels:
    print("  Examples:", list(missing_labels)[:5])

print(f"Labels without images: {len(missing_images)}")
if missing_images:
    print("  Examples:", list(missing_images)[:5])

# マッチするペア数
matched_pairs = image_stems & label_stems
print(f"\nMatched pairs: {len(matched_pairs)}")

# %% [markdown]
# ## 3. BBoxアノテーションの統計分析

# %%
def parse_yolo_label(label_path):
    """YOLOラベルファイルを解析"""
    bboxes = []
    if not label_path.exists():
        return bboxes

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, x_center, y_center, width, height = map(float, parts)
                bboxes.append({
                    'class': int(cls),
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'area': width * height  # 正規化された面積
                })
    return bboxes

# 全ラベルファイルを解析
all_bboxes = []
bboxes_per_image = []
empty_label_count = 0

print("Parsing label files...")
for label_file in tqdm(label_files):
    bboxes = parse_yolo_label(label_file)
    all_bboxes.extend(bboxes)
    bboxes_per_image.append(len(bboxes))
    if len(bboxes) == 0:
        empty_label_count += 1

print(f"\nTotal BBoxes: {len(all_bboxes)}")
print(f"Images with fracture: {len(label_files) - empty_label_count}")
print(f"Images without fracture: {empty_label_count}")
print(f"Fracture ratio: {(len(label_files) - empty_label_count) / len(label_files) * 100:.1f}%")

# %%
# BBoxes per imageの分布
bbox_count_dist = Counter(bboxes_per_image)

print("BBoxes per Image Distribution:")
for count in sorted(bbox_count_dist.keys()):
    print(f"  {count} bbox(es): {bbox_count_dist[count]} images")

# 可視化
plt.figure(figsize=(10, 5))
counts = sorted(bbox_count_dist.keys())
values = [bbox_count_dist[c] for c in counts]
plt.bar(counts, values, color='steelblue', edgecolor='black')
plt.xlabel('Number of BBoxes per Image')
plt.ylabel('Number of Images')
plt.title('BBox Count Distribution')
plt.xticks(counts)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../notebook/exploratory_images/bbox_count_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 4. BBoxサイズ・形状の分析

# %%
# BBoxサイズの統計（正規化値）
if all_bboxes:
    df_bboxes = pd.DataFrame(all_bboxes)

    print("BBox Statistics (normalized):")
    print(df_bboxes[['width', 'height', 'area']].describe())

    # アスペクト比を計算
    df_bboxes['aspect_ratio'] = df_bboxes['width'] / df_bboxes['height']

    print("\nAspect Ratio Statistics:")
    print(df_bboxes['aspect_ratio'].describe())
else:
    print("No bboxes found in the dataset.")
    df_bboxes = None

# %%
# BBoxサイズの可視化
if all_bboxes:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Width分布
    axes[0, 0].hist(df_bboxes['width'], bins=50, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Width (normalized)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('BBox Width Distribution')
    axes[0, 0].grid(alpha=0.3)

    # Height分布
    axes[0, 1].hist(df_bboxes['height'], bins=50, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('Height (normalized)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('BBox Height Distribution')
    axes[0, 1].grid(alpha=0.3)

    # Area分布
    axes[1, 0].hist(df_bboxes['area'], bins=50, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Area (normalized)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('BBox Area Distribution')
    axes[1, 0].grid(alpha=0.3)

    # Aspect ratio分布
    axes[1, 1].hist(df_bboxes['aspect_ratio'], bins=50, color='plum', edgecolor='black')
    axes[1, 1].set_xlabel('Aspect Ratio (width/height)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('BBox Aspect Ratio Distribution')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('../notebook/exploratory_images/bbox_size_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()

# %%
# Width vs Height の散布図
if all_bboxes:
    plt.figure(figsize=(8, 8))
    plt.scatter(df_bboxes['width'], df_bboxes['height'], alpha=0.5, s=10)
    plt.xlabel('Width (normalized)')
    plt.ylabel('Height (normalized)')
    plt.title('BBox Width vs Height')
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('../notebook/exploratory_images/bbox_width_vs_height.png', dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 5. 異常値・エッジケースの検出

# %%
# 異常に小さい/大きいBBoxを検出
if all_bboxes:
    # 面積の閾値（正規化値）
    small_area_threshold = 0.001  # 画像の0.1%以下
    large_area_threshold = 0.5    # 画像の50%以上

    small_bboxes = df_bboxes[df_bboxes['area'] < small_area_threshold]
    large_bboxes = df_bboxes[df_bboxes['area'] > large_area_threshold]

    print(f"Extremely small BBoxes (area < {small_area_threshold}): {len(small_bboxes)}")
    if len(small_bboxes) > 0:
        print(small_bboxes[['width', 'height', 'area']].head())

    print(f"\nExtremely large BBoxes (area > {large_area_threshold}): {len(large_bboxes)}")
    if len(large_bboxes) > 0:
        print(large_bboxes[['width', 'height', 'area']].head())

    # 極端なアスペクト比
    extreme_aspect_ratio = df_bboxes[
        (df_bboxes['aspect_ratio'] < 0.2) | (df_bboxes['aspect_ratio'] > 5.0)
    ]
    print(f"\nExtreme aspect ratios (< 0.2 or > 5.0): {len(extreme_aspect_ratio)}")
    if len(extreme_aspect_ratio) > 0:
        print(extreme_aspect_ratio[['width', 'height', 'aspect_ratio']].head())
else:
    small_bboxes = []
    large_bboxes = []

# %%
# BBox座標の範囲チェック（0~1の範囲外の値を検出）
if all_bboxes:
    out_of_range = df_bboxes[
        (df_bboxes['x_center'] < 0) | (df_bboxes['x_center'] > 1) |
        (df_bboxes['y_center'] < 0) | (df_bboxes['y_center'] > 1) |
        (df_bboxes['width'] <= 0) | (df_bboxes['width'] > 1) |
        (df_bboxes['height'] <= 0) | (df_bboxes['height'] > 1)
    ]

    print(f"BBoxes with out-of-range values: {len(out_of_range)}")
    if len(out_of_range) > 0:
        print("WARNING: Found invalid bbox coordinates!")
        print(out_of_range.head())
    else:
        print("✓ All BBox coordinates are within valid range [0, 1]")
else:
    out_of_range = []

# %% [markdown]
# ## 6. 画像とBBoxの可視化

# %%
def load_nifti_image(nii_path):
    """NIfTI画像を読み込み、正しい向きで表示できるように処理

    Returns:
        tuple: (data, should_flip) - 画像データと反転が必要かのフラグ
    """
    nii = nib.load(str(nii_path))
    data = nii.get_fdata()
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]

    # LAS向きの画像を正しく表示するために左右反転が必要か判定
    # Affine行列の第1成分が負の場合（ほとんどの医療画像）
    should_flip = nii.affine[0, 0] < 0

    if should_flip:
        data = np.fliplr(data)

    return data, should_flip

def draw_yolo_bboxes(ax, image, bboxes, title="", flip_x=False):
    """画像にYOLO BBoxを描画"""
    h, w = image.shape
    ax.imshow(image, cmap='gray')

    for bbox in bboxes:
        # YOLO形式 -> ピクセル座標に変換
        x_center = bbox['x_center'] * w
        y_center = bbox['y_center'] * h
        bbox_w = bbox['width'] * w
        bbox_h = bbox['height'] * h

        # 画像が左右反転されている場合、BBoxも反転
        if flip_x:
            x_center = w - x_center

        # BBox左上座標
        x_min = x_center - bbox_w / 2
        y_min = y_center - bbox_h / 2

        # 矩形描画
        rect = patches.Rectangle(
            (x_min, y_min), bbox_w, bbox_h,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

        # 中心点
        ax.plot(x_center, y_center, 'r+', markersize=10, markeredgewidth=2)

    ax.set_title(f"{title}\nBBoxes: {len(bboxes)}")
    ax.axis('off')

# %%
# ランダムサンプルを可視化
np.random.seed(42)
sample_indices = np.random.choice(len(label_files), min(9, len(label_files)), replace=False)

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for i, idx in enumerate(sample_indices):
    label_file = label_files[idx]
    image_file = images_dir / f"{label_file.stem}.nii"

    if image_file.exists():
        image, flipped = load_nifti_image(image_file)
        bboxes = parse_yolo_label(label_file)
        draw_yolo_bboxes(axes[i], image, bboxes, title=label_file.stem, flip_x=flipped)
    else:
        axes[i].text(0.5, 0.5, 'Image not found', ha='center', va='center')
        axes[i].axis('off')

plt.tight_layout()
plt.savefig('../notebook/exploratory_images/sample_images_with_bboxes.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# 最も多くのBBoxを持つ画像を可視化
if bboxes_per_image:
    max_bbox_count = max(bboxes_per_image)
    max_bbox_indices = [i for i, count in enumerate(bboxes_per_image) if count == max_bbox_count]

    print(f"Maximum BBoxes in a single image: {max_bbox_count}")
    print(f"Number of images with {max_bbox_count} bboxes: {len(max_bbox_indices)}")

    if max_bbox_indices:
        # 最初の3つを表示
        fig, axes = plt.subplots(1, min(3, len(max_bbox_indices)), figsize=(15, 5))
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for i, idx in enumerate(max_bbox_indices[:3]):
            label_file = label_files[idx]
            image_file = images_dir / f"{label_file.stem}.nii"

            if image_file.exists():
                image, flipped = load_nifti_image(image_file)
                bboxes = parse_yolo_label(label_file)
                draw_yolo_bboxes(axes[i], image, bboxes, title=f"Max BBoxes: {label_file.stem}", flip_x=flipped)

        plt.tight_layout()
        plt.savefig('../notebook/exploratory_images/max_bbox_images.png', dpi=150, bbox_inches='tight')
        plt.show()

# %%
# 骨折なし画像のサンプル表示
empty_indices = [i for i, count in enumerate(bboxes_per_image) if count == 0]

print(f"Images without fracture: {len(empty_indices)}")

if empty_indices:
    sample_empty = np.random.choice(empty_indices, min(6, len(empty_indices)), replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, idx in enumerate(sample_empty):
        label_file = label_files[idx]
        image_file = images_dir / f"{label_file.stem}.nii"

        if image_file.exists():
            image, flipped = load_nifti_image(image_file)
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f"No Fracture: {label_file.stem}")
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('../notebook/exploratory_images/no_fracture_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 7. 画像品質のチェック

# %%
# 画像サイズの一貫性確認
image_shapes = []
print("Checking image dimensions...")
for image_file in tqdm(image_files[:100]):  # 最初の100枚をチェック
    image, _ = load_nifti_image(image_file)
    image_shapes.append(image.shape)

unique_shapes = set(image_shapes)
print(f"\nUnique image shapes: {unique_shapes}")

if len(unique_shapes) == 1:
    print(f"✓ All images have consistent shape: {list(unique_shapes)[0]}")
else:
    print("WARNING: Multiple image shapes detected!")
    shape_counts = Counter(image_shapes)
    for shape, count in shape_counts.items():
        print(f"  Shape {shape}: {count} images")

# %%
# 画像の強度値の統計
print("Analyzing image intensity values...")
sample_images_data = []

for image_file in tqdm(image_files[:50]):
    image, _ = load_nifti_image(image_file)
    sample_images_data.append({
        'min': image.min(),
        'max': image.max(),
        'mean': image.mean(),
        'std': image.std()
    })

df_intensity = pd.DataFrame(sample_images_data)
print("\nImage Intensity Statistics:")
print(df_intensity.describe())

# %% [markdown]
# ## 8. サマリーレポート

# %%
print("="*70)
print(f"YOLO Dataset Quality Report - {SPLIT.upper()} SET")
print("="*70)

print(f"\n[Dataset Overview]")
print(f"  Total images: {len(image_files)}")
print(f"  Total labels: {len(label_files)}")
print(f"  Matched pairs: {len(matched_pairs)}")
print(f"  Images with fracture: {len(label_files) - empty_label_count}")
print(f"  Images without fracture: {empty_label_count}")

print(f"\n[BBox Statistics]")
print(f"  Total BBoxes: {len(all_bboxes)}")
if bboxes_per_image:
    print(f"  Avg BBoxes per image: {np.mean(bboxes_per_image):.2f} ± {np.std(bboxes_per_image):.2f}")
    print(f"  Max BBoxes in single image: {max(bboxes_per_image)}")

if all_bboxes:
    print(f"\n[BBox Size (normalized)]")
    print(f"  Width: {df_bboxes['width'].mean():.4f} ± {df_bboxes['width'].std():.4f}")
    print(f"  Height: {df_bboxes['height'].mean():.4f} ± {df_bboxes['height'].std():.4f}")
    print(f"  Area: {df_bboxes['area'].mean():.4f} ± {df_bboxes['area'].std():.4f}")
    print(f"  Aspect ratio: {df_bboxes['aspect_ratio'].mean():.2f} ± {df_bboxes['aspect_ratio'].std():.2f}")

print(f"\n[Data Quality]")
print(f"  Unique image shapes: {len(unique_shapes)}")
print(f"  Out-of-range BBox coordinates: {len(out_of_range) if all_bboxes else 0}")
print(f"  Extreme small BBoxes: {len(small_bboxes) if all_bboxes else 0}")
print(f"  Extreme large BBoxes: {len(large_bboxes) if all_bboxes else 0}")

print("\n" + "="*70)
print("Analysis completed!")
print("Visualizations saved to: vertebrae_YOLO/notebook/exploratory_images/")
print("="*70)

# %%

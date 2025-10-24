"""
YOLO BBox品質分析スクリプト（総合版）

データセットの探索的分析と、元のマスクとの比較による精度検証を統合的に行います。

主な検証項目:
1. データセットの全体構造と統計情報
2. BBoxのサイズ、形状、分布の分析
3. **[重要]** 元マスクとのオーバーレイ比較による視覚的なズレの確認
4. **[重要]** IoU（Intersection over Union）計算によるBBox精度の定量的評価
"""

# %%
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
import pandas as pd
import seaborn as sns
from scipy.ndimage import zoom

# プロット設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')

print("Imports completed!")

# %% [markdown]
# ## 1. 設定とパス確認

# %%
# パス設定
BASE_DIR = Path('../../data/yolo_format')
ORIGINAL_MASK_DIR = Path('../../data/slice_train/axial_mask') # 元マスクのパス

VIEW = 'axial'
SPLIT = 'train' # 'train', 'val', 'test'

images_dir = BASE_DIR / 'images' / VIEW / SPLIT
labels_dir = BASE_DIR / 'labels' / VIEW / SPLIT
output_dir = Path('./yolo_quality_analysis_output_final') # 出力先フォルダ
output_dir.mkdir(exist_ok=True)

print(f"📁 Directory Paths:")
print(f"   Images: {images_dir}")
print(f"   Labels: {labels_dir}")
print(f"   Original Masks: {ORIGINAL_MASK_DIR}")
print(f"   Output: {output_dir}")
print(f"\n✅ Path Validation:")
print(f"   Images exist: {images_dir.exists()}")
print(f"   Labels exist: {labels_dir.exists()}")
print(f"   Masks exist: {ORIGINAL_MASK_DIR.exists()}")

# %% [markdown]
# ## 2. ユーティリティ関数

# %%
def load_nifti(path):
    """NIfTI画像を読み込み、get_fdata()で自動再配向"""
    nii = nib.load(str(path))
    data = nii.get_fdata(dtype=np.float32)
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]
    elif data.ndim > 2:
        data = data.squeeze()
    return data

def normalize_and_pad_mask_for_comparison(mask: np.ndarray, target_size: tuple) -> np.ndarray:
    """比較のため、マスクをYOLO画像と同じサイズに変形する（最近傍補間）"""
    h, w = mask.shape[:2]
    bg_value = 0
    if h == target_size[0] and w == target_size[1]:
        return mask
    if h > target_size[0] or w > target_size[1]:
        zoom_h, zoom_w = target_size[0] / h, target_size[1] / w
        resized_mask = zoom(mask, (zoom_h, zoom_w), order=0, mode='constant', cval=bg_value)
        rh, rw = resized_mask.shape
        th, tw = target_size
        final_mask = np.full(target_size, bg_value, dtype=mask.dtype)
        h_crop, w_crop = min(rh, th), min(rw, tw)
        h_offset, w_offset = (th - h_crop) // 2, (tw - w_crop) // 2
        final_mask[h_offset:h_offset+h_crop, w_offset:w_offset+w_crop] = resized_mask[:h_crop, :w_crop]
        return final_mask
    padded = np.full(target_size, bg_value, dtype=mask.dtype)
    pad_h, pad_w = (target_size[0] - h) // 2, (target_size[1] - w) // 2
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = mask
    return padded

def parse_yolo_label(label_path):
    """YOLOラベルファイルを解析"""
    bboxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = [float(p) for p in line.strip().split()]
            if len(parts) == 5:
                bboxes.append({'class': int(parts[0]), 'x_center': parts[1], 'y_center': parts[2], 'width': parts[3], 'height': parts[4]})
    return bboxes

def yolo_to_pixel(bbox, h, w):
    """YOLO正規化座標をピクセル座標に変換"""
    xc, yc = bbox['x_center'] * w, bbox['y_center'] * h
    bw, bh = bbox['width'] * w, bbox['height'] * h
    return int(xc - bw / 2), int(yc - bh / 2), int(xc + bw / 2), int(yc + bh / 2)

def extract_mask_bboxes(mask):
    """マスクから各インスタンスのBBoxを抽出"""
    bboxes = []
    for val in range(1, 7):
        binary_mask = (mask == val)
        if binary_mask.any():
            y_coords, x_coords = np.where(binary_mask)
            bboxes.append({'value': val, 'x1': x_coords.min(), 'y1': y_coords.min(), 'x2': x_coords.max(), 'y2': y_coords.max()})
    return bboxes

def calculate_iou(box1, box2):
    """IoUを計算"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x1, inter_y1 = max(x1_min, x2_min), max(y1_min, y2_min)
    inter_x2, inter_y2 = min(x1_max, x2_max), min(y1_max, y2_max)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def find_original_mask_file(yolo_name, mask_dir):
    """YOLOファイル名から元のマスクファイルを探す"""
    parts = yolo_name.split('_')
    if len(parts) < 4: return None
    case, vertebra, slice_idx = parts[0], parts[1], parts[3]
    mask_path = mask_dir / case / vertebra / f"mask_{slice_idx}.nii"
    return mask_path if mask_path.exists() else None

print("✅ Utility functions defined!")

# %% [markdown]
# ## 3. データセット統計分析

# %%
image_files = sorted(list(images_dir.glob('*.nii')))
label_files = sorted(list(labels_dir.glob('*.txt')))

# ファイル対応チェック
image_stems = {f.stem for f in image_files}
label_stems = {f.stem for f in label_files}
if image_stems != label_stems:
    print("⚠️ Warning: Mismatch between image and label files found!")
    print(f"   Images without labels: {len(image_stems - label_stems)}")
    print(f"   Labels without images: {len(label_stems - image_stems)}")

# BBox統計
all_bboxes = []
bboxes_per_image = []
for label_file in tqdm(label_files, desc="Parsing labels"):
    bboxes = parse_yolo_label(label_file)
    all_bboxes.extend(bboxes)
    bboxes_per_image.append(len(bboxes))

print(f"\n📊 Dataset Overview:")
print(f"   Total images: {len(image_files)}")
print(f"   Images with fracture: {sum(1 for c in bboxes_per_image if c > 0)}")
print(f"   Images without fracture: {sum(1 for c in bboxes_per_image if c == 0)}")
print(f"   Total BBoxes: {len(all_bboxes)}")

# BBox数の分布をプロット
plt.figure(figsize=(10, 5))
sns.countplot(x=bboxes_per_image, palette="viridis")
plt.title('BBox Count Distribution per Image', fontsize=14, fontweight='bold')
plt.xlabel('Number of BBoxes'); plt.ylabel('Number of Images')
plt.savefig(output_dir / 'bbox_count_distribution.png', dpi=150)
plt.show()

# %% [markdown]
# ## 4. BBoxサイズと形状の分析

# %%
if all_bboxes:
    df_bboxes = pd.DataFrame(all_bboxes)
    df_bboxes['area'] = df_bboxes['width'] * df_bboxes['height']
    df_bboxes['aspect_ratio'] = df_bboxes['width'] / df_bboxes['height']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df_bboxes['width'], bins=50, ax=axes[0, 0], color='skyblue').set_title('Width Distribution')
    sns.histplot(df_bboxes['height'], bins=50, ax=axes[0, 1], color='lightcoral').set_title('Height Distribution')
    sns.histplot(df_bboxes['area'], bins=50, ax=axes[1, 0], color='lightgreen').set_title('Area Distribution')
    sns.histplot(df_bboxes['aspect_ratio'], bins=50, ax=axes[1, 1], color='plum').set_title('Aspect Ratio Distribution')
    plt.tight_layout()
    plt.savefig(output_dir / 'bbox_size_distributions.png', dpi=150)
    plt.show()

# %% [markdown]
# ## 5. マスクとBBoxの比較可視化（オーバーレイ確認）

# %%
print("\n🔍 Generating mask-bbox comparison (overlay check)...")
fig, axes = plt.subplots(3, 3, figsize=(22, 22))
sample_count = 0
fracture_labels = [lf for lf, count in zip(label_files, bboxes_per_image) if count > 0]

for label_file in fracture_labels:
    if sample_count >= 3: break
    mask_path = find_original_mask_file(label_file.stem, ORIGINAL_MASK_DIR)
    image_path = images_dir / f"{label_file.stem}.nii"
    if not (mask_path and image_path.exists()): continue
    
    image = load_nifti(image_path)
    mask_original = load_nifti(mask_path)
    yolo_bboxes = parse_yolo_label(label_file)
    
    # 座標系を合わせるためにマスクを変形
    mask_processed = normalize_and_pad_mask_for_comparison(mask_original, target_size=image.shape)
    mask_bboxes = extract_mask_bboxes(mask_processed)
    
    if not yolo_bboxes or not mask_bboxes: continue
    
    # 可視化
    h, w = image.shape
    row = sample_count
    axes[row, 0].imshow(image, cmap='gray'); axes[row, 0].set_title('YOLO Image'); axes[row, 0].axis('off')
    axes[row, 1].imshow(image, cmap='gray', alpha=0.5); axes[row, 1].imshow(mask_processed, cmap='Reds', alpha=0.5)
    axes[row, 1].set_title(f'Processed Mask ({len(mask_bboxes)})'); axes[row, 1].axis('off')
    axes[row, 2].imshow(image, cmap='gray'); axes[row, 2].imshow(mask_processed, cmap='Blues', alpha=0.2)
    for bbox in yolo_bboxes: axes[row, 2].add_patch(patches.Rectangle(yolo_to_pixel(bbox, h, w)[:2], bbox['width']*w, bbox['height']*h, lw=3, ec='red', fc='none'))
    for mb in mask_bboxes: axes[row, 2].add_patch(patches.Rectangle((mb['x1'], mb['y1']), mb['x2']-mb['x1'], mb['y2']-mb['y1'], lw=3, ec='blue', fc='none', ls='--'))
    iou = calculate_iou(yolo_to_pixel(yolo_bboxes[0], h, w), tuple(mask_bboxes[0].values())[1:])
    axes[row, 2].set_title(f'Overlay (IoU: {iou:.3f})'); axes[row, 2].axis('off')
    
    sample_count += 1

plt.suptitle('BBox vs Processed Mask Comparison', fontsize=16, y=0.92)
plt.tight_layout(rect=[0, 0.03, 1, 0.9])
plt.savefig(output_dir / 'bbox_mask_overlay.png', dpi=150)
print(f"✅ Saved overlay check: {output_dir / 'bbox_mask_overlay.png'}")
plt.show()

# %% [markdown]
# ## 6. IoU統計の詳細計算

# %%
print("\n📊 Computing detailed IoU statistics...")
iou_scores = []
for label_file in tqdm(fracture_labels, desc="Computing IoU"):
    mask_path = find_original_mask_file(label_file.stem, ORIGINAL_MASK_DIR)
    image_path = images_dir / f"{label_file.stem}.nii"
    if not (mask_path and image_path.exists()): continue
    
    image = load_nifti(image_path)
    mask_original = load_nifti(mask_path)
    yolo_bboxes = parse_yolo_label(label_file)
    
    mask_processed = normalize_and_pad_mask_for_comparison(mask_original, target_size=image.shape)
    mask_bboxes = extract_mask_bboxes(mask_processed)
    
    if len(yolo_bboxes) != len(mask_bboxes): continue

    for yb, mb in zip(yolo_bboxes, mask_bboxes):
        iou = calculate_iou(yolo_to_pixel(yb, image.shape[0], image.shape[1]), (mb['x1'], mb['y1'], mb['x2'], mb['y2']))
        iou_scores.append(iou)

if iou_scores:
    print(f"\n📈 [IoU Statistics]")
    print(f"   BBoxes analyzed: {len(iou_scores)}")
    print(f"   Mean IoU: {np.mean(iou_scores):.4f} | Median IoU: {np.median(iou_scores):.4f}")
    print(f"   IoU >= 0.95: {sum(i >= 0.95 for i in iou_scores) / len(iou_scores) * 100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(iou_scores, bins=50, ax=axes[0], color='steelblue').set_title('IoU Distribution')
    sns.ecdfplot(iou_scores, ax=axes[1], color='steelblue').set_title('Cumulative IoU Distribution')
    plt.tight_layout()
    plt.savefig(output_dir / 'iou_distribution.png', dpi=150)
    plt.show()

# %% [markdown]
# ## 7. 最終レポート

# %%
print("\n" + "="*80)
print(f"YOLO DATASET QUALITY REPORT - {SPLIT.upper()} SET")
print("="*80)
print(f"\n[Dataset Overview]")
print(f"   Total images: {len(image_files)}, Images with fracture: {len(fracture_labels)}")
if all_bboxes:
    print(f"\n[BBox Statistics]")
    print(f"   Total BBoxes: {len(all_bboxes)}")
    print(f"   Avg BBoxes per image: {np.mean(bboxes_per_image):.2f}")
    print(f"   Mean Area (normalized): {df_bboxes['area'].mean():.4f}")
if iou_scores:
    mean_iou = np.mean(iou_scores)
    print(f"\n[IoU Quality Assessment]")
    print(f"   Mean IoU: {mean_iou:.4f}")
    if mean_iou >= 0.9: print("   ✅ Excellent BBox quality!")
    elif mean_iou >= 0.8: print("   ✅ Good BBox quality.")
    else: print("   ⚠️ Acceptable quality, but check low IoU cases.")
print("\n" + "="*80)

# %%

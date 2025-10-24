"""
YOLO BBox品質分析スクリプト（修正後版）

修正されたconvert_to_yolo.pyで生成されたYOLOデータセットの品質を検証します。

主な検証項目:
1. BBoxが元のマスク領域を正しくカバーしているか（IoU計算）
2. 画像とマスクの向きが一致しているか（オーバーレイ確認）
3. BBoxのサイズと位置の妥当性
4. 複数インスタンスの分離状況
5. 統計情報（IoU分布、BBox数、サイズ分布）
"""

# %%
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from scipy.ndimage import zoom  # [FIXED] Added for mask resizing

# プロット設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['figure.figsize'] = (16, 10)

print("Imports completed!")

# %% [markdown]
# ## 1. 設定とパス確認

# %%
# パス設定
BASE_DIR = Path('../../data/yolo_format')
ORIGINAL_MASK_DIR = Path('../../data/slice_train/axial_mask')

VIEW = 'axial'
SPLIT = 'train'

images_dir = BASE_DIR / 'images' / VIEW / SPLIT
labels_dir = BASE_DIR / 'labels' / VIEW / SPLIT
output_dir = Path('./yolo_quality_analysis_output')
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
    """
    NIfTI画像を読み込み（convert_to_yolo.pyと同じ方法）
    """
    nii = nib.load(str(path))
    data = nii.get_fdata(dtype=np.float32)
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]
    elif data.ndim > 2:
        data = data[:, :, 0] if data.shape[2] == 1 else data.squeeze()
    return data

# [FIXED] Added mask transformation function to match coordinate systems
def normalize_and_pad_mask_for_comparison(mask: np.ndarray, target_size: tuple = (256, 256)) -> np.ndarray:
    """
    マスクをYOLO画像と同じサイズにリサイズ/パディングします（最近傍補間）。
    これにより、座標系が一致し、正確な比較が可能になります。
    """
    h, w = mask.shape[:2]
    bg_value = 0

    if h == target_size[0] and w == target_size[1]:
        return mask

    if h > target_size[0] or w > target_size[1]:
        zoom_h = target_size[0] / h
        zoom_w = target_size[1] / w
        resized_mask = zoom(
            mask, (zoom_h, zoom_w), order=0, mode='constant', cval=bg_value
        )
        rh, rw = resized_mask.shape
        th, tw = target_size
        final_mask = np.full(target_size, bg_value, dtype=mask.dtype)
        h_crop = min(rh, th)
        w_crop = min(rw, tw)
        h_offset = (th - h_crop) // 2
        w_offset = (tw - w_crop) // 2
        final_mask[h_offset:h_offset+h_crop, w_offset:w_offset+w_crop] = resized_mask[0:h_crop, 0:w_crop]
        return final_mask
        
    padded = np.full(target_size, bg_value, dtype=mask.dtype)
    pad_h = (target_size[0] - h) // 2
    pad_w = (target_size[1] - w) // 2
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = mask
    return padded

def parse_yolo_label(label_path):
    """YOLOラベルファイルを解析"""
    bboxes = []
    if not label_path.exists():
        return bboxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, xc, yc, w, h = map(float, parts)
                bboxes.append({
                    'class': int(cls), 'x_center': xc, 'y_center': yc, 'width': w, 'height': h
                })
    return bboxes

def yolo_to_pixel(bbox, h, w):
    """YOLO正規化座標 → ピクセル座標に変換"""
    xc, yc = bbox['x_center'] * w, bbox['y_center'] * h
    bw, bh = bbox['width'] * w, bbox['height'] * h
    x1 = int(xc - bw / 2)
    y1 = int(yc - bh / 2)
    x2 = int(xc + bw / 2)
    y2 = int(yc + bh / 2)
    return x1, y1, x2, y2

def extract_mask_bboxes(mask):
    """マスクから各インスタンスのBBoxを抽出（値1～6）"""
    bboxes = []
    for val in range(1, 7):
        binary = (mask == val)
        if not binary.any():
            continue
        y_coords, x_coords = np.where(binary)
        bboxes.append({
            'value': val, 'x1': x_coords.min(), 'y1': y_coords.min(),
            'x2': x_coords.max(), 'y2': y_coords.max(), 'area': binary.sum()
        })
    return bboxes

def calculate_iou(box1, box2):
    """IoU（Intersection over Union）を計算"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
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
# ## 3. データ概要の確認

# %%
image_files = sorted(list(images_dir.glob('*.nii')))
label_files = sorted(list(labels_dir.glob('*.txt')))
print(f"📊 Dataset Overview:")
print(f"   Total images: {len(image_files)}")
print(f"   Total labels: {len(label_files)}")

bbox_counts = [len(parse_yolo_label(lf)) for lf in label_files]
print(f"\n📦 BBox Statistics:")
print(f"   Total BBoxes: {sum(bbox_counts)}")
print(f"   Images with BBoxes: {sum(1 for c in bbox_counts if c > 0)}")
print(f"   Images without BBoxes: {sum(1 for c in bbox_counts if c == 0)}")
print(f"   Max BBoxes per image: {max(bbox_counts) if bbox_counts else 0}")

bbox_distribution = defaultdict(int)
for count in bbox_counts: bbox_distribution[count] += 1
print(f"\n📈 BBox Count Distribution:")
for num_bbox in sorted(bbox_distribution.keys()):
    print(f"   {num_bbox} bbox(es): {bbox_distribution[num_bbox]} images")

# %% [markdown]
# ## 4. ランダムサンプルの可視化（BBoxのみ）

# %%
print("\n🎨 Generating random sample visualizations...")
fracture_labels = [lf for lf in label_files if len(parse_yolo_label(lf)) > 0]
if not fracture_labels:
    print("⚠️  No images with BBoxes found!")
else:
    np.random.seed(42)
    sample_size = min(9, len(fracture_labels))
    sample_indices = np.random.choice(len(fracture_labels), sample_size, replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    for i, idx in enumerate(sample_indices):
        label_file = fracture_labels[idx]
        image_file = images_dir / f"{label_file.stem}.nii"
        if not image_file.exists():
            axes[i].axis('off'); continue
        image = load_nifti(image_file)
        bboxes = parse_yolo_label(label_file)
        h, w = image.shape
        axes[i].imshow(image, cmap='gray')
        for bbox in bboxes:
            x1, y1, x2, y2 = yolo_to_pixel(bbox, h, w)
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.5, edgecolor='red', facecolor='none')
            axes[i].add_patch(rect)
            xc, yc = bbox['x_center'] * w, bbox['y_center'] * h
            axes[i].plot(xc, yc, 'r+', markersize=15, markeredgewidth=3)
        axes[i].set_title(f"{label_file.stem}\nBBoxes: {len(bboxes)}", fontsize=10)
        axes[i].axis('off')
    for i in range(sample_size, 9): axes[i].axis('off')
    plt.suptitle('YOLO BBox Random Samples', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'random_samples_bbox.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'random_samples_bbox.png'}")
    plt.show()

# %% [markdown]
# ## 5. マスクとBBoxの比較可視化（オーバーレイ確認）

# %%
print("\n🔍 Generating mask-bbox comparison (overlay check)...")
fig, axes = plt.subplots(3, 3, figsize=(22, 22))
sample_count = 0
max_samples = 3

for label_file in fracture_labels[:100]:
    if sample_count >= max_samples: break
    mask_path = find_original_mask_file(label_file.stem, ORIGINAL_MASK_DIR)
    if mask_path is None: continue
    image_path = images_dir / f"{label_file.stem}.nii"
    if not image_path.exists(): continue

    try:
        image = load_nifti(image_path)
        mask_original = load_nifti(mask_path)
        yolo_bboxes = parse_yolo_label(label_file)
        
        # [FIXED] Transform the original mask to match the YOLO image's coordinate system.
        mask_processed = normalize_and_pad_mask_for_comparison(mask_original, target_size=image.shape)
        # [FIXED] Extract BBoxes from the PROCESSED mask.
        mask_bboxes = extract_mask_bboxes(mask_processed)
        
    except Exception as e:
        print(f"⚠️  Error loading {label_file.stem}: {e}"); continue

    if len(yolo_bboxes) == 0 or len(mask_bboxes) == 0: continue

    h, w = image.shape
    row = sample_count

    # Column 1: YOLO Image
    axes[row, 0].imshow(image, cmap='gray')
    axes[row, 0].set_title('YOLO Image', fontsize=12, fontweight='bold')
    axes[row, 0].axis('off')

    # Column 2: PROCESSED Mask
    axes[row, 1].imshow(image, cmap='gray', alpha=0.5)
    axes[row, 1].imshow(mask_processed, cmap='Reds', alpha=0.5, vmin=0, vmax=6) # [FIXED] Display processed mask
    axes[row, 1].set_title(f'Processed Mask ({len(mask_bboxes)} instances)', fontsize=12, fontweight='bold')
    axes[row, 1].axis('off')

    # Column 3: Overlay
    axes[row, 2].imshow(image, cmap='gray', alpha=0.8)
    axes[row, 2].imshow(mask_processed, cmap='Blues', alpha=0.25, vmin=0, vmax=6) # [FIXED] Display processed mask

    # Red BBox: YOLO
    for bbox in yolo_bboxes:
        x1, y1, x2, y2 = yolo_to_pixel(bbox, h, w)
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='red', facecolor='none', label='YOLO')
        axes[row, 2].add_patch(rect)

    # Blue BBox: Mask
    for mb in mask_bboxes:
        rect = patches.Rectangle((mb['x1'], mb['y1']), mb['x2'] - mb['x1'], mb['y2'] - mb['y1'], linewidth=3, edgecolor='blue', facecolor='none', linestyle='--', label='Mask')
        axes[row, 2].add_patch(rect)

    # Calculate IoU
    iou_list = []
    for yb in yolo_bboxes:
        y_coords = yolo_to_pixel(yb, h, w)
        best_iou = 0.0
        for mb in mask_bboxes:
            m_coords = (mb['x1'], mb['y1'], mb['x2'], mb['y2'])
            iou = calculate_iou(y_coords, m_coords)
            best_iou = max(best_iou, iou)
        iou_list.append(best_iou)
    
    max_iou = max(iou_list) if iou_list else 0.0
    avg_iou = np.mean(iou_list) if iou_list else 0.0

    axes[row, 2].set_title(f'Overlay (Red=YOLO, Blue=Mask)\n'
                           f'YOLO={len(yolo_bboxes)}, Mask={len(mask_bboxes)} | '
                           f'Max IoU: {max_iou:.3f}, Avg IoU: {avg_iou:.3f}', fontsize=12, fontweight='bold')
    axes[row, 2].axis('off')
    sample_count += 1

for i in range(sample_count, max_samples):
    for j in range(3): axes[i, j].axis('off')
plt.suptitle('BBox vs Processed Mask Comparison (Red=YOLO, Blue=Mask)', fontsize=16, y=0.995, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'bbox_mask_overlay.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'bbox_mask_overlay.png'}")
plt.show()

# %% [markdown]
# ## 6. IoU統計の詳細計算

# %%
print("\n📊 Computing detailed IoU statistics...")
iou_scores, count_mismatches, per_image_stats = [], [], []

for label_file in tqdm(label_files, desc="Computing IoU"): # Analyze all files
    yolo_bboxes = parse_yolo_label(label_file)
    if not yolo_bboxes: continue
    
    mask_path = find_original_mask_file(label_file.stem, ORIGINAL_MASK_DIR)
    if not mask_path: continue
    
    image_path = images_dir / f"{label_file.stem}.nii"
    if not image_path.exists(): continue

    try:
        # [FIXED] Load image to get target dimensions, then process the mask
        image = load_nifti(image_path)
        h, w = image.shape
        mask_original = load_nifti(mask_path)
        mask_processed = normalize_and_pad_mask_for_comparison(mask_original, target_size=(h, w))
        mask_bboxes = extract_mask_bboxes(mask_processed)
    except Exception as e:
        print(f"⚠️  Error loading data for {label_file.stem}: {e}"); continue

    if len(yolo_bboxes) != len(mask_bboxes):
        count_mismatches.append({'file': label_file.stem, 'yolo': len(yolo_bboxes), 'mask': len(mask_bboxes)})

    image_ious = []
    for yb in yolo_bboxes:
        y_coords = yolo_to_pixel(yb, h, w)
        max_iou = 0.0
        for mb in mask_bboxes:
            m_coords = (mb['x1'], mb['y1'], mb['x2'], mb['y2'])
            iou = calculate_iou(y_coords, m_coords)
            max_iou = max(max_iou, iou)
        iou_scores.append(max_iou)
        image_ious.append(max_iou)

    per_image_stats.append({
        'file': label_file.stem, 'num_yolo': len(yolo_bboxes), 'num_mask': len(mask_bboxes),
        'avg_iou': np.mean(image_ious) if image_ious else 0.0,
        'min_iou': np.min(image_ious) if image_ious else 0.0,
        'max_iou': np.max(image_ious) if image_ious else 0.0
    })

print(f"\n📈 [IoU Statistics]")
if iou_scores:
    print(f"   Total BBoxes analyzed: {len(iou_scores)}")
    print(f"   Mean IoU: {np.mean(iou_scores):.4f}")
    print(f"   Median IoU: {np.median(iou_scores):.4f}")
    print(f"   Std IoU: {np.std(iou_scores):.4f}")
    print(f"   IoU >= 0.9: {sum(1 for iou in iou_scores if iou >= 0.9) / len(iou_scores) * 100:.1f}%")
    print(f"   IoU >= 0.8: {sum(1 for iou in iou_scores if iou >= 0.8) / len(iou_scores) * 100:.1f}%")
    print(f"   IoU >= 0.5: {sum(1 for iou in iou_scores if iou >= 0.5) / len(iou_scores) * 100:.1f}%")

print(f"\n⚠️  [BBox Count Mismatches]")
print(f"   Total mismatches: {len(count_mismatches)}")
if count_mismatches:
    print(f"   Showing first 10:")
    for item in count_mismatches[:10]: print(f"         {item['file']}: YOLO={item['yolo']}, Mask={item['mask']}")

if iou_scores:
    low_iou_images = [stat for stat in per_image_stats if stat['avg_iou'] < 0.85]
    print(f"\n⚠️  [Images with Low Average IoU (< 0.85)]")
    print(f"   Count: {len(low_iou_images)}")
    for item in sorted(low_iou_images, key=lambda x: x['avg_iou'])[:10]:
        print(f"         {item['file']}: Avg IoU={item['avg_iou']:.3f}, YOLO={item['num_yolo']}, Mask={item['num_mask']}")

# %% [markdown]
# ## 7. IoU分布の可視化

# %%
if iou_scores:
    print("\n📊 Plotting IoU distribution...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # Histogram
    axes[0, 0].hist(iou_scores, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0.9, color='red', linestyle='--', linewidth=2.5, label='Threshold=0.9')
    axes[0, 0].axvline(np.mean(iou_scores), color='green', linestyle='-', linewidth=2.5, label=f'Mean={np.mean(iou_scores):.3f}')
    axes[0, 0].set_title('IoU Distribution Histogram', fontsize=14, fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    # Cumulative Distribution
    sorted_iou, cumulative = np.sort(iou_scores), np.arange(1, len(iou_scores) + 1) / len(iou_scores)
    axes[0, 1].plot(sorted_iou, cumulative, linewidth=3, color='steelblue')
    axes[0, 1].axvline(0.9, color='red', linestyle='--', linewidth=2.5, label='Threshold=0.9')
    axes[0, 1].set_title('Cumulative IoU Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    # Box Plot
    axes[1, 0].boxplot(iou_scores, vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'), medianprops=dict(color='red', linewidth=2.5))
    axes[1, 0].axhline(0.9, color='red', linestyle='--', linewidth=2, label='Threshold=0.9')
    axes[1, 0].set_title('IoU Box Plot', fontsize=14, fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3, axis='y')
    # IoU Range Counts
    iou_ranges = [(0.9, 1.01, '0.9-1.0'), (0.8, 0.9, '0.8-0.9'), (0.7, 0.8, '0.7-0.8'), (0.5, 0.7, '0.5-0.7'), (0.0, 0.5, '0.0-0.5')]
    counts = [sum(1 for iou in iou_scores if low <= iou < high) for low, high, _ in iou_ranges]
    labels, colors = [label for _, _, label in iou_ranges], ['green', 'lightgreen', 'yellow', 'orange', 'red']
    axes[1, 1].barh(labels, counts, color=colors, edgecolor='black', alpha=0.8)
    axes[1, 1].set_title('IoU Range Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='x')
    for i, count in enumerate(counts): axes[1, 1].text(count + max(counts)*0.01, i, str(count), va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'iou_distribution_detailed.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'iou_distribution_detailed.png'}")
    plt.show()

# %% [markdown]
# ## 8. BBoxサイズと形状の分析

# %%
print("\n📐 Analyzing BBox sizes and shapes...")
bbox_details = []
for label_file in tqdm(label_files, desc="Analyzing BBox properties"):
    yolo_bboxes = parse_yolo_label(label_file)
    if not yolo_bboxes: continue
    image_path = images_dir / f"{label_file.stem}.nii"
    if not image_path.exists(): continue
    try: h, w = load_nifti(image_path).shape
    except: continue
    for bbox in yolo_bboxes:
        x1, y1, x2, y2 = yolo_to_pixel(bbox, h, w)
        width_px, height_px = x2 - x1, y2 - y1
        bbox_details.append({ 'width': width_px, 'height': height_px, 'area': width_px * height_px, 'aspect_ratio': max(width_px, height_px) / max(min(width_px, height_px), 1)})
bbox_sizes = [d['area'] for d in bbox_details]
bbox_aspect_ratios = [d['aspect_ratio'] for d in bbox_details]

if bbox_sizes:
    print(f"\n📊 [BBox Size Statistics (in pixels²)]")
    print(f"   Min: {np.min(bbox_sizes):.1f}, Max: {np.max(bbox_sizes):.1f}, Mean: {np.mean(bbox_sizes):.1f} ± {np.std(bbox_sizes):.1f}, Median: {np.median(bbox_sizes):.1f}")
    print(f"\n📏 [BBox Aspect Ratio Statistics]")
    print(f"   Min: {np.min(bbox_aspect_ratios):.2f}, Max: {np.max(bbox_aspect_ratios):.2f}, Mean: {np.mean(bbox_aspect_ratios):.2f} ± {np.std(bbox_aspect_ratios):.2f}, Median: {np.median(bbox_aspect_ratios):.2f}")
    print(f"\n⚠️  [Extreme BBoxes]")
    print(f"   Very small (< 100px²): {len([d for d in bbox_details if d['area'] < 100])}")
    print(f"   Extreme aspect ratio (> 10): {len([d for d in bbox_details if d['aspect_ratio'] > 10])}")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].hist(bbox_sizes, bins=50, color='coral', edgecolor='black', alpha=0.7); axes[0].set_title('BBox Size Distribution'); axes[0].legend()
    axes[1].hist(bbox_aspect_ratios, bins=50, color='lightgreen', edgecolor='black', alpha=0.7); axes[1].set_title('BBox Aspect Ratio Distribution'); axes[1].legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'bbox_size_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {output_dir / 'bbox_size_analysis.png'}")
    plt.show()

# %% [markdown]
# ## 9. 最終レポート

# %%
print("\n" + "="*80)
print("YOLO BBOX QUALITY ANALYSIS REPORT")
print("="*80)
print(f"\n[Dataset Overview]")
print(f"   Total images: {len(image_files)}, Total labels: {len(label_files)}, Images with BBoxes: {sum(1 for c in bbox_counts if c > 0)}, Total BBoxes: {sum(bbox_counts)}")

if iou_scores:
    mean_iou, median_iou = np.mean(iou_scores), np.median(iou_scores)
    print(f"\n[IoU Quality Assessment]")
    print(f"   Mean IoU: {mean_iou:.4f}, Median IoU: {median_iou:.4f}")
    print(f"   IoU >= 0.9: {sum(1 for iou in iou_scores if iou >= 0.9) / len(iou_scores) * 100:.1f}%")
    if mean_iou >= 0.9: print("   ✅ Excellent BBox quality! Alignment is very good.")
    elif mean_iou >= 0.8: print("   ✅ Good BBox quality. Minor alignment issues may exist.")
    else: print("   ⚠️  Acceptable BBox quality, but check low IoU cases.")

if bbox_sizes:
    print(f"\n[BBox Size Statistics]")
    print(f"   Mean area: {np.mean(bbox_sizes):.1f} px², Median area: {np.median(bbox_sizes):.1f} px², Mean aspect ratio: {np.mean(bbox_aspect_ratios):.2f}")

print(f"\n[BBox Count Consistency]")
if not count_mismatches: print("   ✅ All BBox counts match with mask instances.")
else: print(f"   ⚠️  {len(count_mismatches)} files have count mismatches (likely due to quality filtering).")

print(f"\n[Output Files]")
print(f"   All analysis outputs are saved in: {output_dir}/")
print("\n" + "="*80)
print("✅ Analysis completed successfully!")
print("="*80)

# %%

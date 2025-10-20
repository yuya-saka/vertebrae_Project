"""
YOLO BBox Validation and Quality Analysis

YOLOデータセット変換が適切に行われているかを検証する探索的分析スクリプト

検証項目:
1. BBox座標の妥当性（境界条件、範囲チェック）
2. マスク画像との整合性（BBoxが骨折領域を正しくカバーしているか）
3. マルチインスタンス対応の検証（重複・漏れ・分離）
4. 最小BBoxサイズ強制の効果確認（7x7強制処理の検証）
5. BBoxとマスクのIoU計算
6. エッジケース・異常値の詳細分析
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
from scipy import ndimage

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# %% [markdown]
# ## 設定

# %%
BASE_DIR = Path('../../data/yolo_format')
VIEW = 'axial'
SPLIT = 'train'  # 'train', 'val', 'test'

# 元のマスクデータのパス（整合性確認用）
ORIGINAL_MASK_DIR = Path(f'../../data/slice_{SPLIT}/{VIEW}_mask')

images_dir = BASE_DIR / 'images' / VIEW / SPLIT
labels_dir = BASE_DIR / 'labels' / VIEW / SPLIT

# 出力ディレクトリ
output_dir = Path('../notebook/bbox_validation_images')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Images directory: {images_dir}")
print(f"Labels directory: {labels_dir}")
print(f"Original masks: {ORIGINAL_MASK_DIR}")
print(f"Output directory: {output_dir}")

# %% [markdown]
# ## ユーティリティ関数

# %%
def load_nifti_image(nii_path):
    """NIfTI画像を読み込み（dataobj使用で傾き防止）"""
    nii = nib.load(str(nii_path))
    data = np.asarray(nii.dataobj, dtype=np.float32)

    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]
    elif data.ndim > 2:
        data = data[:, :, 0] if data.shape[2] == 1 else data.squeeze()

    # LAS向き補正（左右反転）
    if nii.affine[0, 0] < 0:
        data = np.fliplr(data)

    return data

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
                    'area': width * height
                })
    return bboxes

# %%
def bbox_to_pixel_coords(bbox, img_height, img_width):
    """YOLO正規化座標 → ピクセル座標に変換"""
    x_center = bbox['x_center'] * img_width
    y_center = bbox['y_center'] * img_height
    bbox_w = bbox['width'] * img_width
    bbox_h = bbox['height'] * img_height

    x_min = int(x_center - bbox_w / 2)
    y_min = int(y_center - bbox_h / 2)
    x_max = int(x_center + bbox_w / 2)
    y_max = int(y_center + bbox_h / 2)

    return x_min, y_min, x_max, y_max

# %%
def extract_mask_bboxes(mask):
    """
    マスク画像から実際の骨折領域のBBoxを抽出（Ground Truth）

    Returns:
        list of dict: マスク値ごとのBBox座標
    """
    mask_bboxes = []

    for mask_value in range(1, 7):
        binary_mask = (mask == mask_value)

        if not binary_mask.any():
            continue

        y_coords, x_coords = np.where(binary_mask)
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()

        mask_bboxes.append({
            'mask_value': mask_value,
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'width': x_max - x_min + 1,
            'height': y_max - y_min + 1,
            'area': (x_max - x_min + 1) * (y_max - y_min + 1),
        })

    return mask_bboxes

# %%
def calculate_iou(bbox1_coords, bbox2_coords):
    """
    2つのBBoxのIoU（Intersection over Union）を計算

    Args:
        bbox1_coords: (x_min, y_min, x_max, y_max)
        bbox2_coords: (x_min, y_min, x_max, y_max)

    Returns:
        float: IoU値 [0, 1]
    """
    x1_min, y1_min, x1_max, y1_max = bbox1_coords
    x2_min, y2_min, x2_max, y2_max = bbox2_coords

    # 交差領域
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # 各BBoxの面積
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Union
    union_area = bbox1_area + bbox2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area

# %%
def find_original_mask_file(yolo_filename, original_mask_dir):
    """
    YOLO画像ファイル名から元のマスクファイルパスを復元

    例: inp1003_27_slice_005.nii -> inp1003/27/mask_005.nii
    """
    parts = yolo_filename.split('_')

    if len(parts) < 4:
        return None

    case_name = parts[0]      # inp1003
    vertebra = parts[1]       # 27
    slice_idx = parts[3]      # 005 (from slice_005)

    mask_path = original_mask_dir / case_name / vertebra / f"mask_{slice_idx}.nii"

    return mask_path if mask_path.exists() else None

# %% [markdown]
# ## 1. データセット概要

# %%
print("\n" + "="*80)
print("1. DATASET OVERVIEW")
print("="*80)

image_files = sorted(list(images_dir.glob('*.nii')))
label_files = sorted(list(labels_dir.glob('*.txt')))

print(f"Total images: {len(image_files)}")
print(f"Total labels: {len(label_files)}")

# ファイル対応チェック
image_stems = {f.stem for f in image_files}
label_stems = {f.stem for f in label_files}
matched_pairs = image_stems & label_stems

print(f"Matched pairs: {len(matched_pairs)}")
print(f"Images without labels: {len(image_stems - label_stems)}")
print(f"Labels without images: {len(label_stems - image_stems)}")

# %% [markdown]
# ## 2. BBox座標の妥当性検証

# %%
print("\n" + "="*80)
print("2. BBOX COORDINATE VALIDATION")
print("="*80)

all_bboxes = []
bbox_validation_issues = {
    'out_of_range': [],
    'touches_boundary': [],
    'too_small': [],
    'extreme_aspect_ratio': [],
}

print("Analyzing BBox coordinates...")
for label_file in tqdm(label_files):
    bboxes = parse_yolo_label(label_file)

    for bbox in bboxes:
        all_bboxes.append(bbox)

        # 範囲チェック [0, 1]
        if not (0 <= bbox['x_center'] <= 1 and 0 <= bbox['y_center'] <= 1 and
                0 < bbox['width'] <= 1 and 0 < bbox['height'] <= 1):
            bbox_validation_issues['out_of_range'].append((label_file.stem, bbox))

        # 境界タッチチェック（画像端に接触）
        x_min = bbox['x_center'] - bbox['width'] / 2
        x_max = bbox['x_center'] + bbox['width'] / 2
        y_min = bbox['y_center'] - bbox['height'] / 2
        y_max = bbox['y_center'] + bbox['height'] / 2

        if x_min < 0.01 or x_max > 0.99 or y_min < 0.01 or y_max > 0.99:
            bbox_validation_issues['touches_boundary'].append((label_file.stem, bbox))

        # 極小BBoxチェック（正規化値で0.02以下 ≈ 256px画像で約5px）
        if bbox['width'] < 0.02 or bbox['height'] < 0.02:
            bbox_validation_issues['too_small'].append((label_file.stem, bbox))

        # 極端なアスペクト比
        aspect_ratio = bbox['width'] / bbox['height']
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            bbox_validation_issues['extreme_aspect_ratio'].append((label_file.stem, bbox, aspect_ratio))

print(f"\n[Validation Results]")
print(f"  Out-of-range coordinates: {len(bbox_validation_issues['out_of_range'])}")
print(f"  Touches image boundary: {len(bbox_validation_issues['touches_boundary'])}")
print(f"  Too small BBoxes (< 0.02): {len(bbox_validation_issues['too_small'])}")
print(f"  Extreme aspect ratios: {len(bbox_validation_issues['extreme_aspect_ratio'])}")

if bbox_validation_issues['out_of_range']:
    print("\n⚠️ WARNING: Out-of-range BBoxes detected!")
    for filename, bbox in bbox_validation_issues['out_of_range'][:3]:
        print(f"  {filename}: {bbox}")

# %% [markdown]
# ## 3. マスク画像との整合性検証

# %%
print("\n" + "="*80)
print("3. MASK-BBOX CONSISTENCY VALIDATION")
print("="*80)

iou_scores = []
mask_bbox_comparison = {
    'yolo_count': [],
    'mask_count': [],
    'count_mismatch': [],
    'low_iou': [],  # IoU < 0.8
    'good_iou': [],  # IoU >= 0.8
}

print("Comparing YOLO BBoxes with original masks...")

# サンプル数を制限（全データで実行する場合はこの行を削除）
sample_labels = label_files[:min(100, len(label_files))]

for label_file in tqdm(sample_labels):
    yolo_bboxes = parse_yolo_label(label_file)

    # 元のマスクファイルを見つける
    mask_path = find_original_mask_file(label_file.stem, ORIGINAL_MASK_DIR)

    if mask_path is None or not mask_path.exists():
        continue

    # マスク画像を読み込み
    mask = load_nifti_image(mask_path)
    mask_bboxes = extract_mask_bboxes(mask)

    # BBox数の比較
    mask_bbox_comparison['yolo_count'].append(len(yolo_bboxes))
    mask_bbox_comparison['mask_count'].append(len(mask_bboxes))

    if len(yolo_bboxes) != len(mask_bboxes):
        mask_bbox_comparison['count_mismatch'].append({
            'filename': label_file.stem,
            'yolo': len(yolo_bboxes),
            'mask': len(mask_bboxes)
        })

    # IoU計算（各YOLOのBBoxと最も近いマスクBBoxとのIoU）
    h, w = mask.shape

    for yolo_bbox in yolo_bboxes:
        yolo_coords = bbox_to_pixel_coords(yolo_bbox, h, w)

        # 最も近いマスクBBoxとのIoUを計算
        max_iou = 0.0
        for mask_bbox in mask_bboxes:
            mask_coords = (
                mask_bbox['x_min'], mask_bbox['y_min'],
                mask_bbox['x_max'], mask_bbox['y_max']
            )
            iou = calculate_iou(yolo_coords, mask_coords)
            max_iou = max(max_iou, iou)

        iou_scores.append(max_iou)

        if max_iou < 0.8:
            mask_bbox_comparison['low_iou'].append({
                'filename': label_file.stem,
                'iou': max_iou,
                'yolo_bbox': yolo_bbox
            })
        else:
            mask_bbox_comparison['good_iou'].append(max_iou)

print(f"\n[Mask-BBox Consistency Results]")
print(f"  Samples analyzed: {len(sample_labels)}")
print(f"  BBox count mismatches: {len(mask_bbox_comparison['count_mismatch'])}")
print(f"  Low IoU (< 0.8): {len(mask_bbox_comparison['low_iou'])}")
print(f"  Good IoU (>= 0.8): {len(mask_bbox_comparison['good_iou'])}")

if iou_scores:
    print(f"\n[IoU Statistics]")
    print(f"  Mean IoU: {np.mean(iou_scores):.4f}")
    print(f"  Median IoU: {np.median(iou_scores):.4f}")
    print(f"  Min IoU: {np.min(iou_scores):.4f}")
    print(f"  Max IoU: {np.max(iou_scores):.4f}")

if mask_bbox_comparison['count_mismatch']:
    print(f"\n⚠️ WARNING: BBox count mismatches detected!")
    for item in mask_bbox_comparison['count_mismatch'][:5]:
        print(f"  {item['filename']}: YOLO={item['yolo']}, Mask={item['mask']}")

# %% [markdown]
# ## 4. 最小BBoxサイズ強制の効果確認

# %%
print("\n" + "="*80)
print("4. MINIMUM BBOX SIZE ENFORCEMENT VALIDATION")
print("="*80)

MIN_SIZE_NORMALIZED = 10 / 256  # 10px / 256px ≈ 0.039

small_width_bboxes = [b for b in all_bboxes if b['width'] < MIN_SIZE_NORMALIZED]
small_height_bboxes = [b for b in all_bboxes if b['height'] < MIN_SIZE_NORMALIZED]

print(f"Expected minimum size (normalized): {MIN_SIZE_NORMALIZED:.4f}")
print(f"BBoxes with width < minimum: {len(small_width_bboxes)}")
print(f"BBoxes with height < minimum: {len(small_height_bboxes)}")

if small_width_bboxes or small_height_bboxes:
    print("\n⚠️ WARNING: BBoxes smaller than expected minimum detected!")
    print("This suggests the minimum size enforcement (7x7) may not be working correctly.")

    if small_width_bboxes:
        print(f"\nSmallest widths:")
        sorted_widths = sorted(small_width_bboxes, key=lambda x: x['width'])[:5]
        for bbox in sorted_widths:
            print(f"  Width: {bbox['width']:.6f} ({bbox['width'] * 256:.1f}px)")
else:
    print("\n✓ All BBoxes meet the minimum size requirement.")

# %% [markdown]
# ## 5. 可視化: BBoxとマスクの比較

# %%
def draw_bbox_mask_comparison(ax_image, ax_mask, ax_overlay, image, mask, yolo_bboxes, mask_bboxes, title):
    """画像、マスク、BBoxを並べて可視化"""
    h, w = image.shape

    # 画像のみ
    ax_image.imshow(image, cmap='gray')
    ax_image.set_title(f"{title}\n(Image)")
    ax_image.axis('off')

    # マスクのみ
    ax_mask.imshow(mask, cmap='tab10', vmin=0, vmax=6)
    ax_mask.set_title(f"(Mask: {len(mask_bboxes)} instances)")
    ax_mask.axis('off')

    # 画像 + YOLO BBox + Mask BBox
    ax_overlay.imshow(image, cmap='gray', alpha=0.7)
    ax_overlay.imshow(mask, cmap='Reds', alpha=0.3, vmin=0, vmax=6)

    # YOLO BBox (赤)
    for bbox in yolo_bboxes:
        x_min, y_min, x_max, y_max = bbox_to_pixel_coords(bbox, h, w)
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='red', facecolor='none', label='YOLO'
        )
        ax_overlay.add_patch(rect)

    # Mask BBox (青)
    for mask_bbox in mask_bboxes:
        rect = patches.Rectangle(
            (mask_bbox['x_min'], mask_bbox['y_min']),
            mask_bbox['width'], mask_bbox['height'],
            linewidth=2, edgecolor='blue', facecolor='none', linestyle='--', label='Mask'
        )
        ax_overlay.add_patch(rect)

    ax_overlay.set_title(f"(Overlay: YOLO={len(yolo_bboxes)}, Mask={len(mask_bboxes)})")
    ax_overlay.axis('off')

# %%
# ケース1: 良好なマッチング（IoU高い）
print("\nGenerating visualizations...")

if mask_bbox_comparison['good_iou']:
    print("Creating good IoU examples...")

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    sample_count = 0
    for label_file in label_files[:50]:
        if sample_count >= 3:
            break

        yolo_bboxes = parse_yolo_label(label_file)
        if not yolo_bboxes:
            continue

        mask_path = find_original_mask_file(label_file.stem, ORIGINAL_MASK_DIR)
        if mask_path is None or not mask_path.exists():
            continue

        image_path = images_dir / f"{label_file.stem}.nii"
        if not image_path.exists():
            continue

        image = load_nifti_image(image_path)
        mask = load_nifti_image(mask_path)
        mask_bboxes = extract_mask_bboxes(mask)

        # IoUが高いかチェック
        h, w = mask.shape
        has_good_iou = False
        for yolo_bbox in yolo_bboxes:
            yolo_coords = bbox_to_pixel_coords(yolo_bbox, h, w)
            for mask_bbox in mask_bboxes:
                mask_coords = (mask_bbox['x_min'], mask_bbox['y_min'],
                              mask_bbox['x_max'], mask_bbox['y_max'])
                if calculate_iou(yolo_coords, mask_coords) >= 0.8:
                    has_good_iou = True
                    break
            if has_good_iou:
                break

        if not has_good_iou:
            continue

        row = sample_count
        draw_bbox_mask_comparison(
            axes[row, 0], axes[row, 1], axes[row, 2],
            image, mask, yolo_bboxes, mask_bboxes,
            f"Good Match: {label_file.stem}"
        )
        sample_count += 1

    # 未使用のサブプロットを非表示
    for i in range(sample_count, 3):
        for j in range(3):
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'good_bbox_matches.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'good_bbox_matches.png'}")
    plt.close()

# %%
# ケース2: 不一致・問題のあるケース
if mask_bbox_comparison['low_iou'] or mask_bbox_comparison['count_mismatch']:
    print("Creating problematic cases visualization...")

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    sample_count = 0

    # IoUが低いケース
    for item in mask_bbox_comparison['low_iou'][:3]:
        if sample_count >= 3:
            break

        filename = item['filename']
        label_file = labels_dir / f"{filename}.txt"

        yolo_bboxes = parse_yolo_label(label_file)
        mask_path = find_original_mask_file(filename, ORIGINAL_MASK_DIR)

        if mask_path is None or not mask_path.exists():
            continue

        image_path = images_dir / f"{filename}.nii"
        if not image_path.exists():
            continue

        image = load_nifti_image(image_path)
        mask = load_nifti_image(mask_path)
        mask_bboxes = extract_mask_bboxes(mask)

        row = sample_count
        draw_bbox_mask_comparison(
            axes[row, 0], axes[row, 1], axes[row, 2],
            image, mask, yolo_bboxes, mask_bboxes,
            f"Low IoU ({item['iou']:.3f}): {filename}"
        )
        sample_count += 1

    # 未使用のサブプロットを非表示
    for i in range(sample_count, 3):
        for j in range(3):
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'problematic_bbox_cases.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'problematic_bbox_cases.png'}")
    plt.close()

# %% [markdown]
# ## 6. IoU分布の可視化

# %%
if iou_scores:
    print("Creating IoU distribution plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ヒストグラム
    axes[0].hist(iou_scores, bins=50, color='steelblue', edgecolor='black')
    axes[0].axvline(0.8, color='red', linestyle='--', linewidth=2, label='Threshold (0.8)')
    axes[0].set_xlabel('IoU Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('IoU Distribution (YOLO BBox vs Mask BBox)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 累積分布
    sorted_iou = np.sort(iou_scores)
    cumulative = np.arange(1, len(sorted_iou) + 1) / len(sorted_iou)
    axes[1].plot(sorted_iou, cumulative, linewidth=2, color='steelblue')
    axes[1].axvline(0.8, color='red', linestyle='--', linewidth=2, label='Threshold (0.8)')
    axes[1].set_xlabel('IoU Score')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Cumulative IoU Distribution')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'iou_distribution.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'iou_distribution.png'}")
    plt.close()

# %% [markdown]
# ## 7. サマリーレポート

# %%
print("\n" + "="*80)
print("BBOX VALIDATION SUMMARY REPORT")
print("="*80)

print(f"\n[Dataset]")
print(f"  Split: {SPLIT}")
print(f"  View: {VIEW}")
print(f"  Total images: {len(image_files)}")
print(f"  Total labels: {len(label_files)}")
print(f"  Total BBoxes: {len(all_bboxes)}")

print(f"\n[Coordinate Validation]")
print(f"  Out-of-range: {len(bbox_validation_issues['out_of_range'])} ❌" if bbox_validation_issues['out_of_range'] else "  Out-of-range: 0 ✓")
print(f"  Touches boundary: {len(bbox_validation_issues['touches_boundary'])}")
print(f"  Too small (< 0.02): {len(bbox_validation_issues['too_small'])}")
print(f"  Extreme aspect ratio: {len(bbox_validation_issues['extreme_aspect_ratio'])}")

print(f"\n[Mask Consistency] (analyzed: {len(sample_labels)} samples)")
print(f"  BBox count mismatches: {len(mask_bbox_comparison['count_mismatch'])} ⚠️" if mask_bbox_comparison['count_mismatch'] else "  BBox count mismatches: 0 ✓")
print(f"  Low IoU (< 0.8): {len(mask_bbox_comparison['low_iou'])} ⚠️" if mask_bbox_comparison['low_iou'] else "  Low IoU: 0 ✓")

if iou_scores:
    print(f"  Mean IoU: {np.mean(iou_scores):.4f}")
    print(f"  Median IoU: {np.median(iou_scores):.4f}")
    print(f"  IoU >= 0.8: {sum(1 for iou in iou_scores if iou >= 0.8) / len(iou_scores) * 100:.1f}%")

print(f"\n[Minimum Size Enforcement]")
print(f"  Expected minimum (normalized): {MIN_SIZE_NORMALIZED:.4f}")
print(f"  Violations (width): {len(small_width_bboxes)} ❌" if small_width_bboxes else "  Violations (width): 0 ✓")
print(f"  Violations (height): {len(small_height_bboxes)} ❌" if small_height_bboxes else "  Violations (height): 0 ✓")

print(f"\n[Overall Assessment]")
all_checks_passed = (
    len(bbox_validation_issues['out_of_range']) == 0 and
    len(mask_bbox_comparison['count_mismatch']) == 0 and
    len(mask_bbox_comparison['low_iou']) < len(sample_labels) * 0.1 and  # 10%未満
    len(small_width_bboxes) == 0 and
    len(small_height_bboxes) == 0
)

if all_checks_passed:
    print("  ✅ YOLO BBoxes are correctly generated!")
    print("  All validation checks passed.")
else:
    print("  ⚠️ Some issues detected. Please review the detailed output above.")

print(f"\n[Output]")
print(f"  Visualizations saved to: {output_dir}/")
print("="*80)

print("\nValidation completed!")

# %%

"""
YOLO BBox Visual Quality Check

YOLOデータセット変換の品質を可視化で確認するスクリプト

主な確認項目:
1. BBoxが実際の骨折領域を正しくカバーしているか
2. マスク画像との整合性（IoU計算）
3. 複数インスタンスの分離状況
4. BBoxサイズと形状の妥当性
"""

# %%
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm

# プロット設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['figure.figsize'] = (16, 10)

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
output_dir = Path('./bbox_validation_images')
output_dir.mkdir(exist_ok=True)

print(f"Images: {images_dir}")
print(f"Labels: {labels_dir}")
print(f"Masks: {ORIGINAL_MASK_DIR}")
print(f"Output: {output_dir}")
print(f"\nImages exist: {images_dir.exists()}")
print(f"Labels exist: {labels_dir.exists()}")
print(f"Masks exist: {ORIGINAL_MASK_DIR.exists()}")

# %% [markdown]
# ## 2. ユーティリティ関数

# %%
def load_nifti(path):
    """NIfTI画像を読み込み"""
    nii = nib.load(str(path))
    data = np.asarray(nii.dataobj, dtype=np.float32)

    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]

    # LAS向き補正
    if nii.affine[0, 0] < 0:
        data = np.fliplr(data)

    return data

def parse_yolo_label(label_path):
    """YOLOラベルを解析"""
    bboxes = []
    if not label_path.exists():
        return bboxes

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, xc, yc, w, h = map(float, parts)
                bboxes.append({
                    'class': int(cls),
                    'x_center': xc,
                    'y_center': yc,
                    'width': w,
                    'height': h
                })
    return bboxes

def yolo_to_pixel(bbox, h, w):
    """YOLO正規化座標 → ピクセル座標"""
    xc = bbox['x_center'] * w
    yc = bbox['y_center'] * h
    bw = bbox['width'] * w
    bh = bbox['height'] * h

    x1 = int(xc - bw / 2)
    y1 = int(yc - bh / 2)
    x2 = int(xc + bw / 2)
    y2 = int(yc + bh / 2)

    return x1, y1, x2, y2

def extract_mask_bboxes(mask):
    """マスクから骨折領域のBBoxを抽出"""
    bboxes = []
    for val in range(1, 7):
        binary = (mask == val)
        if not binary.any():
            continue

        y_coords, x_coords = np.where(binary)
        bboxes.append({
            'value': val,
            'x1': x_coords.min(),
            'y1': y_coords.min(),
            'x2': x_coords.max(),
            'y2': y_coords.max()
        })
    return bboxes

def calculate_iou(box1, box2):
    """IoU計算"""
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

def find_mask_file(yolo_name, mask_dir):
    """YOLO名から元のマスクファイルを探す"""
    parts = yolo_name.split('_')
    if len(parts) < 4:
        return None

    case = parts[0]
    vertebra = parts[1]
    slice_idx = parts[3]

    mask_path = mask_dir / case / vertebra / f"mask_{slice_idx}.nii"
    return mask_path if mask_path.exists() else None

# %% [markdown]
# ## 3. データ概要確認

# %%
image_files = sorted(list(images_dir.glob('*.nii')))
label_files = sorted(list(labels_dir.glob('*.txt')))

print(f"Total images: {len(image_files)}")
print(f"Total labels: {len(label_files)}")

# BBox数を集計
bbox_counts = []
for label_file in label_files:
    bboxes = parse_yolo_label(label_file)
    bbox_counts.append(len(bboxes))

print(f"\nBBox statistics:")
print(f"  Total BBoxes: {sum(bbox_counts)}")
print(f"  Images with BBox: {sum(1 for c in bbox_counts if c > 0)}")
print(f"  Images without BBox: {sum(1 for c in bbox_counts if c == 0)}")
print(f"  Max BBoxes per image: {max(bbox_counts) if bbox_counts else 0}")

# %% [markdown]
# ## 4. ランダムサンプルの可視化（BBoxのみ）

# %%
print("Generating random sample visualizations...")

# 骨折ありの画像をサンプリング
fracture_labels = [lf for lf in label_files if len(parse_yolo_label(lf)) > 0]
np.random.seed(42)
sample_indices = np.random.choice(len(fracture_labels), min(9, len(fracture_labels)), replace=False)

fig, axes = plt.subplots(3, 3, figsize=(18, 18))
axes = axes.flatten()

for i, idx in enumerate(sample_indices):
    label_file = fracture_labels[idx]
    image_file = images_dir / f"{label_file.stem}.nii"

    if not image_file.exists():
        axes[i].axis('off')
        continue

    # 画像とBBoxを読み込み
    image = load_nifti(image_file)
    bboxes = parse_yolo_label(label_file)
    h, w = image.shape

    # 描画
    axes[i].imshow(image, cmap='gray')

    for bbox in bboxes:
        x1, y1, x2, y2 = yolo_to_pixel(bbox, h, w)
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        axes[i].add_patch(rect)

        # 中心点
        xc = bbox['x_center'] * w
        yc = bbox['y_center'] * h
        axes[i].plot(xc, yc, 'r+', markersize=12, markeredgewidth=2)

    axes[i].set_title(f"{label_file.stem}\nBBoxes: {len(bboxes)}", fontsize=10)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(output_dir / 'random_samples.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'random_samples.png'}")
plt.show()

# %% [markdown]
# ## 5. マスクとBBoxの比較可視化

# %%
print("\nGenerating mask-bbox comparison...")

fig, axes = plt.subplots(3, 3, figsize=(20, 20))

sample_count = 0
for label_file in fracture_labels[:100]:
    if sample_count >= 3:
        break

    # マスクファイルを探す
    mask_path = find_mask_file(label_file.stem, ORIGINAL_MASK_DIR)
    if mask_path is None:
        continue

    image_path = images_dir / f"{label_file.stem}.nii"
    if not image_path.exists():
        continue

    # データ読み込み
    image = load_nifti(image_path)
    mask = load_nifti(mask_path)
    yolo_bboxes = parse_yolo_label(label_file)
    mask_bboxes = extract_mask_bboxes(mask)

    if len(yolo_bboxes) == 0 or len(mask_bboxes) == 0:
        continue

    h, w = image.shape

    # 3列構成: 画像, マスク, オーバーレイ
    row = sample_count

    # 列1: 画像のみ
    axes[row, 0].imshow(image, cmap='gray')
    axes[row, 0].set_title('Image', fontsize=12)
    axes[row, 0].axis('off')

    # 列2: マスク
    axes[row, 1].imshow(mask, cmap='tab10', vmin=0, vmax=6)
    axes[row, 1].set_title(f'Mask ({len(mask_bboxes)} instances)', fontsize=12)
    axes[row, 1].axis('off')

    # 列3: オーバーレイ (画像 + YOLO赤 + Mask青)
    axes[row, 2].imshow(image, cmap='gray', alpha=0.7)
    axes[row, 2].imshow(mask, cmap='Reds', alpha=0.3, vmin=0, vmax=6)

    # YOLOのBBox (赤)
    for bbox in yolo_bboxes:
        x1, y1, x2, y2 = yolo_to_pixel(bbox, h, w)
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor='red', facecolor='none', label='YOLO'
        )
        axes[row, 2].add_patch(rect)

    # マスクのBBox (青破線)
    for mb in mask_bboxes:
        rect = patches.Rectangle(
            (mb['x1'], mb['y1']), mb['x2'] - mb['x1'], mb['y2'] - mb['y1'],
            linewidth=2.5, edgecolor='blue', facecolor='none',
            linestyle='--', label='Mask'
        )
        axes[row, 2].add_patch(rect)

    # IoU計算
    max_iou = 0.0
    for yb in yolo_bboxes:
        y_coords = yolo_to_pixel(yb, h, w)
        for mb in mask_bboxes:
            m_coords = (mb['x1'], mb['y1'], mb['x2'], mb['y2'])
            iou = calculate_iou(y_coords, m_coords)
            max_iou = max(max_iou, iou)

    axes[row, 2].set_title(
        f'Overlay (YOLO={len(yolo_bboxes)}, Mask={len(mask_bboxes)})\nMax IoU: {max_iou:.3f}',
        fontsize=12
    )
    axes[row, 2].axis('off')

    sample_count += 1

# 未使用セルを非表示
for i in range(sample_count, 3):
    for j in range(3):
        axes[i, j].axis('off')

plt.suptitle('BBox vs Mask Comparison (Red=YOLO, Blue=Mask)', fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'bbox_mask_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'bbox_mask_comparison.png'}")
plt.show()

# %% [markdown]
# ## 6. IoU統計の計算

# %%
print("\nCalculating IoU statistics...")

iou_scores = []
count_mismatches = []

# サンプル数を制限（全データで実行する場合は[:100]を削除）
for label_file in tqdm(label_files[:100], desc="Computing IoU"):
    yolo_bboxes = parse_yolo_label(label_file)
    if len(yolo_bboxes) == 0:
        continue

    mask_path = find_mask_file(label_file.stem, ORIGINAL_MASK_DIR)
    if mask_path is None:
        continue

    mask = load_nifti(mask_path)
    mask_bboxes = extract_mask_bboxes(mask)
    h, w = mask.shape

    # BBox数の不一致をチェック
    if len(yolo_bboxes) != len(mask_bboxes):
        count_mismatches.append({
            'file': label_file.stem,
            'yolo': len(yolo_bboxes),
            'mask': len(mask_bboxes)
        })

    # IoU計算
    for yb in yolo_bboxes:
        y_coords = yolo_to_pixel(yb, h, w)
        max_iou = 0.0
        for mb in mask_bboxes:
            m_coords = (mb['x1'], mb['y1'], mb['x2'], mb['y2'])
            iou = calculate_iou(y_coords, m_coords)
            max_iou = max(max_iou, iou)
        iou_scores.append(max_iou)

print(f"\n[IoU Statistics]")
print(f"  Samples: {len(iou_scores)}")
if iou_scores:
    print(f"  Mean IoU: {np.mean(iou_scores):.4f}")
    print(f"  Median IoU: {np.median(iou_scores):.4f}")
    print(f"  Min IoU: {np.min(iou_scores):.4f}")
    print(f"  Max IoU: {np.max(iou_scores):.4f}")
    print(f"  IoU >= 0.8: {sum(1 for iou in iou_scores if iou >= 0.8) / len(iou_scores) * 100:.1f}%")

print(f"\n[BBox Count Mismatches]")
print(f"  Total: {len(count_mismatches)}")
if count_mismatches:
    for item in count_mismatches[:5]:
        print(f"  {item['file']}: YOLO={item['yolo']}, Mask={item['mask']}")

# %% [markdown]
# ## 7. IoU分布の可視化

# %%
if iou_scores:
    print("\nPlotting IoU distribution...")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # ヒストグラム
    axes[0].hist(iou_scores, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(0.8, color='red', linestyle='--', linewidth=2, label='Threshold=0.8')
    axes[0].set_xlabel('IoU Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('IoU Distribution', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 累積分布
    sorted_iou = np.sort(iou_scores)
    cumulative = np.arange(1, len(sorted_iou) + 1) / len(sorted_iou)
    axes[1].plot(sorted_iou, cumulative, linewidth=2.5, color='steelblue')
    axes[1].axvline(0.8, color='red', linestyle='--', linewidth=2, label='Threshold=0.8')
    axes[1].axhline(0.8, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[1].set_xlabel('IoU Score', fontsize=12)
    axes[1].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1].set_title('Cumulative IoU Distribution', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'iou_distribution.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'iou_distribution.png'}")
    plt.show()

# %% [markdown]
# ## 8. 最終レポート

# %%
print("\n" + "="*70)
print("BBOX VALIDATION REPORT")
print("="*70)

print(f"\n[Dataset]")
print(f"  Total images: {len(image_files)}")
print(f"  Total labels: {len(label_files)}")
print(f"  Images with BBox: {sum(1 for c in bbox_counts if c > 0)}")
print(f"  Total BBoxes: {sum(bbox_counts)}")

if iou_scores:
    print(f"\n[IoU Quality]")
    print(f"  Mean IoU: {np.mean(iou_scores):.4f}")
    print(f"  Median IoU: {np.median(iou_scores):.4f}")
    print(f"  IoU >= 0.8: {sum(1 for iou in iou_scores if iou >= 0.8) / len(iou_scores) * 100:.1f}%")

    if np.mean(iou_scores) >= 0.8:
        print("  ✅ Excellent BBox quality!")
    elif np.mean(iou_scores) >= 0.6:
        print("  ⚠️  Acceptable, but could be improved")
    else:
        print("  ❌ Poor BBox quality, review needed")

print(f"\n[BBox Count Consistency]")
if len(count_mismatches) == 0:
    print("  ✅ All BBox counts match with mask instances")
else:
    print(f"  ⚠️  {len(count_mismatches)} files have count mismatches")

print(f"\n[Output]")
print(f"  Visualizations saved to: {output_dir}/")
print("="*70)

print("\n✅ Validation completed!")
print(f"\nCheck the following files:")
print(f"  - {output_dir / 'random_samples.png'}")
print(f"  - {output_dir / 'bbox_mask_comparison.png'}")
print(f"  - {output_dir / 'iou_distribution.png'}")

# %%

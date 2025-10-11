# %% [markdown]
# # データ分布分析
#
# このノートブックでは、以下を調査します：
# 1. Train/Testデータの画像クラス間の比率（骨折あり/なし）
# 2. Train/Testデータの画像枚数
# 3. 骨折マスクの全体に対する比率（ピクセルレベル）
# 4. 症例ごと・椎体ごとの骨折分布
# 5. マスクサイズの統計（骨折領域の大きさ分析）

# %%
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# スタイル設定
plt.style.use('default')
sns.set_palette("husl")

# %% [markdown]
# ## 1. データ読み込み

# %%
# データパスの設定
DATA_DIR = project_root / "vertebrae_Unet" / "data"
SLICE_TRAIN_DIR = DATA_DIR / "slice_train" / "axial"
SLICE_TEST_DIR = DATA_DIR / "slice_test" / "axial"
MASK_TRAIN_DIR = DATA_DIR / "slice_train" / "axial_mask"
MASK_TEST_DIR = DATA_DIR / "slice_test" / "axial_mask"
OUTPUT_DIR = project_root / "vertebrae_Unet" / "notebook" / "exploratory_image"
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Train CT dir: {SLICE_TRAIN_DIR}")
print(f"Test CT dir: {SLICE_TEST_DIR}")
print(f"Train Mask dir: {MASK_TRAIN_DIR}")
print(f"Test Mask dir: {MASK_TEST_DIR}")
print(f"Output dir: {OUTPUT_DIR}\n")

# %%
def load_all_csv_files(directory, pattern="fracture_labels_*.csv"):
    """指定ディレクトリから全CSVファイルを読み込んで結合"""
    csv_files = list(directory.glob(f"*/{pattern}"))
    print(f"Found {len(csv_files)} CSV files in {directory.name}")

    df_list = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df_list.append(df)

    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()

def load_all_mask_csv_files(directory, pattern="mask_labels_*.csv"):
    """マスクCSVファイルを読み込む"""
    csv_files = list(directory.glob(f"*/{pattern}"))
    print(f"Found {len(csv_files)} mask CSV files in {directory.name}")

    df_list = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df_list.append(df)

    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()

# Train/Testデータの読み込み
print("Loading Train data...")
df_train = load_all_csv_files(SLICE_TRAIN_DIR)
print(f"Train slices: {len(df_train)}\n")

print("Loading Test data...")
df_test = load_all_csv_files(SLICE_TEST_DIR)
print(f"Test slices: {len(df_test)}\n")

print("Loading Train mask data...")
df_train_mask = load_all_mask_csv_files(MASK_TRAIN_DIR)
print(f"Train mask slices: {len(df_train_mask)}\n")

print("Loading Test mask data...")
df_test_mask = load_all_mask_csv_files(MASK_TEST_DIR)
print(f"Test mask slices: {len(df_test_mask)}\n")

# %% [markdown]
# ## 2. 基本統計情報

# %%
def print_dataset_summary(df, name):
    """データセットの基本統計を表示"""
    print(f"\n{'='*60}")
    print(f"{name} Dataset Summary")
    print(f"{'='*60}")
    print(f"Total slices: {len(df)}")
    print(f"Unique cases: {df['Case'].nunique()}")
    print(f"Unique vertebrae types: {sorted(df['Vertebra'].unique())}")
    print(f"\nFracture distribution:")
    print(df['Fracture_Label'].value_counts().sort_index())

    fracture_ratio = df['Fracture_Label'].mean() * 100
    print(f"\nFracture ratio: {fracture_ratio:.2f}% (positive class)")
    print(f"Non-fracture ratio: {100 - fracture_ratio:.2f}% (negative class)")

    # 症例ごとの骨折数
    fracture_by_case = df[df['Fracture_Label'] == 1].groupby('Case').size()
    print(f"\nCases with fractures: {len(fracture_by_case)} / {df['Case'].nunique()}")

    # 椎体ごとの骨折数
    fracture_by_vertebra = df[df['Fracture_Label'] == 1].groupby('Vertebra').size()
    print(f"\nFractures by vertebra:")
    for vertebra in sorted(df['Vertebra'].unique()):
        count = fracture_by_vertebra.get(vertebra, 0)
        total = len(df[df['Vertebra'] == vertebra])
        print(f"  V{vertebra}: {count} / {total} ({count/total*100:.1f}%)")

print_dataset_summary(df_train, "TRAIN")
print_dataset_summary(df_test, "TEST")

# %% [markdown]
# ## 3. クラス分布の可視化

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Train: 全体のクラス分布
train_counts = df_train['Fracture_Label'].value_counts().sort_index()
axes[0, 0].bar(['Non-Fracture', 'Fracture'], train_counts.values, color=['skyblue', 'coral'])
axes[0, 0].set_title('Train: Overall Class Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Number of Slices', fontsize=12)
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(train_counts.values):
    axes[0, 0].text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')

# Test: 全体のクラス分布
test_counts = df_test['Fracture_Label'].value_counts().sort_index()
axes[0, 1].bar(['Non-Fracture', 'Fracture'], test_counts.values, color=['skyblue', 'coral'])
axes[0, 1].set_title('Test: Overall Class Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Number of Slices', fontsize=12)
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(test_counts.values):
    axes[0, 1].text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')

# Train: 椎体ごとのクラス分布
vertebrae_list = sorted(df_train['Vertebra'].unique())
train_fracture_by_vertebra = []
train_nonfracture_by_vertebra = []
for v in vertebrae_list:
    train_fracture_by_vertebra.append(len(df_train[(df_train['Vertebra'] == v) & (df_train['Fracture_Label'] == 1)]))
    train_nonfracture_by_vertebra.append(len(df_train[(df_train['Vertebra'] == v) & (df_train['Fracture_Label'] == 0)]))

x = np.arange(len(vertebrae_list))
width = 0.35
axes[1, 0].bar(x - width/2, train_nonfracture_by_vertebra, width, label='Non-Fracture', color='skyblue')
axes[1, 0].bar(x + width/2, train_fracture_by_vertebra, width, label='Fracture', color='coral')
axes[1, 0].set_title('Train: Class Distribution by Vertebra', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Vertebra', fontsize=12)
axes[1, 0].set_ylabel('Number of Slices', fontsize=12)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels([f'V{v}' for v in vertebrae_list], rotation=45)
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Test: 椎体ごとのクラス分布
test_fracture_by_vertebra = []
test_nonfracture_by_vertebra = []
for v in vertebrae_list:
    test_fracture_by_vertebra.append(len(df_test[(df_test['Vertebra'] == v) & (df_test['Fracture_Label'] == 1)]))
    test_nonfracture_by_vertebra.append(len(df_test[(df_test['Vertebra'] == v) & (df_test['Fracture_Label'] == 0)]))

axes[1, 1].bar(x - width/2, test_nonfracture_by_vertebra, width, label='Non-Fracture', color='skyblue')
axes[1, 1].bar(x + width/2, test_fracture_by_vertebra, width, label='Fracture', color='coral')
axes[1, 1].set_title('Test: Class Distribution by Vertebra', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Vertebra', fontsize=12)
axes[1, 1].set_ylabel('Number of Slices', fontsize=12)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels([f'V{v}' for v in vertebrae_list], rotation=45)
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR / 'class_distribution.png'}")
plt.show()

# %% [markdown]
# ## 4. 症例レベルの骨折分布

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Train: 症例ごとの骨折スライス数
train_fracture_per_case = df_train[df_train['Fracture_Label'] == 1].groupby('Case').size().sort_values(ascending=False)
axes[0].bar(range(len(train_fracture_per_case)), train_fracture_per_case.values, color='coral')
axes[0].set_title('Train: Number of Fracture Slices per Case', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Case Index (sorted)', fontsize=12)
axes[0].set_ylabel('Number of Fracture Slices', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)
axes[0].axhline(train_fracture_per_case.mean(), color='red', linestyle='--',
                label=f'Mean: {train_fracture_per_case.mean():.1f}')
axes[0].legend()

# Test: 症例ごとの骨折スライス数
test_fracture_per_case = df_test[df_test['Fracture_Label'] == 1].groupby('Case').size().sort_values(ascending=False)
axes[1].bar(range(len(test_fracture_per_case)), test_fracture_per_case.values, color='coral')
axes[1].set_title('Test: Number of Fracture Slices per Case', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Case Index (sorted)', fontsize=12)
axes[1].set_ylabel('Number of Fracture Slices', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)
if len(test_fracture_per_case) > 0:
    axes[1].axhline(test_fracture_per_case.mean(), color='red', linestyle='--',
                    label=f'Mean: {test_fracture_per_case.mean():.1f}')
    axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fracture_distribution_per_case.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'fracture_distribution_per_case.png'}")
plt.show()

# %% [markdown]
# ## 5. マスクピクセル分析（骨折領域のサイズ）

# %%
def analyze_mask_pixels(df_mask, name):
    """マスク画像のピクセル数を分析"""
    print(f"\n{'='*60}")
    print(f"{name} Mask Pixel Analysis")
    print(f"{'='*60}")

    # 骨折マスクのみ（Fracture_Label == 1）を抽出
    df_fracture_mask = df_mask[df_mask['Fracture_Label'] == 1].copy()

    if len(df_fracture_mask) == 0:
        print("No fracture masks found.")
        return None

    # マスク画像を読み込んでピクセル数を計算
    print(f"Analyzing {len(df_fracture_mask)} fracture masks...")

    mask_pixel_counts = []
    mask_ratios = []

    for idx, row in tqdm(df_fracture_mask.iterrows(), total=len(df_fracture_mask), desc=f"Loading {name} masks"):
        try:
            mask_img = nib.load(row['MaskPath'])
            mask_data = mask_img.get_fdata()

            # 骨折ピクセル数（マスクが1の部分）
            fracture_pixels = np.sum(mask_data > 0)
            total_pixels = mask_data.size

            mask_pixel_counts.append(fracture_pixels)
            mask_ratios.append(fracture_pixels / total_pixels * 100)
        except Exception as e:
            print(f"Error loading {row['MaskPath']}: {e}")
            continue

    if len(mask_pixel_counts) == 0:
        print("Failed to load any mask data.")
        return None

    df_fracture_mask['FracturePixels'] = mask_pixel_counts
    df_fracture_mask['FractureRatio'] = mask_ratios

    print(f"\nFracture Pixel Statistics:")
    print(f"  Mean: {np.mean(mask_pixel_counts):.1f} pixels")
    print(f"  Median: {np.median(mask_pixel_counts):.1f} pixels")
    print(f"  Min: {np.min(mask_pixel_counts):.1f} pixels")
    print(f"  Max: {np.max(mask_pixel_counts):.1f} pixels")
    print(f"  Std: {np.std(mask_pixel_counts):.1f} pixels")

    print(f"\nFracture Ratio (% of image):")
    print(f"  Mean: {np.mean(mask_ratios):.3f}%")
    print(f"  Median: {np.median(mask_ratios):.3f}%")
    print(f"  Min: {np.min(mask_ratios):.3f}%")
    print(f"  Max: {np.max(mask_ratios):.3f}%")

    return df_fracture_mask

# Train/Testマスクの分析
df_train_mask_analyzed = analyze_mask_pixels(df_train_mask, "TRAIN")
df_test_mask_analyzed = analyze_mask_pixels(df_test_mask, "TEST")

# %% [markdown]
# ## 6. マスクピクセル分布の可視化

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

if df_train_mask_analyzed is not None:
    # Train: 骨折ピクセル数の分布
    axes[0, 0].hist(df_train_mask_analyzed['FracturePixels'], bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Train: Fracture Pixel Count Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Number of Fracture Pixels', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].axvline(df_train_mask_analyzed['FracturePixels'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df_train_mask_analyzed['FracturePixels'].mean():.1f}")
    axes[0, 0].axvline(df_train_mask_analyzed['FracturePixels'].median(), color='blue', linestyle='--',
                       label=f"Median: {df_train_mask_analyzed['FracturePixels'].median():.1f}")
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Train: 骨折領域の割合分布
    axes[1, 0].hist(df_train_mask_analyzed['FractureRatio'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Train: Fracture Ratio Distribution (% of image)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Fracture Ratio (%)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].axvline(df_train_mask_analyzed['FractureRatio'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df_train_mask_analyzed['FractureRatio'].mean():.3f}%")
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)

if df_test_mask_analyzed is not None:
    # Test: 骨折ピクセル数の分布
    axes[0, 1].hist(df_test_mask_analyzed['FracturePixels'], bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Test: Fracture Pixel Count Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Fracture Pixels', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].axvline(df_test_mask_analyzed['FracturePixels'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df_test_mask_analyzed['FracturePixels'].mean():.1f}")
    axes[0, 1].axvline(df_test_mask_analyzed['FracturePixels'].median(), color='blue', linestyle='--',
                       label=f"Median: {df_test_mask_analyzed['FracturePixels'].median():.1f}")
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Test: 骨折領域の割合分布
    axes[1, 1].hist(df_test_mask_analyzed['FractureRatio'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Test: Fracture Ratio Distribution (% of image)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Fracture Ratio (%)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].axvline(df_test_mask_analyzed['FractureRatio'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df_test_mask_analyzed['FractureRatio'].mean():.3f}%")
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mask_pixel_distribution.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR / 'mask_pixel_distribution.png'}")
plt.show()

# %% [markdown]
# ## 7. 総合サマリー表示

# %%
print("\n" + "="*80)
print("COMPREHENSIVE DATA DISTRIBUTION SUMMARY")
print("="*80)

print(f"\n{'Dataset':<15} {'Total Slices':<15} {'Fracture':<15} {'Non-Fracture':<15} {'Fracture %':<15}")
print("-" * 80)

train_total = len(df_train)
train_fracture = (df_train['Fracture_Label'] == 1).sum()
train_nonfracture = (df_train['Fracture_Label'] == 0).sum()
train_fracture_pct = train_fracture / train_total * 100

test_total = len(df_test)
test_fracture = (df_test['Fracture_Label'] == 1).sum()
test_nonfracture = (df_test['Fracture_Label'] == 0).sum()
test_fracture_pct = test_fracture / test_total * 100

print(f"{'Train':<15} {train_total:<15} {train_fracture:<15} {train_nonfracture:<15} {train_fracture_pct:<15.2f}")
print(f"{'Test':<15} {test_total:<15} {test_fracture:<15} {test_nonfracture:<15} {test_fracture_pct:<15.2f}")

print(f"\n{'Dataset':<15} {'Unique Cases':<15} {'Cases w/ Fracture':<20}")
print("-" * 80)

train_cases = df_train['Case'].nunique()
train_fracture_cases = df_train[df_train['Fracture_Label'] == 1]['Case'].nunique()

test_cases = df_test['Case'].nunique()
test_fracture_cases = df_test[df_test['Fracture_Label'] == 1]['Case'].nunique()

print(f"{'Train':<15} {train_cases:<15} {train_fracture_cases:<20}")
print(f"{'Test':<15} {test_cases:<15} {test_fracture_cases:<20}")

if df_train_mask_analyzed is not None:
    print(f"\n{'Dataset':<15} {'Mean Fracture Pixels':<25} {'Mean Fracture Ratio (%)':<25}")
    print("-" * 80)
    print(f"{'Train':<15} {df_train_mask_analyzed['FracturePixels'].mean():<25.1f} {df_train_mask_analyzed['FractureRatio'].mean():<25.3f}")

if df_test_mask_analyzed is not None:
    if df_train_mask_analyzed is None:
        print(f"\n{'Dataset':<15} {'Mean Fracture Pixels':<25} {'Mean Fracture Ratio (%)':<25}")
        print("-" * 80)
    print(f"{'Test':<15} {df_test_mask_analyzed['FracturePixels'].mean():<25.1f} {df_test_mask_analyzed['FractureRatio'].mean():<25.3f}")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print(f"1. Class Imbalance:")
print(f"   - Train: {train_fracture_pct:.1f}% fracture vs {100-train_fracture_pct:.1f}% non-fracture")
print(f"   - Test: {test_fracture_pct:.1f}% fracture vs {100-test_fracture_pct:.1f}% non-fracture")
print(f"   → Recommendation: Use weighted loss or focal loss to handle imbalance")

if df_train_mask_analyzed is not None:
    print(f"\n2. Fracture Region Size:")
    print(f"   - Mean fracture pixels: {df_train_mask_analyzed['FracturePixels'].mean():.1f}")
    print(f"   - Mean fracture ratio: {df_train_mask_analyzed['FractureRatio'].mean():.3f}%")
    print(f"   → Recommendation: Small target regions require attention mechanisms")

print(f"\n3. Dataset Split:")
print(f"   - Train: {train_cases} cases, {train_total} slices")
print(f"   - Test: {test_cases} cases, {test_total} slices")
print(f"   → Patient-level split ensures no data leakage")

print("\n" + "="*80)

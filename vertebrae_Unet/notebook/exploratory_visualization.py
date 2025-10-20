# %% [markdown]
# # 椎体画像の探索的可視化
#
# このノートブックでは、以下を確認します：
# 1. 画像サイズの分布（どのようにサイズを統一するか決定）
# 2. HU値の分布（正規化範囲の確認）
# 3. 骨折あり/なしの画像比較
# 4. 異なる椎体の画像サイズ比較
# 5. マスク画像の確認

# %%
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import re

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# %% [markdown]
# ## 1. データ読み込みとサマリー

# %%
# データパスの設定
DATA_DIR = project_root / "data"
SLICE_TRAIN_DIR = DATA_DIR / "slice_train" / "axial"
PROCESSED_TRAIN_DIR = DATA_DIR / "processed_train"

# 全CSVファイルを結合
all_csv_files = list(SLICE_TRAIN_DIR.glob("inp*/fracture_labels_inp*.csv"))
print(f"Found {len(all_csv_files)} CSV files")

df_list = []
for csv_file in all_csv_files:
    df = pd.read_csv(csv_file)
    df_list.append(df)

df_all = pd.concat(df_list, ignore_index=True)
print(f"\nTotal slices: {len(df_all)}")
print(f"Unique cases: {df_all['Case'].nunique()}")
print(f"Unique vertebrae: {sorted(df_all['Vertebra'].unique())}")
print(f"\nFracture distribution:")
print(df_all['Fracture_Label'].value_counts())

# %%
# 画像サイズの統計
print("\n=== Image Size Statistics ===")
print(f"Height (CT_H): min={df_all['CT_H'].min()}, max={df_all['CT_H'].max()}, mean={df_all['CT_H'].mean():.2f}")
print(f"Width (CT_W): min={df_all['CT_W'].min()}, max={df_all['CT_W'].max()}, mean={df_all['CT_W'].mean():.2f}")
print(f"Depth (CT_D): min={df_all['CT_D'].min()}, max={df_all['CT_D'].max()}, mean={df_all['CT_D'].mean():.2f}")

"""
各変数の意味
H (Height): 画像の高さ方向のピクセル数です。2次元で見たときの縦のサイズにあたります。(Y軸)

W (Width): 画像の幅方向のピクセル数です。2次元で見たときの横のサイズにあたります。(X軸)

D (Depth): 画像の奥行き、つまりスライスの枚数です。CT画像は薄い輪切りの画像を何枚も重ねて3Dデータにしているため、その輪切りの枚数がDにあたります。(Z軸)

CT_H, CT_W, CT_Dはそれぞれ、CT画像の3次元的なサイズを表しています。
"""

# %% [markdown]
# ## 2. 画像サイズの分布を可視化

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Height分布
axes[0, 0].hist(df_all['CT_H'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Height (CT_H) Distribution')
axes[0, 0].set_xlabel('Height (pixels)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(df_all['CT_H'].mean(), color='red', linestyle='--', label=f'Mean: {df_all["CT_H"].mean():.1f}')
axes[0, 0].axvline(df_all['CT_H'].max(), color='green', linestyle='--', label=f'Max: {df_all["CT_H"].max()}')
axes[0, 0].legend()

# Width分布
axes[0, 1].hist(df_all['CT_W'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Width (CT_W) Distribution')
axes[0, 1].set_xlabel('Width (pixels)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(df_all['CT_W'].mean(), color='red', linestyle='--', label=f'Mean: {df_all["CT_W"].mean():.1f}')
axes[0, 1].axvline(df_all['CT_W'].max(), color='green', linestyle='--', label=f'Max: {df_all["CT_W"].max()}')
axes[0, 1].legend()

# Depth分布
axes[1, 0].hist(df_all['CT_D'], bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Depth (CT_D) Distribution')
axes[1, 0].set_xlabel('Depth (slices)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(df_all['CT_D'].mean(), color='red', linestyle='--', label=f'Mean: {df_all["CT_D"].mean():.1f}')
axes[1, 0].legend()

# Height vs Width散布図
axes[1, 1].scatter(df_all['CT_W'], df_all['CT_H'], alpha=0.3)
axes[1, 1].set_title('Height vs Width')
axes[1, 1].set_xlabel('Width (pixels)')
axes[1, 1].set_ylabel('Height (pixels)')
axes[1, 1].axhline(df_all['CT_H'].max(), color='green', linestyle='--', alpha=0.5, label='Max H')
axes[1, 1].axvline(df_all['CT_W'].max(), color='blue', linestyle='--', alpha=0.5, label='Max W')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(project_root / "vertebrae_Unet" / "notebook" / "size_distribution.png", dpi=150)
print("Saved: size_distribution.png")
plt.show()

# %% [markdown]
# ## 3. 椎体ごとの画像サイズを確認

# %%
# 椎体ごとのサイズ統計
vertebra_stats = df_all.groupby('Vertebra').agg({
    'CT_H': ['min', 'max', 'mean'],
    'CT_W': ['min', 'max', 'mean'],
    'CT_D': ['min', 'max', 'mean'],
    'Fracture_Label': 'sum'
}).round(2)
vertebra_stats.columns = ['_'.join(col).strip() for col in vertebra_stats.columns.values]
vertebra_stats = vertebra_stats.rename(columns={'Fracture_Label_sum': 'Fracture_Count'})
print("\n=== Vertebra-wise Statistics ===")
print(vertebra_stats)

# %%
# 椎体ごとのサイズ可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

vertebrae_list = sorted(df_all['Vertebra'].unique())
df_unique = df_all.drop_duplicates(subset=['Case', 'Vertebra'])

# Height by vertebra
height_by_vertebra = [df_unique[df_unique['Vertebra'] == v]['CT_H'].values for v in vertebrae_list]
axes[0].boxplot(height_by_vertebra, labels=vertebrae_list)
axes[0].set_title('Height Distribution by Vertebra')
axes[0].set_xlabel('Vertebra')
axes[0].set_ylabel('Height (pixels)')
axes[0].grid(axis='y', alpha=0.3)

# Width by vertebra
width_by_vertebra = [df_unique[df_unique['Vertebra'] == v]['CT_W'].values for v in vertebrae_list]
axes[1].boxplot(width_by_vertebra, labels=vertebrae_list)
axes[1].set_title('Width Distribution by Vertebra')
axes[1].set_xlabel('Vertebra')
axes[1].set_ylabel('Width (pixels)')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(project_root / "vertebrae_Unet" / "notebook" / "size_by_vertebra.png", dpi=150)
print("Saved: size_by_vertebra.png")
plt.show()

# %% [markdown]
# ## 4. サンプル画像の読み込みと可視化

# %%
def load_nifti_slice(slice_path):
    """NIfTIスライス画像を読み込む"""
    img = nib.load(slice_path)
    data = img.get_fdata()
    return data

def load_nifti_mask(mask_path):
    """NIfTIマスク画像を読み込む"""
    img = nib.load(mask_path)
    data = img.get_fdata()
    return data

def get_mask_path_from_slice(slice_path):
    """スライス画像パスから対応するマスク画像パスを生成"""
    # /slice_train/axial/ -> /slice_train/axial_mask/
    mask_path = slice_path.replace('/slice_train/axial/', '/slice_train/axial_mask/')
    # slice_XXX.nii -> mask_XXX.nii
    import re
    mask_path = re.sub(r'/slice_(\d+)\.nii', r'/mask_\1.nii', mask_path)
    return mask_path

# 骨折ありのサンプルを取得
fracture_samples = df_all[df_all['Fracture_Label'] == 1].sample(min(5, len(df_all[df_all['Fracture_Label'] == 1])))
# 骨折なしのサンプルを取得
no_fracture_samples = df_all[df_all['Fracture_Label'] == 0].sample(5)

print("\n=== Fracture Samples ===")
print(fracture_samples[['Case', 'Vertebra', 'SliceIndex', 'CT_H', 'CT_W']])

# %%
# 骨折ありサンプルの可視化（CT + マスク + オーバーレイ）
fig, axes = plt.subplots(3, 5, figsize=(20, 12))

for idx, (_, row) in enumerate(fracture_samples.iterrows()):
    if idx >= 5:
        break

    # CT画像読み込み
    ct_slice = load_nifti_slice(row['FullPath'])

    # マスク画像パス生成と読み込み
    mask_path = get_mask_path_from_slice(row['FullPath'])
    mask_exists = os.path.exists(mask_path)

    if mask_exists:
        mask_slice = load_nifti_mask(mask_path)
    else:
        mask_slice = np.zeros_like(ct_slice)
        print(f"Warning: Mask not found at {mask_path}")

    # CT画像表示
    axes[0, idx].imshow(ct_slice.squeeze(), cmap='gray', vmin=0, vmax=1800)
    axes[0, idx].set_title(f"CT: Case {row['Case']}\nV{row['Vertebra']}, Slice {row['SliceIndex']}")
    axes[0, idx].axis('off')

    # マスク画像表示
    axes[1, idx].imshow(mask_slice.squeeze(), cmap='hot', vmin=0, vmax=1)
    axes[1, idx].set_title(f"Mask\nPixels: {int(mask_slice.sum())}")
    axes[1, idx].axis('off')

    # オーバーレイ表示
    axes[2, idx].imshow(ct_slice.squeeze(), cmap='gray', vmin=0, vmax=1800)
    axes[2, idx].imshow(mask_slice.squeeze(), cmap='hot', alpha=0.5, vmin=0, vmax=1)
    axes[2, idx].set_title(f"Overlay\nShape: {ct_slice.shape}")
    axes[2, idx].axis('off')

plt.tight_layout()
plt.savefig(project_root / "vertebrae_Unet" / "notebook" / "fracture_samples.png", dpi=150)
print("Saved: fracture_samples.png")
plt.show()

# %%
# 骨折なしサンプルの可視化（CT + マスク + オーバーレイ）
fig, axes = plt.subplots(3, 5, figsize=(20, 12))

for idx, (_, row) in enumerate(no_fracture_samples.iterrows()):
    if idx >= 5:
        break

    # CT画像読み込み
    ct_slice = load_nifti_slice(row['FullPath'])

    # マスク画像パス生成と読み込み
    mask_path = get_mask_path_from_slice(row['FullPath'])
    mask_exists = os.path.exists(mask_path)

    if mask_exists:
        mask_slice = load_nifti_mask(mask_path)
    else:
        mask_slice = np.zeros_like(ct_slice)
        print(f"Warning: Mask not found at {mask_path}")

    # CT画像表示
    axes[0, idx].imshow(ct_slice.squeeze(), cmap='gray', vmin=0, vmax=1800)
    axes[0, idx].set_title(f"CT: Case {row['Case']}\nV{row['Vertebra']}, Slice {row['SliceIndex']}")
    axes[0, idx].axis('off')

    # マスク画像表示
    axes[1, idx].imshow(mask_slice.squeeze(), cmap='hot', vmin=0, vmax=1)
    axes[1, idx].set_title(f"Mask\nPixels: {int(mask_slice.sum())}")
    axes[1, idx].axis('off')

    # オーバーレイ表示
    axes[2, idx].imshow(ct_slice.squeeze(), cmap='gray', vmin=0, vmax=1800)
    axes[2, idx].imshow(mask_slice.squeeze(), cmap='hot', alpha=0.5, vmin=0, vmax=1)
    axes[2, idx].set_title(f"Overlay\nShape: {ct_slice.shape}")
    axes[2, idx].axis('off')

plt.tight_layout()
plt.savefig(project_root / "vertebrae_Unet" / "notebook" / "no_fracture_samples.png", dpi=150)
print("Saved: no_fracture_samples.png")
plt.show()

# %% [markdown]
# ## 5. HU値の統計情報

# %%
# ランダムに100スライスサンプリングしてHU値を調査
sample_size = min(100, len(df_all))
sampled_slices = df_all.sample(sample_size)

hu_values = []
for _, row in sampled_slices.iterrows():
    try:
        ct_slice = load_nifti_slice(row['FullPath'])
        hu_values.extend(ct_slice.flatten())
    except Exception as e:
        print(f"Error loading {row['FullPath']}: {e}")
        continue

hu_values = np.array(hu_values)

print("\n=== HU Value Statistics (from 100 random slices) ===")
print(f"Min: {hu_values.min():.2f}")
print(f"Max: {hu_values.max():.2f}")
print(f"Mean: {hu_values.mean():.2f}")
print(f"Median: {np.median(hu_values):.2f}")
print(f"Std: {hu_values.std():.2f}")
print(f"5th percentile: {np.percentile(hu_values, 5):.2f}")
print(f"95th percentile: {np.percentile(hu_values, 95):.2f}")

# %%
# HU値の全体分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 全体ヒストグラム
axes[0].hist(hu_values, bins=100, edgecolor='black', alpha=0.7)
axes[0].set_title('Overall HU Value Distribution')
axes[0].set_xlabel('HU Value')
axes[0].set_ylabel('Frequency')
axes[0].axvline(0, color='red', linestyle='--', label='HU=0')
axes[0].axvline(1800, color='green', linestyle='--', label='HU=1800')
axes[0].legend()

# 0-1800範囲のヒストグラム
hu_clipped = np.clip(hu_values, 0, 1800)
axes[1].hist(hu_clipped, bins=100, edgecolor='black', alpha=0.7)
axes[1].set_title('HU Value Distribution (Clipped to 0-1800)')
axes[1].set_xlabel('HU Value')
axes[1].set_ylabel('Frequency')
axes[1].axvline(0, color='red', linestyle='--', label='HU=0')
axes[1].axvline(1800, color='green', linestyle='--', label='HU=1800')
axes[1].legend()

plt.tight_layout()
plt.savefig(project_root / "vertebrae_Unet" / "notebook" / "hu_distribution.png", dpi=150)
print("Saved: hu_distribution.png")
plt.show()

# %% [markdown]
# ## 6. マスク画像の詳細確認

# %%
# スライスマスクの詳細分析
print("\n=== Mask Analysis ===")

# 複数のサンプルでマスクを確認
sample_count = min(3, len(fracture_samples))
fig, axes = plt.subplots(sample_count, 3, figsize=(15, 5 * sample_count))

if sample_count == 1:
    axes = axes.reshape(1, -1)

for idx in range(sample_count):
    sample_row = fracture_samples.iloc[idx]

    # CT画像とマスク画像を読み込み
    ct_path = sample_row['FullPath']
    mask_path = get_mask_path_from_slice(ct_path)

    print(f"\nSample {idx + 1}:")
    print(f"  CT path: {ct_path}")
    print(f"  Mask path: {mask_path}")
    print(f"  Mask exists: {os.path.exists(mask_path)}")

    ct_slice = load_nifti_slice(ct_path)

    if os.path.exists(mask_path):
        mask_slice = load_nifti_mask(mask_path)
        print(f"  Mask shape: {mask_slice.shape}")
        print(f"  Mask unique values: {np.unique(mask_slice)}")
        print(f"  Fracture pixels: {int(mask_slice.sum())}")
        print(f"  Fracture ratio: {mask_slice.sum() / mask_slice.size * 100:.2f}%")
    else:
        mask_slice = np.zeros_like(ct_slice)
        print(f"  Warning: Mask not found!")

    # CT画像
    axes[idx, 0].imshow(ct_slice.squeeze(), cmap='gray', vmin=0, vmax=1800)
    axes[idx, 0].set_title(f'CT: Case {sample_row["Case"]}, V{sample_row["Vertebra"]}, Slice {sample_row["SliceIndex"]}')
    axes[idx, 0].axis('off')

    # マスク画像
    axes[idx, 1].imshow(mask_slice.squeeze(), cmap='hot', vmin=0, vmax=1)
    axes[idx, 1].set_title(f'Mask: {int(mask_slice.sum())} pixels')
    axes[idx, 1].axis('off')

    # オーバーレイ
    axes[idx, 2].imshow(ct_slice.squeeze(), cmap='gray', vmin=0, vmax=1800)
    axes[idx, 2].imshow(mask_slice.squeeze(), cmap='hot', alpha=0.5, vmin=0, vmax=1)
    axes[idx, 2].set_title('Overlay (CT + Mask)')
    axes[idx, 2].axis('off')

plt.tight_layout()
plt.savefig(project_root / "vertebrae_Unet" / "notebook" / "mask_detailed_examples.png", dpi=150)
print("\nSaved: mask_detailed_examples.png")
plt.show()

# %% [markdown]
# ## 7. サマリーと推奨事項

# %%
print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

max_h = df_all['CT_H'].max()
max_w = df_all['CT_W'].max()
mean_h = df_all['CT_H'].mean()
mean_w = df_all['CT_W'].mean()

print(f"\n1. IMAGE SIZE UNIFICATION:")
print(f"   - Maximum size: ({max_h}, {max_w})")
print(f"   - Average size: ({mean_h:.1f}, {mean_w:.1f})")
print(f"   - Recommendation: Zero-padding to ({max_h}, {max_w}) while preserving aspect ratio")

print(f"\n2. HU VALUE NORMALIZATION:")
print(f"   - Current range: [{hu_values.min():.0f}, {hu_values.max():.0f}]")
print(f"   - Target range for normalization: [0, 1800]")
print(f"   - Recommendation: Clip to [0, 1800] then normalize to [0, 1]")
print(f"   - Formula: normalized = clip(HU, 0, 1800) / 1800")

print(f"\n3. DATA DISTRIBUTION:")
print(f"   - Total slices: {len(df_all)}")
print(f"   - Fracture slices: {(df_all['Fracture_Label'] == 1).sum()} ({(df_all['Fracture_Label'] == 1).sum() / len(df_all) * 100:.2f}%)")
print(f"   - Non-fracture slices: {(df_all['Fracture_Label'] == 0).sum()} ({(df_all['Fracture_Label'] == 0).sum() / len(df_all) * 100:.2f}%)")
print(f"   - Recommendation: Consider weighted loss or focal loss due to class imbalance")

print(f"\n4. INPUT STRATEGY:")
print(f"   - Option A: Single slice input (B, 1, {max_h}, {max_w})")
print(f"   - Option B: Multi-slice sequence (B, N, {max_h}, {max_w}) for LSTM")
print(f"   - Recommendation: Start with single slice (Phase 1), then add LSTM (Phase 2)")

print("\n" + "="*80)

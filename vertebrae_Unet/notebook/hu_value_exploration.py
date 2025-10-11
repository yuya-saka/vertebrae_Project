# %% [markdown]
# # HU値ウィンドウイング探索的分析
#
# このノートブックでは、以下を確認します：
# 1. 異なるHU値範囲での画像の見え方
# 2. 最適なウィンドウレベル・ウィンドウ幅の決定
# 3. 骨折領域の視認性の比較
# 4. 正規化方法の検証

# %%
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 出力ディレクトリの作成
output_dir = Path(__file__).parent / "exploratory_image"
output_dir.mkdir(exist_ok=True)

# %% [markdown]
# ## 1. サンプルデータの読み込み

# %%
# データパスの設定
DATA_DIR = project_root / "vertebrae_Unet" / "data"
SLICE_TRAIN_DIR = DATA_DIR / "slice_train" / "axial"
SLICE_MASK_DIR = DATA_DIR / "slice_train" / "axial_mask"

# CSVファイルを読み込み
all_csv_files = list(SLICE_TRAIN_DIR.glob("inp*/fracture_labels_inp*.csv"))
print(f"Found {len(all_csv_files)} CSV files")

df_list = []
for csv_file in all_csv_files:
    df = pd.read_csv(csv_file)
    df_list.append(df)

df_all = pd.concat(df_list, ignore_index=True)
print(f"Total slices: {len(df_all)}")

# 骨折ありのサンプルを取得
fracture_samples = df_all[df_all['Fracture_Label'] == 1].sample(min(3, len(df_all[df_all['Fracture_Label'] == 1])), random_state=42)
print(f"\nSelected {len(fracture_samples)} fracture samples for analysis")
print(fracture_samples[['Case', 'Vertebra', 'SliceIndex']])

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
    import re
    mask_path = slice_path.replace('/slice_train/axial/', '/slice_train/axial_mask/')
    mask_path = re.sub(r'/slice_(\d+)\.nii', r'/mask_\1.nii', mask_path)
    return mask_path

# %% [markdown]
# ## 2. 様々なHU値範囲での可視化

# %%
# 異なるHU値ウィンドウ設定
hu_windows = [
    {"name": "Full Range", "min": -1000, "max": 3000, "description": "全HU値範囲"},
    {"name": "Bone Window", "min": -200, "max": 1800, "description": "骨組織強調（一般的）"},
    {"name": "Current Setting", "min": 0, "max": 1800, "description": "現在の設定"},
    {"name": "Soft Tissue", "min": -160, "max": 240, "description": "軟部組織"},
    {"name": "Lung Window", "min": -300, "max": 800, "description": "骨折が含まれる範囲"},
    {"name": "Narrow Bone", "min": 200, "max": 1200, "description": "骨組織狭域"},
]

# 1つのサンプルで全てのウィンドウ設定を比較
sample_row = fracture_samples.iloc[0]
ct_slice = load_nifti_slice(sample_row['FullPath'])
mask_path = get_mask_path_from_slice(sample_row['FullPath'])

if os.path.exists(mask_path):
    mask_slice = load_nifti_mask(mask_path)
else:
    mask_slice = np.zeros_like(ct_slice)
    print(f"Warning: Mask not found at {mask_path}")

print(f"\n=== HU Value Statistics for Sample ===")
print(f"Min HU: {ct_slice.min():.2f}")
print(f"Max HU: {ct_slice.max():.2f}")
print(f"Mean HU: {ct_slice.mean():.2f}")
print(f"Median HU: {np.median(ct_slice):.2f}")

# %%
# ウィンドウ設定を比較
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for idx, window in enumerate(hu_windows):
    # CT画像表示
    axes[idx].imshow(ct_slice.squeeze(), cmap='gray', vmin=window['min'], vmax=window['max'])
    axes[idx].imshow(mask_slice.squeeze(), cmap='hot', alpha=0.3, vmin=0, vmax=1)
    axes[idx].set_title(f"{window['name']}\n[{window['min']}, {window['max']}]\n{window['description']}")
    axes[idx].axis('off')

# HU値ヒストグラム
axes[6].hist(ct_slice.flatten(), bins=100, edgecolor='black', alpha=0.7)
axes[6].set_title('HU Value Distribution')
axes[6].set_xlabel('HU Value')
axes[6].set_ylabel('Frequency')
axes[6].axvline(0, color='red', linestyle='--', alpha=0.5, label='HU=0')
axes[6].axvline(1800, color='green', linestyle='--', alpha=0.5, label='HU=1800')
axes[6].legend()
axes[6].grid(alpha=0.3)

# 骨折領域のHU値分布
if mask_slice.sum() > 0:
    fracture_hu = ct_slice[mask_slice > 0]
    axes[7].hist(fracture_hu.flatten(), bins=50, edgecolor='black', alpha=0.7, color='red')
    axes[7].set_title(f'Fracture Region HU Distribution\n(n={len(fracture_hu)} pixels)')
    axes[7].set_xlabel('HU Value')
    axes[7].set_ylabel('Frequency')
    axes[7].grid(alpha=0.3)

    print(f"\n=== Fracture Region HU Statistics ===")
    print(f"Min HU: {fracture_hu.min():.2f}")
    print(f"Max HU: {fracture_hu.max():.2f}")
    print(f"Mean HU: {fracture_hu.mean():.2f}")
    print(f"Median HU: {np.median(fracture_hu):.2f}")
else:
    axes[7].text(0.5, 0.5, 'No fracture pixels', ha='center', va='center', transform=axes[7].transAxes)
    axes[7].set_title('Fracture Region HU Distribution')

# 残りのサブプロットを非表示
axes[8].axis('off')

plt.tight_layout()
plt.savefig(output_dir / "hu_window_comparison.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: {output_dir / 'hu_window_comparison.png'}")
plt.show()

# %% [markdown]
# ## 3. 複数サンプルでの比較

# %%
# 3つの推奨ウィンドウ設定で複数サンプルを比較
recommended_windows = [
    {"name": "Bone [0, 1800]", "min": 0, "max": 1800},
    {"name": "Bone [-200, 1800]", "min": -200, "max": 1800},
    {"name": "Narrow [200, 1200]", "min": 200, "max": 1200},
]

n_samples = len(fracture_samples)
fig, axes = plt.subplots(n_samples, len(recommended_windows), figsize=(15, 5 * n_samples))

if n_samples == 1:
    axes = axes.reshape(1, -1)

for sample_idx, (_, sample_row) in enumerate(fracture_samples.iterrows()):
    ct_slice = load_nifti_slice(sample_row['FullPath'])
    mask_path = get_mask_path_from_slice(sample_row['FullPath'])

    if os.path.exists(mask_path):
        mask_slice = load_nifti_mask(mask_path)
    else:
        mask_slice = np.zeros_like(ct_slice)

    for window_idx, window in enumerate(recommended_windows):
        axes[sample_idx, window_idx].imshow(ct_slice.squeeze(), cmap='gray', vmin=window['min'], vmax=window['max'])
        axes[sample_idx, window_idx].imshow(mask_slice.squeeze(), cmap='hot', alpha=0.4, vmin=0, vmax=1)

        if sample_idx == 0:
            axes[sample_idx, window_idx].set_title(f"{window['name']}")

        # 左側に症例情報を表示
        if window_idx == 0:
            axes[sample_idx, window_idx].set_ylabel(
                f"Case {sample_row['Case']}\nV{sample_row['Vertebra']}, Slice {sample_row['SliceIndex']}",
                fontsize=10
            )

        axes[sample_idx, window_idx].axis('off')

plt.tight_layout()
plt.savefig(output_dir / "multi_sample_window_comparison.png", dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'multi_sample_window_comparison.png'}")
plt.show()

# %% [markdown]
# ## 4. 正規化方法の検証

# %%
# 異なる正規化方法を比較
sample_row = fracture_samples.iloc[0]
ct_slice = load_nifti_slice(sample_row['FullPath'])
mask_path = get_mask_path_from_slice(sample_row['FullPath'])

if os.path.exists(mask_path):
    mask_slice = load_nifti_mask(mask_path)
else:
    mask_slice = np.zeros_like(ct_slice)

# 正規化方法
normalization_methods = [
    {
        "name": "Original",
        "data": ct_slice,
        "description": "オリジナル（正規化なし）"
    },
    {
        "name": "Clip [0, 1800] → [0, 1]",
        "data": np.clip(ct_slice, 0, 1800) / 1800,
        "description": "HU値を0-1800でクリップ後、0-1に正規化"
    },
    {
        "name": "Clip [-200, 1800] → [0, 1]",
        "data": np.clip(ct_slice, -200, 1800) / 2000,
        "description": "HU値を-200-1800でクリップ後、0-1に正規化"
    },
    {
        "name": "Min-Max [0, 1]",
        "data": (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min()),
        "description": "各画像のmin-maxで0-1に正規化"
    },
    {
        "name": "Z-score",
        "data": (ct_slice - ct_slice.mean()) / ct_slice.std(),
        "description": "平均0、標準偏差1に標準化"
    },
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, method in enumerate(normalization_methods):
    if idx < len(axes):
        # 正規化後の画像を表示
        axes[idx].imshow(method['data'].squeeze(), cmap='gray')
        axes[idx].imshow(mask_slice.squeeze(), cmap='hot', alpha=0.3, vmin=0, vmax=1)
        axes[idx].set_title(f"{method['name']}\n{method['description']}\nRange: [{method['data'].min():.2f}, {method['data'].max():.2f}]")
        axes[idx].axis('off')

# 最後のサブプロットに統計情報を表示
axes[5].axis('off')
stats_text = "Normalization Statistics:\n\n"
for method in normalization_methods:
    stats_text += f"{method['name']}:\n"
    stats_text += f"  Min: {method['data'].min():.4f}\n"
    stats_text += f"  Max: {method['data'].max():.4f}\n"
    stats_text += f"  Mean: {method['data'].mean():.4f}\n"
    stats_text += f"  Std: {method['data'].std():.4f}\n\n"

axes[5].text(0.1, 0.9, stats_text, transform=axes[5].transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig(output_dir / "normalization_comparison.png", dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'normalization_comparison.png'}")
plt.show()

# %% [markdown]
# ## 5. 骨折領域と非骨折領域のHU値比較

# %%
# 全骨折サンプルからHU値を収集
fracture_hu_all = []
non_fracture_hu_all = []

print("\n=== Collecting HU values from all fracture samples ===")
for _, sample_row in fracture_samples.iterrows():
    ct_slice = load_nifti_slice(sample_row['FullPath'])
    mask_path = get_mask_path_from_slice(sample_row['FullPath'])

    if os.path.exists(mask_path):
        mask_slice = load_nifti_mask(mask_path)

        if mask_slice.sum() > 0:
            # 骨折領域のHU値
            fracture_hu = ct_slice[mask_slice > 0].flatten()
            fracture_hu_all.extend(fracture_hu)

            # 非骨折領域のHU値
            non_fracture_hu = ct_slice[mask_slice == 0].flatten()
            non_fracture_hu_all.extend(non_fracture_hu)

fracture_hu_all = np.array(fracture_hu_all)
non_fracture_hu_all = np.array(non_fracture_hu_all)

print(f"Fracture pixels: {len(fracture_hu_all)}")
print(f"Non-fracture pixels: {len(non_fracture_hu_all)}")

# %%
# HU値分布の比較
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 全体分布
axes[0, 0].hist(fracture_hu_all, bins=100, alpha=0.7, label='Fracture', color='red', edgecolor='black')
axes[0, 0].hist(non_fracture_hu_all, bins=100, alpha=0.5, label='Non-fracture', color='blue', edgecolor='black')
axes[0, 0].set_title('HU Distribution: Fracture vs Non-fracture (Full Range)')
axes[0, 0].set_xlabel('HU Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 0-1800範囲
axes[0, 1].hist(np.clip(fracture_hu_all, 0, 1800), bins=100, alpha=0.7, label='Fracture', color='red', edgecolor='black')
axes[0, 1].hist(np.clip(non_fracture_hu_all, 0, 1800), bins=100, alpha=0.5, label='Non-fracture', color='blue', edgecolor='black')
axes[0, 1].set_title('HU Distribution: Fracture vs Non-fracture [0, 1800]')
axes[0, 1].set_xlabel('HU Value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Box plot
data_to_plot = [fracture_hu_all, non_fracture_hu_all]
axes[1, 0].boxplot(data_to_plot, labels=['Fracture', 'Non-fracture'])
axes[1, 0].set_title('HU Value Distribution (Box Plot)')
axes[1, 0].set_ylabel('HU Value')
axes[1, 0].grid(alpha=0.3)

# 統計情報テーブル
stats_data = {
    'Metric': ['Count', 'Min', 'Max', 'Mean', 'Median', 'Std', 'Q1 (25%)', 'Q3 (75%)'],
    'Fracture': [
        len(fracture_hu_all),
        f"{fracture_hu_all.min():.2f}",
        f"{fracture_hu_all.max():.2f}",
        f"{fracture_hu_all.mean():.2f}",
        f"{np.median(fracture_hu_all):.2f}",
        f"{fracture_hu_all.std():.2f}",
        f"{np.percentile(fracture_hu_all, 25):.2f}",
        f"{np.percentile(fracture_hu_all, 75):.2f}",
    ],
    'Non-fracture': [
        len(non_fracture_hu_all),
        f"{non_fracture_hu_all.min():.2f}",
        f"{non_fracture_hu_all.max():.2f}",
        f"{non_fracture_hu_all.mean():.2f}",
        f"{np.median(non_fracture_hu_all):.2f}",
        f"{non_fracture_hu_all.std():.2f}",
        f"{np.percentile(non_fracture_hu_all, 25):.2f}",
        f"{np.percentile(non_fracture_hu_all, 75):.2f}",
    ]
}

axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=[[stats_data['Metric'][i], stats_data['Fracture'][i], stats_data['Non-fracture'][i]]
                                    for i in range(len(stats_data['Metric']))],
                          colLabels=['Metric', 'Fracture', 'Non-fracture'],
                          cellLoc='center',
                          loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
axes[1, 1].set_title('Statistical Summary', pad=20)

plt.tight_layout()
plt.savefig(output_dir / "fracture_vs_nonfracture_hu.png", dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'fracture_vs_nonfracture_hu.png'}")
plt.show()

# %% [markdown]
# ## 6. 推奨事項のサマリー

# %%
print("\n" + "="*80)
print("HU VALUE EXPLORATION SUMMARY & RECOMMENDATIONS")
print("="*80)

print("\n1. HU WINDOW SETTINGS:")
print("   Based on visual inspection and fracture visibility:")
print("   - Recommended: [0, 1800] or [-200, 1800]")
print("   - Reasoning: Best balance between bone structure and fracture visibility")
print("   - Alternative: [200, 1200] for focusing on dense bone structures")

print("\n2. NORMALIZATION METHOD:")
print("   Recommended: Clip [0, 1800] → [0, 1]")
print("   - Formula: normalized = np.clip(HU, 0, 1800) / 1800")
print("   - Pros: Consistent across all images, preserves relative HU relationships")
print("   - Cons: May lose some information outside [0, 1800] range")

print("\n3. FRACTURE REGION HU CHARACTERISTICS:")
if len(fracture_hu_all) > 0:
    print(f"   - Mean HU: {fracture_hu_all.mean():.2f}")
    print(f"   - Median HU: {np.median(fracture_hu_all):.2f}")
    print(f"   - Range: [{fracture_hu_all.min():.2f}, {fracture_hu_all.max():.2f}]")
    print(f"   - 95% of values in: [{np.percentile(fracture_hu_all, 2.5):.2f}, {np.percentile(fracture_hu_all, 97.5):.2f}]")

print("\n4. DATA PREPROCESSING PIPELINE:")
print("   Step 1: Load NIfTI image")
print("   Step 2: Clip HU values to [0, 1800]")
print("   Step 3: Normalize to [0, 1]: img_norm = np.clip(img, 0, 1800) / 1800")
print("   Step 4: Apply zero-padding to uniform size while preserving aspect ratio")
print("   Step 5: Apply data augmentation (rotation, scaling, brightness)")

print("\n5. NEXT STEPS:")
print("   ✓ HU value exploration completed")
print("   → Implement dataset class with decided normalization")
print("   → Implement zero-padding for size unification")
print("   → Start model implementation (Attention U-Net)")

print("\n" + "="*80)

# %%
# 推奨設定での最終確認
print("\n=== Final Visualization with Recommended Settings ===")

fig, axes = plt.subplots(len(fracture_samples), 3, figsize=(15, 5 * len(fracture_samples)))

if len(fracture_samples) == 1:
    axes = axes.reshape(1, -1)

for sample_idx, (_, sample_row) in enumerate(fracture_samples.iterrows()):
    ct_slice = load_nifti_slice(sample_row['FullPath'])
    mask_path = get_mask_path_from_slice(sample_row['FullPath'])

    if os.path.exists(mask_path):
        mask_slice = load_nifti_mask(mask_path)
    else:
        mask_slice = np.zeros_like(ct_slice)

    # オリジナル
    axes[sample_idx, 0].imshow(ct_slice.squeeze(), cmap='gray', vmin=0, vmax=1800)
    axes[sample_idx, 0].set_title(f"Original [0, 1800]\nCase {sample_row['Case']}, V{sample_row['Vertebra']}")
    axes[sample_idx, 0].axis('off')

    # 正規化後
    normalized = np.clip(ct_slice, 0, 1800) / 1800
    axes[sample_idx, 1].imshow(normalized.squeeze(), cmap='gray', vmin=0, vmax=1)
    axes[sample_idx, 1].set_title('Normalized [0, 1]')
    axes[sample_idx, 1].axis('off')

    # オーバーレイ
    axes[sample_idx, 2].imshow(normalized.squeeze(), cmap='gray', vmin=0, vmax=1)
    axes[sample_idx, 2].imshow(mask_slice.squeeze(), cmap='hot', alpha=0.5, vmin=0, vmax=1)
    axes[sample_idx, 2].set_title('Overlay with Mask')
    axes[sample_idx, 2].axis('off')

plt.tight_layout()
plt.savefig(output_dir / "final_recommended_settings.png", dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'final_recommended_settings.png'}")
plt.show()

print("\n✓ All analyses completed!")
print(f"✓ Images saved to: {output_dir}")

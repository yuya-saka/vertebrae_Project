"""
生データの検証スクリプト

目的:
1. NIfTIファイルの実際のHU値を確認
2. マスクと画像のアライメントを確認
3. 正規化前後の値を比較
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def main():
    """メイン検証関数"""
    print("="*80)
    print("生データ検証スクリプト")
    print("="*80)

    # CSVから骨折サンプルを取得
    csv_path = project_root / "data/slice_train/axial_mask/inp1003/mask_labels_inp1003.csv"
    df = pd.read_csv(csv_path)

    # 骨折ありのサンプルを取得
    fracture_df = df[df['Fracture_Label'] == 1]
    print(f"\n骨折サンプル数: {len(fracture_df)}/{len(df)}")

    # 複数のサンプルを確認
    num_samples = min(5, len(fracture_df))
    print(f"検証サンプル数: {num_samples}")

    fig, axes = plt.subplots(num_samples, 6, figsize=(24, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx in range(num_samples):
        row = fracture_df.iloc[idx]
        print(f"\n[サンプル {idx+1}]")
        print(f"  Case: {row['Case']}, Vertebra: {row['Vertebra']}, Slice: {row['SliceIndex']}")
        print(f"  マスクパス: {row['MaskPath']}")

        # マスク読み込み
        mask_path = Path(row['MaskPath'])
        mask_nii = nib.load(str(mask_path))
        mask_data = mask_nii.get_fdata()
        if mask_data.ndim == 3:
            mask_data = mask_data[:, :, 0]

        print(f"  マスクshape: {mask_data.shape}")
        print(f"  マスク陽性ピクセル: {(mask_data > 0).sum()}")
        print(f"  マスク値範囲: [{mask_data.min()}, {mask_data.max()}]")

        # 対応する画像を探す
        # MaskPathから画像パスを推測
        # 例: .../axial_mask/inp1003/27/mask_000.nii -> .../axial/inp1003/27/slice_000.nii
        parts = mask_path.parts
        case_id = f"inp{row['Case']}"
        vertebra = str(row['Vertebra'])
        slice_idx = f"slice_{row['SliceIndex']:03d}.nii"

        image_path = project_root / "data/slice_train/axial" / case_id / vertebra / slice_idx
        print(f"  画像パス: {image_path}")
        print(f"  画像存在: {image_path.exists()}")

        if not image_path.exists():
            continue

        # 画像読み込み
        image_nii = nib.load(str(image_path))
        image_data = image_nii.get_fdata()
        if image_data.ndim == 3:
            image_data = image_data[:, :, 0]

        print(f"  画像shape: {image_data.shape}")
        print(f"  画像HU値範囲: [{image_data.min():.1f}, {image_data.max():.1f}]")
        print(f"  画像HU値平均±std: {image_data.mean():.1f}±{image_data.std():.1f}")

        # HU値の分布を確認
        bone_region = image_data[mask_data > 0] if (mask_data > 0).sum() > 0 else np.array([])
        if len(bone_region) > 0:
            print(f"  骨折領域HU値: [{bone_region.min():.1f}, {bone_region.max():.1f}], "
                  f"平均{bone_region.mean():.1f}±{bone_region.std():.1f}")

        # 正規化処理
        def normalize_hu(img, hu_min, hu_max):
            img_clipped = np.clip(img, hu_min, hu_max)
            return (img_clipped - hu_min) / (hu_max - hu_min)

        ch1 = normalize_hu(image_data.copy(), 0, 1800)
        ch2 = normalize_hu(image_data.copy(), -200, 300)
        ch3 = normalize_hu(image_data.copy(), 200, 1200)

        # 可視化
        # 原画像
        axes[idx, 0].imshow(image_data, cmap='gray', vmin=-200, vmax=300)
        axes[idx, 0].set_title(f'Original HU\n[{image_data.min():.0f}, {image_data.max():.0f}]')
        axes[idx, 0].axis('off')

        # Ch1
        axes[idx, 1].imshow(ch1, cmap='gray', vmin=0, vmax=1)
        axes[idx, 1].set_title(f'Ch1 [0,1800]HU')
        axes[idx, 1].axis('off')

        # Ch2
        axes[idx, 2].imshow(ch2, cmap='gray', vmin=0, vmax=1)
        axes[idx, 2].set_title(f'Ch2 [-200,300]HU')
        axes[idx, 2].axis('off')

        # Ch3
        axes[idx, 3].imshow(ch3, cmap='gray', vmin=0, vmax=1)
        axes[idx, 3].set_title(f'Ch3 [200,1200]HU')
        axes[idx, 3].axis('off')

        # マスク
        axes[idx, 4].imshow(mask_data, cmap='hot', vmin=0, vmax=1)
        axes[idx, 4].set_title(f'Mask\nPixels={(mask_data>0).sum()}')
        axes[idx, 4].axis('off')

        # オーバーレイ
        axes[idx, 5].imshow(ch2, cmap='gray', vmin=0, vmax=1)
        if (mask_data > 0).sum() > 0:
            mask_overlay = np.ma.masked_where(mask_data == 0, mask_data)
            axes[idx, 5].imshow(mask_overlay, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
        axes[idx, 5].set_title(f'Overlay\nCase{row["Case"]} V{row["Vertebra"]} S{row["SliceIndex"]}')
        axes[idx, 5].axis('off')

    plt.tight_layout()

    # 保存
    output_dir = project_root / "notebook" / "dataloader_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "raw_data_verification.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n保存: {output_path}")
    plt.close()

    # HU値のヒストグラム
    print("\n" + "="*80)
    print("HU値分布の分析")
    print("="*80)
    analyze_hu_distribution(df)


def analyze_hu_distribution(df, num_samples=100):
    """HU値の分布を分析"""

    all_hu_values = []
    fracture_hu_values = []
    non_fracture_hu_values = []

    # ランダムにサンプリング
    sampled_df = df.sample(n=min(num_samples, len(df)), random_state=42)

    for idx, row in sampled_df.iterrows():
        case_id = f"inp{row['Case']}"
        vertebra = str(row['Vertebra'])
        slice_idx = f"slice_{row['SliceIndex']:03d}.nii"

        image_path = project_root / "data/slice_train/axial" / case_id / vertebra / slice_idx

        if not image_path.exists():
            continue

        # 画像読み込み
        image_nii = nib.load(str(image_path))
        image_data = image_nii.get_fdata()
        if image_data.ndim == 3:
            image_data = image_data[:, :, 0]

        # マスク読み込み
        mask_path = Path(row['MaskPath'])
        mask_nii = nib.load(str(mask_path))
        mask_data = mask_nii.get_fdata()
        if mask_data.ndim == 3:
            mask_data = mask_data[:, :, 0]

        all_hu_values.extend(image_data.flatten())

        if (mask_data > 0).sum() > 0:
            fracture_hu_values.extend(image_data[mask_data > 0].flatten())

        if (mask_data == 0).sum() > 0:
            # サンプリングして追加（メモリ節約）
            non_fracture_pixels = image_data[mask_data == 0].flatten()
            sample_size = min(1000, len(non_fracture_pixels))
            non_fracture_hu_values.extend(
                np.random.choice(non_fracture_pixels, sample_size, replace=False)
            )

    all_hu_values = np.array(all_hu_values)
    fracture_hu_values = np.array(fracture_hu_values)
    non_fracture_hu_values = np.array(non_fracture_hu_values)

    print(f"\n全体HU値統計 (N={len(all_hu_values):,}):")
    print(f"  範囲: [{all_hu_values.min():.1f}, {all_hu_values.max():.1f}]")
    print(f"  平均±std: {all_hu_values.mean():.1f}±{all_hu_values.std():.1f}")
    print(f"  パーセンタイル [5%, 25%, 50%, 75%, 95%]:")
    print(f"    {np.percentile(all_hu_values, [5, 25, 50, 75, 95])}")

    if len(fracture_hu_values) > 0:
        print(f"\n骨折領域HU値統計 (N={len(fracture_hu_values):,}):")
        print(f"  範囲: [{fracture_hu_values.min():.1f}, {fracture_hu_values.max():.1f}]")
        print(f"  平均±std: {fracture_hu_values.mean():.1f}±{fracture_hu_values.std():.1f}")
        print(f"  パーセンタイル [5%, 25%, 50%, 75%, 95%]:")
        print(f"    {np.percentile(fracture_hu_values, [5, 25, 50, 75, 95])}")

    if len(non_fracture_hu_values) > 0:
        print(f"\n非骨折領域HU値統計 (N={len(non_fracture_hu_values):,}):")
        print(f"  範囲: [{non_fracture_hu_values.min():.1f}, {non_fracture_hu_values.max():.1f}]")
        print(f"  平均±std: {non_fracture_hu_values.mean():.1f}±{non_fracture_hu_values.std():.1f}")
        print(f"  パーセンタイル [5%, 25%, 50%, 75%, 95%]:")
        print(f"    {np.percentile(non_fracture_hu_values, [5, 25, 50, 75, 95])}")

    # ヒストグラムを作成
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(all_hu_values, bins=100, alpha=0.7, edgecolor='black')
    axes[0].set_title('全体HU値分布')
    axes[0].set_xlabel('HU値')
    axes[0].set_ylabel('頻度')
    axes[0].axvline(0, color='r', linestyle='--', label='HU=0')
    axes[0].axvline(200, color='g', linestyle='--', label='HU=200 (骨)')
    axes[0].axvline(1800, color='b', linestyle='--', label='HU=1800 (高密度骨)')
    axes[0].legend()
    axes[0].set_xlim(-500, 2000)

    if len(fracture_hu_values) > 0:
        axes[1].hist(fracture_hu_values, bins=100, alpha=0.7, color='red', edgecolor='black')
        axes[1].set_title('骨折領域HU値分布')
        axes[1].set_xlabel('HU値')
        axes[1].set_ylabel('頻度')
        axes[1].set_xlim(-500, 2000)

    if len(non_fracture_hu_values) > 0:
        axes[2].hist(non_fracture_hu_values, bins=100, alpha=0.7, color='blue', edgecolor='black')
        axes[2].set_title('非骨折領域HU値分布')
        axes[2].set_xlabel('HU値')
        axes[2].set_ylabel('頻度')
        axes[2].set_xlim(-500, 2000)

    plt.tight_layout()

    output_dir = project_root / "notebook" / "dataloader_verification"
    output_path = output_dir / "hu_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nヒストグラム保存: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()

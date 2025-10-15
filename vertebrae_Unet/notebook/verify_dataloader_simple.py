"""
データローダーの簡易検証スクリプト

目的:
1. 画像とマスクの対応が正しいか確認
2. HU窓の正規化が適切か確認
3. 骨折領域が正しく読み込まれているか確認
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.datamodule.dataloader import VertebralFractureDataModule


def main():
    """メイン検証関数"""
    print("="*80)
    print("データローダー検証スクリプト")
    print("="*80)

    # 設定を直接定義
    data_dir = str(project_root / "data")
    image_size = (256, 256)
    batch_size = 8

    hu_windows = {
        'channel_1': {'min': 0, 'max': 1800},
        'channel_2': {'min': -200, 'max': 300},
        'channel_3': {'min': 200, 'max': 1200},
    }

    augmentation = {
        'rotation_degrees': 15,
        'translation_pixels': 10,
        'scale_range': [0.95, 1.05],
        'horizontal_flip_prob': 0.5,
        'brightness_hu': 50,
        'contrast_range': [0.95, 1.05],
        'gaussian_noise_std': 0.01,
    }

    print(f"\n設定:")
    print(f"  データディレクトリ: {data_dir}")
    print(f"  画像サイズ: {image_size[0]}x{image_size[1]}")
    print(f"  バッチサイズ: {batch_size}")

    # CSVファイル確認
    print("\n" + "="*80)
    print("CSVファイル確認")
    print("="*80)
    csv_pattern = f"{data_dir}/slice_train/axial_mask/*/mask_labels_*.csv"
    csv_files = sorted(glob.glob(csv_pattern))
    print(f"見つかったCSVファイル数: {len(csv_files)}")
    if csv_files:
        print(f"例: {csv_files[0]}")
        df = pd.read_csv(csv_files[0])
        print(f"カラム: {df.columns.tolist()}")
        print(f"骨折スライス: {(df['Fracture_Label']==1).sum()}/{len(df)}")

    # DataModule 初期化
    print("\n" + "="*80)
    print("DataModule 初期化")
    print("="*80)
    datamodule = VertebralFractureDataModule(
        data_dir=data_dir,
        hu_windows=hu_windows,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=0,
        n_folds=5,
        fold_id=0,
        augmentation=augmentation,
        oversample_fracture=True,
        oversample_factor=9,
    )

    datamodule.setup(stage="fit")

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    print(f"訓練データセット: {len(datamodule.train_dataset)} サンプル")
    print(f"検証データセット: {len(datamodule.val_dataset)} サンプル")

    # 訓練データの検証
    print("\n" + "="*80)
    print("訓練データ検証（データ拡張あり）")
    print("="*80)
    check_batches(train_loader, num_batches=2, prefix="train")

    # 検証データの検証
    print("\n" + "="*80)
    print("検証データ検証（データ拡張なし）")
    print("="*80)
    check_batches(val_loader, num_batches=2, prefix="val")

    print("\n" + "="*80)
    print("検証完了！")
    print("="*80)


def check_batches(dataloader, num_batches=2, prefix="data"):
    """バッチデータを検証"""
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        images = batch['image']  # (B, C, H, W)
        masks = batch['mask']    # (B, 1, H, W)
        labels = batch['label']
        metadata = batch['metadata']

        print(f"\n[バッチ {batch_idx+1}]")
        print(f"  画像shape: {images.shape}")
        print(f"  マスクshape: {masks.shape}")
        print(f"  ラベル: {labels.tolist()}")
        print(f"  画像範囲: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  画像平均±std: {images.mean():.3f}±{images.std():.3f}")
        print(f"  マスク陽性ピクセル: {masks.sum().item():.0f} ({masks.sum()/masks.numel()*100:.4f}%)")

        # チャンネルごとの統計
        for ch in range(3):
            ch_data = images[:, ch, :, :]
            print(f"  Ch{ch+1}: min={ch_data.min():.3f}, max={ch_data.max():.3f}, "
                  f"mean={ch_data.mean():.3f}, std={ch_data.std():.3f}")

        # 可視化
        visualize_batch(images, masks, labels, metadata, batch_idx, prefix)


def visualize_batch(images, masks, labels, metadata, batch_idx, prefix):
    """バッチを可視化"""

    batch_size = images.shape[0]

    # 骨折サンプルを優先して選択
    fracture_indices = [i for i, label in enumerate(labels) if label == 1]
    non_fracture_indices = [i for i, label in enumerate(labels) if label == 0]

    # 最大4サンプルを選択
    selected_indices = []
    if fracture_indices:
        selected_indices.extend(fracture_indices[:2])
    remaining = min(4, batch_size) - len(selected_indices)
    if remaining > 0 and non_fracture_indices:
        selected_indices.extend(non_fracture_indices[:remaining])

    if not selected_indices:
        return

    num_samples = len(selected_indices)
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for row_idx, sample_idx in enumerate(selected_indices):
        img = images[sample_idx].cpu().numpy()  # (3, H, W)
        mask = masks[sample_idx, 0].cpu().numpy()  # (H, W)
        label = labels[sample_idx].item()

        case_id = metadata['case'][sample_idx]
        vertebra = metadata['vertebra'][sample_idx]
        slice_idx = metadata['slice_index'][sample_idx]

        # Ch1
        axes[row_idx, 0].imshow(img[0], cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 0].set_title(f'Ch1 [0,1800]HU\nCase{case_id} {vertebra} slice{slice_idx}')
        axes[row_idx, 0].axis('off')

        # Ch2
        axes[row_idx, 1].imshow(img[1], cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 1].set_title(f'Ch2 [-200,300]HU')
        axes[row_idx, 1].axis('off')

        # Ch3
        axes[row_idx, 2].imshow(img[2], cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 2].set_title(f'Ch3 [200,1200]HU')
        axes[row_idx, 2].axis('off')

        # Mask
        axes[row_idx, 3].imshow(mask, cmap='hot', vmin=0, vmax=1)
        axes[row_idx, 3].set_title(f'Mask\nPixels={mask.sum():.0f}')
        axes[row_idx, 3].axis('off')

        # Overlay
        axes[row_idx, 4].imshow(img[1], cmap='gray', vmin=0, vmax=1)
        if mask.sum() > 0:
            mask_overlay = np.ma.masked_where(mask == 0, mask)
            axes[row_idx, 4].imshow(mask_overlay, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
            axes[row_idx, 4].set_title(f'Overlay\nLabel={label}')
        else:
            axes[row_idx, 4].set_title(f'Overlay\nLabel={label} (No fracture)')
        axes[row_idx, 4].axis('off')

    plt.tight_layout()

    # 保存
    output_dir = project_root / "notebook" / "dataloader_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prefix}_batch{batch_idx:02d}.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"  保存: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()

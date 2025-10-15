"""
データローダーの検証スクリプト

目的:
1. 画像とマスクの対応が正しいか確認
2. HU窓の正規化が適切か確認
3. データ拡張の動作確認
4. 骨折領域が正しく読み込まれているか確認
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.datamodule.dataloader import VertebralFractureDataModule
from omegaconf import OmegaConf


def load_config():
    """設定ファイルを読み込む"""
    config_path = project_root / "run/conf/config.yaml"
    cfg = OmegaConf.load(config_path)

    # defaults を手動で読み込む
    constants_path = project_root / "run/conf/constants.yaml"
    train_path = project_root / "run/conf/train.yaml"
    dir_path = project_root / "run/conf/dir/local.yaml"
    model_path = project_root / "run/conf/model/attention_unet.yaml"
    split_path = project_root / "run/conf/split/fold_0.yaml"

    cfg_constants = OmegaConf.load(constants_path)
    cfg_train = OmegaConf.load(train_path)
    cfg_dir_raw = OmegaConf.load(dir_path)
    cfg_model = OmegaConf.load(model_path)
    cfg_split = OmegaConf.load(split_path)

    # dirの下に配置
    cfg_dir = OmegaConf.create({"dir": cfg_dir_raw})

    # マージ
    cfg = OmegaConf.merge(cfg_constants, cfg_train, cfg_dir, cfg_model, cfg_split, cfg)

    return cfg


def verify_dataloader(num_batches=3, num_samples_per_batch=4):
    """データローダーを検証する"""

    print("="*80)
    print("データローダー検証スクリプト")
    print("="*80)

    # 設定読み込み
    print("\n1. 設定ファイル読み込み中...")
    cfg = load_config()

    print(f"   データディレクトリ: {cfg.dir.data.base}")
    print(f"   画像サイズ: {cfg.image_size.height}x{cfg.image_size.width}")
    print(f"   バッチサイズ: {cfg.training.batch_size}")

    # DataModule 初期化
    print("\n2. DataModule 初期化中...")
    datamodule = VertebralFractureDataModule(
        data_dir=cfg.dir.data.base,
        hu_windows={
            'channel_1': cfg.hu_windows.channel_1,
            'channel_2': cfg.hu_windows.channel_2,
            'channel_3': cfg.hu_windows.channel_3,
        },
        image_size=(cfg.image_size.height, cfg.image_size.width),
        batch_size=cfg.training.batch_size,
        num_workers=0,  # デバッグ用に0に設定
        n_folds=cfg.n_folds,
        fold_id=cfg.split.fold_id,
        augmentation=OmegaConf.to_container(cfg.augmentation, resolve=True),
        oversample_fracture=cfg.class_balance.oversample_fracture,
        oversample_factor=cfg.class_balance.oversample_factor,
    )

    # データセットセットアップ
    datamodule.setup(stage="fit")

    # DataLoader取得
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    print(f"\n3. データセット情報:")
    print(f"   訓練データセット: {len(datamodule.train_dataset)} サンプル")
    print(f"   検証データセット: {len(datamodule.val_dataset)} サンプル")

    # 訓練データの検証
    print("\n4. 訓練データのバッチを検証中...")
    verify_batches(train_loader, num_batches, num_samples_per_batch,
                   prefix="train", with_augmentation=True)

    # 検証データの検証
    print("\n5. 検証データのバッチを検証中...")
    verify_batches(val_loader, num_batches, num_samples_per_batch,
                   prefix="val", with_augmentation=False)

    print("\n" + "="*80)
    print("検証完了！")
    print("="*80)


def verify_batches(dataloader, num_batches, num_samples_per_batch,
                   prefix="train", with_augmentation=False):
    """バッチデータを検証して可視化"""

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        images = batch['image']  # (B, C, H, W)
        masks = batch['mask']    # (B, 1, H, W)
        labels = batch['label']  # (B,)
        metadata = batch['metadata']

        print(f"\n  バッチ {batch_idx + 1}/{num_batches}:")
        print(f"    画像shape: {images.shape}")
        print(f"    マスクshape: {masks.shape}")
        print(f"    ラベル: {labels.tolist()}")

        # 統計情報
        print(f"    画像 [min, max]: [{images.min():.3f}, {images.max():.3f}]")
        print(f"    画像 [mean, std]: [{images.mean():.3f}, {images.std():.3f}]")
        print(f"    マスク sum: {masks.sum().item():.0f} pixels")
        print(f"    マスク 陽性率: {(masks.sum() / masks.numel() * 100):.4f}%")

        # チャンネルごとの統計
        for ch in range(3):
            ch_data = images[:, ch, :, :]
            print(f"    Ch{ch+1} [min, max, mean, std]: "
                  f"[{ch_data.min():.3f}, {ch_data.max():.3f}, "
                  f"{ch_data.mean():.3f}, {ch_data.std():.3f}]")

        # サンプル可視化
        visualize_samples(images, masks, labels, metadata,
                         batch_idx, num_samples_per_batch,
                         prefix, with_augmentation)


def visualize_samples(images, masks, labels, metadata,
                     batch_idx, num_samples, prefix, with_augmentation):
    """サンプルを可視化して保存"""

    batch_size = images.shape[0]
    num_samples = min(num_samples, batch_size)

    # 骨折サンプルと非骨折サンプルを分ける
    fracture_indices = [i for i, label in enumerate(labels) if label == 1]
    non_fracture_indices = [i for i, label in enumerate(labels) if label == 0]

    # 可視化するインデックスを選択
    selected_indices = []

    # 骨折サンプルを優先
    if fracture_indices:
        selected_indices.extend(fracture_indices[:min(2, len(fracture_indices))])

    # 非骨折サンプルを追加
    remaining = num_samples - len(selected_indices)
    if remaining > 0 and non_fracture_indices:
        selected_indices.extend(non_fracture_indices[:min(remaining, len(non_fracture_indices))])

    if not selected_indices:
        selected_indices = list(range(num_samples))

    # 可視化
    num_rows = len(selected_indices)
    fig, axes = plt.subplots(num_rows, 4, figsize=(16, 4 * num_rows))

    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for row_idx, sample_idx in enumerate(selected_indices):
        img = images[sample_idx].cpu().numpy()  # (3, H, W)
        mask = masks[sample_idx, 0].cpu().numpy()  # (H, W)
        label = labels[sample_idx].item()

        case_id = metadata['case'][sample_idx]
        vertebra = metadata['vertebra'][sample_idx]
        slice_idx = metadata['slice_index'][sample_idx]

        # Ch1: [0, 1800] HU
        axes[row_idx, 0].imshow(img[0], cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 0].set_title(f'Ch1 [0,1800]HU\nCase{case_id} {vertebra} slice{slice_idx}')
        axes[row_idx, 0].axis('off')

        # Ch2: [-200, 300] HU
        axes[row_idx, 1].imshow(img[1], cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 1].set_title(f'Ch2 [-200,300]HU')
        axes[row_idx, 1].axis('off')

        # Ch3: [200, 1200] HU
        axes[row_idx, 2].imshow(img[2], cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 2].set_title(f'Ch3 [200,1200]HU')
        axes[row_idx, 2].axis('off')

        # Mask overlay
        axes[row_idx, 3].imshow(img[1], cmap='gray', vmin=0, vmax=1)
        if mask.sum() > 0:
            # マスクを赤色でオーバーレイ
            mask_overlay = np.ma.masked_where(mask == 0, mask)
            axes[row_idx, 3].imshow(mask_overlay, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
            axes[row_idx, 3].set_title(f'Mask Overlay\nLabel={label}, Pixels={mask.sum():.0f}')
        else:
            axes[row_idx, 3].set_title(f'Mask Overlay\nLabel={label}, No fracture')
        axes[row_idx, 3].axis('off')

    plt.tight_layout()

    # 保存
    output_dir = project_root / "notebook" / "dataloader_verification"
    output_dir.mkdir(parents=True, exist_ok=True)

    aug_suffix = "_aug" if with_augmentation else "_noaug"
    output_path = output_dir / f"{prefix}_batch{batch_idx:02d}{aug_suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    保存先: {output_path}")
    plt.close()


def check_data_consistency():
    """データの整合性をチェック"""
    print("\n6. データ整合性チェック...")

    cfg = load_config()

    # CSVファイルを1つ読み込んで確認
    import glob
    import pandas as pd

    csv_pattern = f"{cfg.dir.data.base}/slice_train/axial_mask/*/mask_labels_*.csv"
    csv_files = sorted(glob.glob(csv_pattern))

    if csv_files:
        print(f"   CSVファイル例: {csv_files[0]}")
        df = pd.read_csv(csv_files[0])
        print(f"   カラム: {df.columns.tolist()}")
        print(f"\n   最初の5行:")
        print(df.head())

        # 骨折ラベルの分布
        fracture_count = (df['Fracture_Label'] == 1).sum()
        total_count = len(df)
        print(f"\n   骨折スライス: {fracture_count}/{total_count} ({fracture_count/total_count*100:.2f}%)")

        # ファイルパスの確認
        if 'FullPath' in df.columns:
            sample_path = df.iloc[0]['FullPath']
            print(f"\n   画像パス例: {sample_path}")
            print(f"   ファイル存在確認: {Path(sample_path).exists()}")

        if 'MaskPath' in df.columns:
            sample_mask_path = df.iloc[0]['MaskPath']
            print(f"   マスクパス例: {sample_mask_path}")
            print(f"   ファイル存在確認: {Path(sample_mask_path).exists()}")


if __name__ == "__main__":
    # データ整合性チェック
    check_data_consistency()

    # データローダー検証
    verify_dataloader(num_batches=2, num_samples_per_batch=4)

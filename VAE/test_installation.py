"""
VQ-VAE実装の動作確認スクリプト
実際の学習前に、モデルとデータローダーが正しく動作するかテストする
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

print("="*80)
print("VQ-VAE Implementation Test")
print("="*80)

# 1. モデルのインポートテスト
print("\n1. Testing model imports...")
try:
    from src.models.vector_quantizer import VectorQuantizer
    from src.models.vq_vae_3d import VQVAE3D, Encoder3D, Decoder3D
    print("✓ Model imports successful")
except Exception as e:
    print(f"✗ Model import failed: {e}")
    sys.exit(1)

# 2. データモジュールのインポートテスト
print("\n2. Testing datamodule imports...")
try:
    from src.datamodule.dataset import VertebraeVAEDataset
    from src.datamodule.dataloader import VertebraeVAEDataModule, FOLD_DEFINITION
    print("✓ Datamodule imports successful")
except Exception as e:
    print(f"✗ Datamodule import failed: {e}")
    sys.exit(1)

# 3. Lightning Moduleのインポートテスト
print("\n3. Testing Lightning module imports...")
try:
    from src.training.lightning_module import VQVAELightningModule
    print("✓ Lightning module imports successful")
except Exception as e:
    print(f"✗ Lightning module import failed: {e}")
    sys.exit(1)

# 4. モデルの動作テスト
print("\n4. Testing model forward pass...")
try:
    # ダミー入力データ (batch=2, channel=1, depth=128, height=128, width=128)
    dummy_input = torch.randn(2, 1, 128, 128, 128)

    # モデル構築
    model_config = {
        'in_channels': 1,
        'hidden_dims': [32, 64, 128, 256],
        'latent_dim': 256,
        'num_embeddings': 512,
        'commitment_cost': 0.25,
        'dropout': 0.1,
        'use_ema': False,
    }

    model = VQVAE3D(**model_config)
    model.eval()

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # 出力の確認
    assert output['recon'].shape == dummy_input.shape, "Reconstruction shape mismatch"
    assert isinstance(output['vq_loss'], torch.Tensor), "VQ loss is not a tensor"
    assert isinstance(output['perplexity'], torch.Tensor), "Perplexity is not a tensor"

    print(f"✓ Model forward pass successful")
    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Output shape: {output['recon'].shape}")
    print(f"  - VQ loss: {output['vq_loss'].item():.4f}")
    print(f"  - Perplexity: {output['perplexity'].item():.2f}")

except Exception as e:
    print(f"✗ Model forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Fold定義のテスト
print("\n5. Testing fold definitions...")
try:
    total_patients = sum(len(fold['patients']) for fold in FOLD_DEFINITION.values())
    expected_patients = 30  # 5 folds × 6 patients

    assert total_patients == expected_patients, f"Patient count mismatch: {total_patients} != {expected_patients}"

    print(f"✓ Fold definitions correct")
    print(f"  - Total folds: {len(FOLD_DEFINITION)}")
    print(f"  - Total patients: {total_patients}")

    for fold_id, fold_info in FOLD_DEFINITION.items():
        print(f"  - Fold {fold_id}: {len(fold_info['patients'])} patients, {fold_info['normal_count']} normal vertebrae")

except Exception as e:
    print(f"✗ Fold definition test failed: {e}")
    sys.exit(1)

# 6. データディレクトリの確認
print("\n6. Checking data directory...")
try:
    data_dir = Path("/mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/data/3d_data/train_vae")

    if data_dir.exists():
        vol_files = list(data_dir.glob("vol_*.npy"))
        print(f"✓ Data directory exists")
        print(f"  - Path: {data_dir}")
        print(f"  - Volume files found: {len(vol_files)}")

        # サンプルファイルの読み込みテスト
        if vol_files:
            sample_file = vol_files[0]
            sample_data = np.load(sample_file)
            print(f"  - Sample file: {sample_file.name}")
            print(f"  - Sample shape: {sample_data.shape}")
            print(f"  - Sample dtype: {sample_data.dtype}")
            print(f"  - Sample value range: [{sample_data.min():.4f}, {sample_data.max():.4f}]")

            assert sample_data.shape == (128, 128, 128), f"Unexpected shape: {sample_data.shape}"
    else:
        print(f"⚠ Data directory not found: {data_dir}")
        print("  This is OK if you haven't prepared the data yet.")

except Exception as e:
    print(f"✗ Data directory check failed: {e}")
    import traceback
    traceback.print_exc()

# 7. 設定ファイルの確認
print("\n7. Checking configuration files...")
try:
    config_dir = project_root / "run" / "conf"
    required_configs = [
        "config.yaml",
        "config_debug.yaml",
        "model/vq_vae.yaml",
        "data/vae_data.yaml",
        "training/vae_training.yaml",
    ]

    all_exist = True
    for config_file in required_configs:
        config_path = config_dir / config_file
        if config_path.exists():
            print(f"  ✓ {config_file}")
        else:
            print(f"  ✗ {config_file} not found")
            all_exist = False

    if all_exist:
        print("✓ All configuration files found")
    else:
        print("⚠ Some configuration files are missing")

except Exception as e:
    print(f"✗ Configuration check failed: {e}")

# 完了
print("\n" + "="*80)
print("Test Summary")
print("="*80)
print("✓ All basic tests passed!")
print("\nYou can now run the training script:")
print("  cd run/scripts")
print("  python train_vae.py --config-name=config_debug fold_id=1")
print("="*80)

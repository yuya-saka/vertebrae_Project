"""
Test script to verify the setup is correct
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
import numpy as np


def test_imports():
    """Test all required imports."""
    print("Testing imports...")

    try:
        import pytorch_lightning as pl
        print(f"  ✓ PyTorch Lightning: {pl.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch Lightning: {e}")

    try:
        import hydra
        print(f"  ✓ Hydra: {hydra.__version__}")
    except ImportError as e:
        print(f"  ✗ Hydra: {e}")

    try:
        import nibabel as nib
        print(f"  ✓ NiBabel: {nib.__version__}")
    except ImportError as e:
        print(f"  ✗ NiBabel: {e}")

    try:
        import cv2
        print(f"  ✓ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"  ✗ OpenCV: {e}")

    try:
        import pandas as pd
        print(f"  ✓ Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"  ✗ Pandas: {e}")

    print(f"  ✓ PyTorch: {torch.__version__}")
    print(f"  ✓ NumPy: {np.__version__}")


def test_model():
    """Test model instantiation."""
    print("\nTesting model...")

    from src.model.attention_gate import AttentionGate
    from src.model.attention_unet import AttentionUNet

    # Test Attention Gate
    print("  Testing AttentionGate...")
    attention_gate = AttentionGate(F_g=64, F_l=64, F_int=32)
    g = torch.randn(2, 64, 32, 32)
    x = torch.randn(2, 64, 32, 32)
    out = attention_gate(g, x)
    assert out.shape == x.shape
    print(f"    ✓ AttentionGate output shape: {out.shape}")

    # Test Attention U-Net
    print("  Testing AttentionUNet...")
    model = AttentionUNet(in_channels=3, out_channels=1, init_features=64, depth=4)
    model.initialize_weights()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256)
    print(f"    ✓ AttentionUNet output shape: {out.shape}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"    ✓ Number of parameters: {num_params:,}")


def test_losses_and_metrics():
    """Test loss functions and metrics."""
    print("\nTesting losses and metrics...")

    from src.modelmodule.losses import DiceLoss, CombinedLoss
    from src.modelmodule.metrics import calculate_all_metrics

    # Test data
    predictions = torch.randn(2, 1, 256, 256)
    targets = torch.randint(0, 2, (2, 1, 256, 256)).float()

    # Test Dice Loss
    dice_loss = DiceLoss()
    loss = dice_loss(predictions, targets)
    print(f"  ✓ DiceLoss: {loss.item():.4f}")

    # Test Combined Loss
    combined_loss = CombinedLoss()
    total_loss, dice, bce = combined_loss(predictions, targets)
    print(f"  ✓ CombinedLoss: {total_loss.item():.4f} (Dice: {dice.item():.4f}, BCE: {bce.item():.4f})")

    # Test metrics
    metrics = calculate_all_metrics(predictions, targets)
    print(f"  ✓ Metrics: {metrics}")


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")

    from omegaconf import OmegaConf

    config_dir = project_root / "run" / "conf"
    config_file = config_dir / "config.yaml"

    if config_file.exists():
        cfg = OmegaConf.load(config_file)
        print(f"  ✓ Config file loaded: {config_file}")
        print(f"    Experiment: {cfg.get('experiment', {}).get('name', 'N/A')}")
    else:
        print(f"  ✗ Config file not found: {config_file}")


def main():
    """Run all tests."""
    print("="*80)
    print("Setup Test")
    print("="*80)

    test_imports()
    test_model()
    test_losses_and_metrics()
    test_config()

    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)


if __name__ == '__main__':
    main()

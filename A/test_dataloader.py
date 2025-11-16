"""
Test script for new PNG dataset dataloader.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf

from A.src.datamodule.dataloader import create_dataloaders
from A.src.utils.common import set_seed, split_patients


@hydra.main(version_base=None, config_path="run/conf", config_name="config")
def test_dataloader(cfg: DictConfig):
    """Test dataloader with new PNG dataset."""

    print("="*80)
    print("Testing DataLoader with New PNG Dataset")
    print("="*80)

    # Display configuration
    print("\nConfiguration:")
    print(f"  Axis: {cfg.data_direction.axis}")
    print(f"  CSV file: {cfg.data_direction.csv_file}")
    print(f"  Image size: {cfg.data_direction.image_size}")
    print(f"  Batch size: {cfg.training.batch_size}")
    print(f"  Fold: {cfg.split.fold_id}/{cfg.split.n_folds}")

    # Set seed
    set_seed(cfg.seed)

    # Split patients
    train_ids, val_ids = split_patients(
        cfg.train_patient_ids,
        cfg.split.n_folds,
        cfg.split.fold_id
    )

    print(f"\nPatient Split:")
    print(f"  Train patients ({len(train_ids)}): {train_ids}")
    print(f"  Val patients ({len(val_ids)}): {val_ids}")

    # Create dataloaders
    print("\n" + "="*80)
    print("Creating DataLoaders...")
    print("="*80)

    try:
        train_loader, val_loader = create_dataloaders(
            train_patient_ids=train_ids,
            val_patient_ids=val_ids,
            cfg=cfg
        )

        print(f"\n✓ DataLoaders created successfully!")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")

        # Test loading one batch
        print("\n" + "="*80)
        print("Testing batch loading...")
        print("="*80)

        train_iter = iter(train_loader)
        batch = next(train_iter)

        print(f"\nBatch structure:")
        print(f"  Image shape: {batch['image'].shape}")  # Expected: (B, 3, 256, 256)
        print(f"  Mask shape: {batch['mask'].shape}")    # Expected: (B, 1, 256, 256)
        print(f"  Label shape: {batch['label_class'].shape}")  # Expected: (B,)
        print(f"  Label values: {batch['label_class']}")
        print(f"  Metadata keys: {batch['metadata'].keys()}")

        # Check class balance in batch
        labels = batch['label_class'].numpy()
        fracture_count = (labels == 1).sum()
        non_fracture_count = (labels == 0).sum()
        print(f"\nClass balance in batch:")
        print(f"  Fracture: {fracture_count}")
        print(f"  Non-fracture: {non_fracture_count}")
        print(f"  Ratio: {fracture_count}/{non_fracture_count}")

        # Check image value range
        print(f"\nImage value range:")
        print(f"  Min: {batch['image'].min().item():.3f}")
        print(f"  Max: {batch['image'].max().item():.3f}")
        print(f"  Mean: {batch['image'].mean().item():.3f}")

        print(f"\nMask value range:")
        print(f"  Min: {batch['mask'].min().item():.3f}")
        print(f"  Max: {batch['mask'].max().item():.3f}")
        print(f"  Mean: {batch['mask'].mean().item():.3f}")

        print("\n" + "="*80)
        print("✓ All tests passed!")
        print("="*80)

    except Exception as e:
        print(f"\n✗ Error during testing:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_dataloader()

"""
Training script for Attention U-Net Vertebral Fracture Segmentation

Usage:
    python train.py
    python train.py experiment.name=my_experiment training.batch_size=32
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger

from src.datamodule.dataloader import VertebralFractureDataModule
from src.modelmodule.model_module import SegmentationModule


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""

    # Print configuration
    print("="*80)
    print("Configuration:")
    print("="*80)
    print(OmegaConf.to_yaml(cfg))
    print("="*80)

    # Set random seed
    pl.seed_everything(cfg.training.seed, workers=True)

    # Create output directory
    experiment_name = cfg.experiment.name
    fold_id = cfg.split.fold_id
    output_dir = Path(cfg.dir.output.train) / experiment_name / f"fold_{fold_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Save configuration
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        OmegaConf.save(cfg, f)
    print(f"Configuration saved to: {config_save_path}")

    # Initialize DataModule
    print("\n" + "="*80)
    print("Initializing DataModule...")
    print("="*80)

    datamodule = VertebralFractureDataModule(
        data_dir=cfg.dir.data.base,
        hu_windows={
            'channel_1': cfg.hu_windows.channel_1,
            'channel_2': cfg.hu_windows.channel_2,
            'channel_3': cfg.hu_windows.channel_3,
        },
        image_size=(cfg.image_size.height, cfg.image_size.width),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        n_folds=cfg.n_folds,
        fold_id=cfg.split.fold_id,
        augmentation=OmegaConf.to_container(cfg.augmentation, resolve=True),
        oversample_fracture=cfg.class_balance.oversample_fracture,
        oversample_factor=cfg.class_balance.oversample_factor,
    )

    # Initialize Model
    print("\n" + "="*80)
    print("Initializing Model...")
    print("="*80)

    model = SegmentationModule(
        model_config={
            'in_channels': cfg.n_input_channels,
            'out_channels': cfg.n_output_channels,
            'init_features': cfg.model.architecture.init_features,
            'depth': cfg.model.architecture.depth,
            'attention_mode': cfg.model.architecture.attention.mode,
            'dropout': cfg.model.architecture.dropout,
        },
        optimizer_config=OmegaConf.to_container(cfg.optimizer, resolve=True),
        scheduler_config=OmegaConf.to_container(cfg.scheduler, resolve=True),
        loss_config=OmegaConf.to_container(cfg.loss, resolve=True),
        threshold_optimization_config=OmegaConf.to_container(cfg.threshold_optimization, resolve=True),
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")

    # Setup callbacks
    print("\n" + "="*80)
    print("Setting up callbacks...")
    print("="*80)

    callbacks = []

    # Model checkpoint
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=cfg.checkpoint.filename,
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=cfg.checkpoint.save_last,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    print(f"  ✓ ModelCheckpoint: {checkpoint_dir}")

    # Early stopping
    early_stopping_callback = EarlyStopping(
        monitor=cfg.early_stopping.monitor,
        patience=cfg.early_stopping.patience,
        mode=cfg.early_stopping.mode,
        min_delta=cfg.early_stopping.min_delta,
        verbose=True,
    )
    callbacks.append(early_stopping_callback)
    print(f"  ✓ EarlyStopping: monitor={cfg.early_stopping.monitor}, patience={cfg.early_stopping.patience}")

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    print(f"  ✓ LearningRateMonitor")

    # Setup logger
    print("\n" + "="*80)
    print("Setting up logger...")
    print("="*80)

    if cfg.wandb.project:
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{experiment_name}_fold{fold_id}",
            save_dir=cfg.wandb.save_dir,
            log_model=cfg.wandb.log_model,
        )
        # Log hyperparameters
        wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
        print(f"  ✓ W&B Logger: {cfg.wandb.project}")
        logger = wandb_logger
    else:
        logger = None
        print(f"  ✓ No logger configured")

    # Initialize Trainer
    print("\n" + "="*80)
    print("Initializing Trainer...")
    print("="*80)

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        log_every_n_steps=cfg.training.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
    )

    print(f"  Max epochs: {cfg.training.max_epochs}")
    print(f"  Precision: {cfg.training.precision}")
    print(f"  Gradient clip: {cfg.training.gradient_clip_val}")

    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)

    trainer.fit(model, datamodule=datamodule)

    # Save final results
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)

    print(f"\nBest model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best {cfg.checkpoint.monitor}: {checkpoint_callback.best_model_score:.4f}")

    # Save best model info
    best_model_info = {
        'best_model_path': checkpoint_callback.best_model_path,
        'best_model_score': float(checkpoint_callback.best_model_score),
        'monitor': cfg.checkpoint.monitor,
    }

    import json
    with open(output_dir / "best_model_info.json", 'w') as f:
        json.dump(best_model_info, f, indent=2)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()

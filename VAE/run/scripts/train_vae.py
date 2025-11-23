"""
Training Script for 3D VQ-VAE

Usage:
    # 1つのFoldで学習
    python train_vae.py fold_id=1

    # デバッグモード (少ないepoch)
    python train_vae.py fold_id=1 training.max_epochs=5 dataset.batch_size=2

    # 実験名を変更
    python train_vae.py fold_id=1 experiment.name=vqvae_exp1

    # 全Foldで学習 (別々に実行)
    for i in {1..5}; do python train_vae.py fold_id=$i; done
"""

import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
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

from src.datamodule.dataloader import VertebraeVAEDataModule
from src.training.lightning_module import VQVAELightningModule


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """メイン学習関数"""

    # 設定の表示
    print("=" * 80)
    print("VQ-VAE Training Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # シード設定
    pl.seed_everything(cfg.training.seed, workers=True)

    # 出力ディレクトリの作成
    experiment_name = cfg.experiment.name
    fold_id = cfg.fold_id
    output_dir = Path(cfg.output_dir) / experiment_name / f"fold_{fold_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput Directory: {output_dir}")
    print(f"Fold ID: {fold_id}")
    print("=" * 80)

    # 設定の保存
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        OmegaConf.save(cfg, f)
    print(f"Configuration saved to: {config_save_path}\n")

    # DataModule初期化
    print("=" * 80)
    print("Initializing DataModule...")
    print("=" * 80)

    datamodule = VertebraeVAEDataModule(
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        fold_id=fold_id,
        augmentation=OmegaConf.to_container(cfg.dataset.augmentation, resolve=True),
        pin_memory=cfg.dataset.pin_memory,
    )

    # Model初期化
    print("=" * 80)
    print("Initializing Model...")
    print("=" * 80)

    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    lightning_module = VQVAELightningModule(
        model_config=model_config,
        learning_rate=cfg.training.learning_rate,
        optimizer=cfg.training.optimizer,
        weight_decay=cfg.training.weight_decay,
        scheduler=cfg.training.scheduler,
        scheduler_config=OmegaConf.to_container(cfg.training.scheduler_config, resolve=True),
        recon_loss_type=cfg.training.recon_loss_type,
        log_image_freq=cfg.training.log_image_freq,
    )

    print(f"Model Architecture:")
    print(lightning_module.model)
    print("=" * 80)

    # WandB Logger初期化
    print("\nInitializing WandB Logger...")
    wandb_logger = WandbLogger(
        project=cfg.training.wandb.project,
        entity=cfg.training.wandb.entity,
        name=f"{experiment_name}_fold{fold_id}",
        save_dir=str(output_dir),
        tags=cfg.training.wandb.tags + [f"fold_{fold_id}"],
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Callbacks設定
    print("Setting up Callbacks...")

    # Model Checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="vqvae-{epoch:03d}-{val/total_loss:.4f}",
        monitor=cfg.training.checkpoint.monitor,
        mode=cfg.training.checkpoint.mode,
        save_top_k=cfg.training.checkpoint.save_top_k,
        save_last=cfg.training.checkpoint.save_last,
        verbose=True,
    )

    # Early Stopping
    early_stopping_callback = EarlyStopping(
        monitor=cfg.training.early_stopping.monitor,
        patience=cfg.training.early_stopping.patience,
        mode=cfg.training.early_stopping.mode,
        verbose=True,
    )

    # Learning Rate Monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    callbacks = [checkpoint_callback, early_stopping_callback, lr_monitor]

    # Trainer初期化
    print("=" * 80)
    print("Initializing Trainer...")
    print("=" * 80)

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.training.deterministic,
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # 勾配クリッピング
    )

    print(f"Max Epochs: {cfg.training.max_epochs}")
    print(f"Accelerator: {cfg.accelerator}")
    print(f"Devices: {cfg.devices}")
    print(f"Precision: {cfg.precision}")
    print("=" * 80)

    # 学習開始
    print("\nStarting Training...")
    print("=" * 80)

    trainer.fit(lightning_module, datamodule=datamodule)

    # 学習完了
    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    print("=" * 80)

    # WandB終了
    wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()

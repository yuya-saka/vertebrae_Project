#!/usr/bin/env python3
"""
Multi-Task U-Net Training Script

Usage:
    python train.py                          # Default: axial/fold_0
    python train.py split=fold_1             # Axial/fold_1
    python train.py data=coronal split=fold_2 # Coronal/fold_2
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime
import logging

from src.model.multitask_unet import MultiTaskUNet
from src.modelmodule.multitask_loss import MultiTaskLoss
from src.modelmodule.metrics import compute_pr_auc, compute_metrics_batch
from src.datamodule.dataloader import create_dataloaders
from src.utils.common import set_seed, split_patients


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""

    # ========================================
    # 0. Experiment setup
    # ========================================
    axis = cfg.data_direction.axis
    fold_id = cfg.split.fold_id
    model_name = cfg.model.encoder_name

    # Experiment name: axis/fold_X
    experiment_name = f"{axis}/fold_{fold_id}"

    # Output directory: A/output/axis/fold_X/
    output_dir = Path(cfg.output_base_dir) / axis / f"fold_{fold_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Subdirectories
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    # Update config with experiment info
    OmegaConf.set_struct(cfg, False)
    if 'experiment' not in cfg:
        cfg.experiment = {}
    cfg.experiment.name = experiment_name
    cfg.output_dir = str(output_dir)
    cfg.checkpoint_dir = str(checkpoint_dir)
    cfg.log_dir = str(log_dir)
    OmegaConf.set_struct(cfg, True)

    # ========================================
    # 1. Logging setup
    # ========================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.addHandler(file_handler)

    print("=" * 80)
    print("Multi-Task U-Net Training")
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Axis: {axis}")
    print(f"Fold: {fold_id}")
    print(f"Model: {model_name}")
    print(f"Description: {cfg.experiment.description}")
    print(f"Output dir: {output_dir}")
    print(f"Log file: {log_file}")
    print("=" * 80)

    # Set seed
    set_seed(cfg.seed)

    # Device setup
    if torch.cuda.is_available():
        if hasattr(cfg, 'device') and cfg.device.gpu_id is not None:
            device = torch.device(f'cuda:{cfg.device.gpu_id}')
            torch.cuda.set_device(cfg.device.gpu_id)
            print(f"Device: GPU {cfg.device.gpu_id} ({torch.cuda.get_device_name(cfg.device.gpu_id)})")
        else:
            device = torch.device('cuda')
            print(f"Device: GPU 0 (auto-detected) ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"Device: CPU")

    # Save config
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        OmegaConf.save(cfg, f)
    print(f"Config saved: {config_save_path}")

    # ========================================
    # 2. WandB initialization
    # ========================================
    if cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=experiment_name,
            group=axis,
            tags=[axis, f"fold_{fold_id}", model_name, "multitask", "attention"],
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode
        )
        wandb.config.update({
            "experiment_name": experiment_name,
            "output_dir": str(output_dir),
            "axis": axis,
            "fold_id": fold_id
        })

    # ========================================
    # 3. Patient split
    # ========================================
    train_ids, val_ids = split_patients(
        cfg.train_patient_ids,
        cfg.split.n_folds,
        cfg.split.fold_id
    )
    print(f"\nPatient Split (Fold {cfg.split.fold_id}/{cfg.split.n_folds}):")
    print(f"  Train patients: {len(train_ids)} - {train_ids[:5]}...")
    print(f"  Val patients: {len(val_ids)} - {val_ids}")

    # ========================================
    # 4. Create DataLoaders
    # ========================================
    print("\nCreating DataLoaders...")
    train_loader, val_loader = create_dataloaders(
        train_patient_ids=train_ids,
        val_patient_ids=val_ids,
        cfg=cfg
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # ========================================
    # 5. Create model
    # ========================================
    print("\nCreating Model...")
    model = MultiTaskUNet(cfg.model).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ========================================
    # 6. Loss function
    # ========================================
    criterion = MultiTaskLoss(
        w_class=cfg.loss.w_class,
        w_seg=cfg.loss.w_seg,
        seg_loss_type=cfg.loss.seg_loss_type,
        focal_alpha=cfg.loss.focal_alpha,
        focal_gamma=cfg.loss.focal_gamma
    )

    # ========================================
    # 7. Optimizer
    # ========================================
    if cfg.optimizer.use_differential_lr:
        optimizer = torch.optim.AdamW([
            {
                'params': model.get_encoder_params(),
                'lr': cfg.optimizer.lr * cfg.optimizer.encoder_lr_factor
            },
            {
                'params': model.get_decoder_params(),
                'lr': cfg.optimizer.lr
            }
        ], weight_decay=cfg.optimizer.weight_decay)
        print(f"\nOptimizer: AdamW with differential LR")
        print(f"  Encoder LR: {cfg.optimizer.lr * cfg.optimizer.encoder_lr_factor}")
        print(f"  Decoder LR: {cfg.optimizer.lr}")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )
        print(f"\nOptimizer: AdamW with LR={cfg.optimizer.lr}")

    # ========================================
    # 8. Scheduler
    # ========================================
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg.scheduler.mode,
        factor=cfg.scheduler.factor,
        patience=cfg.scheduler.patience,
        min_lr=cfg.scheduler.min_lr
    )

    # ========================================
    # 9. Training loop
    # ========================================
    best_val_metric = 0.0 if cfg.training.checkpoint.mode == "max" else float('inf')
    patience_counter = 0

    for epoch in range(cfg.training.max_epochs):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{cfg.training.max_epochs}")
        print(f"{'=' * 80}")

        # Training
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            cfg=cfg
        )

        # Validation
        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch
        )

        # Scheduler step
        scheduler.step(val_metrics['val_loss'])

        # Current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Print metrics
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f} | Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Train PR-AUC: {train_metrics['train_pr_auc']:.4f} | Val PR-AUC: {val_metrics['val_pr_auc']:.4f}")
        print(f"  Val Dice: {val_metrics['val_dice']:.4f} | Val IoU: {val_metrics['val_iou']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # WandB logging
        if cfg.wandb.mode != "disabled":
            wandb.log({
                **train_metrics,
                **val_metrics,
                'lr': current_lr,
                'epoch': epoch
            })

        # Checkpoint saving
        monitor_metric = val_metrics[cfg.training.checkpoint.monitor]
        is_best = False

        if cfg.training.checkpoint.mode == "max":
            if monitor_metric > best_val_metric:
                best_val_metric = monitor_metric
                is_best = True
                patience_counter = 0
            else:
                patience_counter += 1
        else:  # mode == "min"
            if monitor_metric < best_val_metric:
                best_val_metric = monitor_metric
                is_best = True
                patience_counter = 0
            else:
                patience_counter += 1

        if is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                cfg=cfg,
                filename='best_model.pth'
            )
            print(f"  ✓ Best model saved! ({cfg.training.checkpoint.monitor}={monitor_metric:.4f})")

        # Save last checkpoint
        if cfg.training.checkpoint.save_last:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                cfg=cfg,
                filename='last_model.pth'
            )

        # Early stopping
        if cfg.training.early_stopping.enabled:
            if patience_counter >= cfg.training.early_stopping.patience:
                print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
                print(f"  No improvement for {patience_counter} epochs")
                break

    # Training completed
    print(f"\n{'=' * 80}")
    print("Training completed!")
    print(f"Best {cfg.training.checkpoint.monitor}: {best_val_metric:.4f}")
    print(f"{'=' * 80}")

    if cfg.wandb.mode != "disabled":
        wandb.finish()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    cfg: DictConfig
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_loss_class = 0.0
    total_loss_seg = 0.0

    all_preds_class = []
    all_targets_class = []

    pbar = tqdm(loader, desc=f"Train", leave=False)

    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        labels_class = batch['label_class'].to(device).float()

        # Forward
        pred_class, pred_seg = model(images)

        # Loss
        losses = criterion(pred_class, pred_seg, labels_class, masks)
        loss = losses['total']

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Collect metrics
        total_loss += loss.item()
        total_loss_class += losses['class']
        total_loss_seg += losses['seg']

        all_preds_class.extend(pred_class.detach().cpu().numpy())
        all_targets_class.extend(labels_class.detach().cpu().numpy())

        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'loss_cls': losses['class'],
            'loss_seg': losses['seg']
        })

    # Epoch metrics
    avg_loss = total_loss / len(loader)
    avg_loss_class = total_loss_class / len(loader)
    avg_loss_seg = total_loss_seg / len(loader)

    pr_auc = compute_pr_auc(
        np.array(all_preds_class),
        np.array(all_targets_class)
    )

    return {
        'train_loss': avg_loss,
        'train_loss_class': avg_loss_class,
        'train_loss_seg': avg_loss_seg,
        'train_pr_auc': pr_auc
    }


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    epoch: int
) -> dict:
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    total_loss_class = 0.0
    total_loss_seg = 0.0

    all_preds_class = []
    all_targets_class = []

    all_dice = []
    all_iou = []

    pbar = tqdm(loader, desc=f"Val", leave=False)

    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            labels_class = batch['label_class'].to(device).float()

            # Forward
            pred_class, pred_seg = model(images)

            # Loss
            losses = criterion(pred_class, pred_seg, labels_class, masks)
            loss = losses['total']

            # Collect metrics
            total_loss += loss.item()
            total_loss_class += losses['class']
            total_loss_seg += losses['seg']

            all_preds_class.extend(pred_class.cpu().numpy())
            all_targets_class.extend(labels_class.cpu().numpy())

            # Segmentation metrics
            batch_metrics = compute_metrics_batch(pred_class, pred_seg, labels_class, masks)
            all_dice.append(batch_metrics['dice'])
            all_iou.append(batch_metrics['iou'])

            pbar.set_postfix({'loss': loss.item()})

    # Epoch metrics
    avg_loss = total_loss / len(loader)
    avg_loss_class = total_loss_class / len(loader)
    avg_loss_seg = total_loss_seg / len(loader)

    pr_auc = compute_pr_auc(
        np.array(all_preds_class),
        np.array(all_targets_class)
    )

    avg_dice = np.mean(all_dice)
    avg_iou = np.mean(all_iou)

    return {
        'val_loss': avg_loss,
        'val_loss_class': avg_loss_class,
        'val_loss_seg': avg_loss_seg,
        'val_pr_auc': pr_auc,
        'val_dice': avg_dice,
        'val_iou': avg_iou
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    cfg: DictConfig,
    filename: str
):
    """Save model checkpoint."""
    checkpoint_dir = Path(cfg.checkpoint_dir)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': OmegaConf.to_container(cfg, resolve=True),
        'axis': cfg.data_direction.axis,
        'fold_id': cfg.split.fold_id,
        'model_name': cfg.model.encoder_name
    }

    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    main()

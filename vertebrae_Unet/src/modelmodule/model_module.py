"""
PyTorch Lightning Module for Vertebral Fracture Segmentation
"""

import torch
import pytorch_lightning as pl
from typing import Dict, Optional

from ..model.attention_unet import AttentionUNet
from ..model.pretrained_unet import PretrainedUNet
from .losses import CombinedLoss, FocalTverskyLossCombined
from .metrics import calculate_all_metrics, pr_auc_score


class SegmentationModule(pl.LightningModule):
    """
    PyTorch Lightning Module for segmentation training and evaluation.

    Args:
        model_config: Model configuration dict
        optimizer_config: Optimizer configuration dict
        scheduler_config: Learning rate scheduler configuration dict
        loss_config: Loss function configuration dict
    """

    def __init__(
        self,
        model_config: Dict,
        optimizer_config: Dict,
        scheduler_config: Dict,
        loss_config: Dict,
        threshold_optimization_config: Optional[Dict] = None,
    ):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Initialize model based on _target_ in config
        model_target = model_config.get('_target_', '')

        if 'PretrainedUNet' in model_target or model_config.get('encoder_name'):
            # Pre-trained U-Net with ImageNet backbone
            self.model = PretrainedUNet(
                encoder_name=model_config.get('encoder_name', 'resnet34'),
                encoder_weights=model_config.get('encoder_weights', 'imagenet'),
                in_channels=model_config.get('in_channels', 3),
                classes=model_config.get('classes', 1),
                architecture=model_config.get('architecture', 'unetplusplus'),
                decoder_attention_type=model_config.get('decoder_attention_type', 'scse'),
                encoder_depth=model_config.get('encoder_depth', 5),
                decoder_channels=model_config.get('decoder_channels', None),
            )
            self.model_type = 'pretrained'
            print(f"Initialized PretrainedUNet with {model_config.get('encoder_name', 'resnet34')} encoder")
        else:
            # Custom Attention U-Net (backward compatibility)
            self.model = AttentionUNet(
                in_channels=model_config.get('in_channels', 3),
                out_channels=model_config.get('out_channels', 1),
                init_features=model_config.get('init_features', 64),
                depth=model_config.get('depth', 4),
                attention_mode=model_config.get('attention_mode', 'additive'),
                dropout=model_config.get('dropout', 0.1),
            )
            # Initialize weights for custom model
            self.model.initialize_weights()
            self.model_type = 'custom'
            print(f"Initialized custom Attention U-Net")

        # Loss function - choose based on loss_config
        loss_type = loss_config.get('loss_type', 'combined')  # 'combined' or 'focal_tversky'

        if loss_type == 'focal_tversky':
            self.criterion = FocalTverskyLossCombined(
                focal_tversky_weight=loss_config.get('focal_tversky_weight', 0.7),
                focal_weight=loss_config.get('focal_weight', 0.3),
                tversky_alpha=loss_config.get('tversky_alpha', 0.7),
                tversky_beta=loss_config.get('tversky_beta', 0.3),
                tversky_gamma=loss_config.get('tversky_gamma', 0.75),
                focal_alpha=loss_config.get('focal_alpha', 0.25),
                focal_gamma=loss_config.get('focal_gamma', 2.0),
                smooth=loss_config.get('smooth', 1.0),
            )
            self.loss_type = 'focal_tversky'
        else:
            self.criterion = CombinedLoss(
                dice_weight=loss_config.get('dice_weight', 0.5),
                bce_weight=loss_config.get('bce_weight', 0.5),
                smooth=loss_config.get('smooth', 1.0),
                pos_weight=loss_config.get('pos_weight', None),
            )
            self.loss_type = 'combined'

        # Store configs
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        # Metrics threshold
        self.threshold = 0.5

        # Threshold optimization configuration
        if threshold_optimization_config is None:
            threshold_optimization_config = {}

        self.threshold_optimization_enabled = threshold_optimization_config.get('enabled', True)

        if self.threshold_optimization_enabled:
            min_threshold = threshold_optimization_config.get('min_threshold', 0.01)
            max_threshold = threshold_optimization_config.get('max_threshold', 0.95)
            num_candidates = threshold_optimization_config.get('num_candidates', 95)
            self.threshold_candidates = torch.linspace(min_threshold, max_threshold, num_candidates).tolist()
        else:
            self.threshold_candidates = [0.5]  # Only use default threshold

        self.validation_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        images = batch['image']
        masks = batch['mask']

        # Forward pass
        predictions = self.forward(images)

        # Calculate loss
        total_loss, loss1, loss2 = self.criterion(predictions, masks)

        # Calculate metrics
        metrics = calculate_all_metrics(predictions.detach(), masks, self.threshold)

        # Log losses
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.loss_type == 'focal_tversky':
            self.log('train_focal_tversky_loss', loss1, on_step=False, on_epoch=True, logger=True)
            self.log('train_focal_loss', loss2, on_step=False, on_epoch=True, logger=True)
        else:
            self.log('train_dice_loss', loss1, on_step=False, on_epoch=True, logger=True)
            self.log('train_bce_loss', loss2, on_step=False, on_epoch=True, logger=True)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            self.log(f'train_{metric_name}', metric_value, on_step=False, on_epoch=True, logger=True)

        return total_loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        """Validation step."""
        images = batch['image']
        masks = batch['mask']

        # Forward pass
        predictions = self.forward(images)

        # Calculate loss
        total_loss, loss1, loss2 = self.criterion(predictions, masks)

        # Calculate metrics with default threshold
        metrics = calculate_all_metrics(predictions, masks, self.threshold)

        # Log losses
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.loss_type == 'focal_tversky':
            self.log('val_focal_tversky_loss', loss1, on_step=False, on_epoch=True, logger=True)
            self.log('val_focal_loss', loss2, on_step=False, on_epoch=True, logger=True)
        else:
            self.log('val_dice_loss', loss1, on_step=False, on_epoch=True, logger=True)
            self.log('val_bce_loss', loss2, on_step=False, on_epoch=True, logger=True)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            self.log(f'val_{metric_name}', metric_value, on_step=False, on_epoch=True, prog_bar=(metric_name=='dice'), logger=True)

        # Store probabilities and targets for optimal threshold search
        # Convert to probabilities and move to CPU to save GPU memory
        probabilities = torch.sigmoid(predictions).detach().cpu()
        targets_cpu = masks.detach().cpu()

        self.validation_outputs.append({
            'probabilities': probabilities,
            'targets': targets_cpu
        })

    def on_validation_epoch_end(self) -> None:
        """Calculate optimal threshold based on PRAUC at the end of validation epoch."""
        if not self.validation_outputs or not self.threshold_optimization_enabled:
            self.validation_outputs.clear()
            return

        # Gather outputs from all processes (DDP support)
        all_probabilities = torch.cat([x['probabilities'] for x in self.validation_outputs], dim=0)
        all_targets = torch.cat([x['targets'] for x in self.validation_outputs], dim=0)

        # All-gather for DDP
        if self.trainer.world_size > 1:
            # Gather from all processes
            all_probabilities = self.all_gather(all_probabilities)
            all_targets = self.all_gather(all_targets)
            # Flatten the gathered tensors (world_size, batch, ...)
            all_probabilities = all_probabilities.reshape(-1, *all_probabilities.shape[2:])
            all_targets = all_targets.reshape(-1, *all_targets.shape[2:])

        # Calculate PRAUC (threshold-independent metric based on probability scores)
        prauc = pr_auc_score(all_probabilities, all_targets)

        # Move to device once before the loop for efficiency
        device = self.device
        all_probabilities = all_probabilities.to(device)
        all_targets = all_targets.to(device)

        # Search for optimal threshold that maximizes F1 score
        # Note: PRAUC itself is threshold-independent, but we optimize threshold for F1
        best_f1 = 0.0
        best_threshold = 0.5
        best_metrics = None

        for threshold in self.threshold_candidates:
            # Calculate metrics at this threshold
            metrics = calculate_all_metrics(all_probabilities, all_targets, threshold)

            # Use F1 score for threshold optimization
            f1 = metrics['f1']

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = metrics

        # Update the threshold for future predictions
        self.threshold = best_threshold
        self.hparams.threshold = best_threshold

        # Log optimal threshold and PRAUC
        self.log('val_optimal_threshold', best_threshold, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_prauc', prauc, on_epoch=True, prog_bar=True, logger=True)

        # Log metrics at optimal threshold
        if best_metrics:
            for metric_name, metric_value in best_metrics.items():
                self.log(f'val_optimal_{metric_name}', metric_value, on_epoch=True, logger=True)

        # Clear validation outputs
        self.validation_outputs.clear()

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        """Test step."""
        images = batch['image']
        masks = batch['mask']

        # Forward pass
        predictions = self.forward(images)

        # Calculate loss
        total_loss, loss1, loss2 = self.criterion(predictions, masks)

        # Calculate metrics
        metrics = calculate_all_metrics(predictions, masks, self.threshold)

        # Log losses
        self.log('test_loss', total_loss, on_step=False, on_epoch=True, logger=True)

        if self.loss_type == 'focal_tversky':
            self.log('test_focal_tversky_loss', loss1, on_step=False, on_epoch=True, logger=True)
            self.log('test_focal_loss', loss2, on_step=False, on_epoch=True, logger=True)
        else:
            self.log('test_dice_loss', loss1, on_step=False, on_epoch=True, logger=True)
            self.log('test_bce_loss', loss2, on_step=False, on_epoch=True, logger=True)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            self.log(f'test_{metric_name}', metric_value, on_step=False, on_epoch=True, logger=True)

    def predict_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Prediction step."""
        images = batch['image']

        # Forward pass
        predictions = self.forward(images)

        # Apply sigmoid and threshold
        probabilities = torch.sigmoid(predictions)
        binary_predictions = (probabilities > self.threshold).float()

        return {
            'predictions': binary_predictions,
            'probabilities': probabilities,
            'metadata': batch.get('metadata', None),
        }

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Optimizer
        optimizer_name = self.optimizer_config.get('name', 'adamw').lower()
        base_lr = self.optimizer_config.get('lr', 1e-4)
        weight_decay = self.optimizer_config.get('weight_decay', 1e-5)
        betas = self.optimizer_config.get('betas', (0.9, 0.999))

        # Check if differential learning rate is enabled (for pretrained models)
        use_differential_lr = self.optimizer_config.get('use_differential_lr', False)
        encoder_lr_multiplier = self.optimizer_config.get('encoder_lr_multiplier', 0.1)

        # Prepare parameter groups
        if use_differential_lr and self.model_type == 'pretrained':
            # Differential learning rates: lower LR for pretrained encoder
            encoder_params = self.model.get_encoder_lr_params()
            decoder_params = self.model.get_decoder_lr_params()

            param_groups = [
                {
                    'params': encoder_params,
                    'lr': base_lr * encoder_lr_multiplier,
                    'name': 'encoder'
                },
                {
                    'params': decoder_params,
                    'lr': base_lr,
                    'name': 'decoder'
                }
            ]
            print(f"Using differential learning rates:")
            print(f"  Encoder LR: {base_lr * encoder_lr_multiplier:.2e}")
            print(f"  Decoder LR: {base_lr:.2e}")
        else:
            # Standard single learning rate for all parameters
            param_groups = self.parameters()

        # Create optimizer
        if optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=base_lr,
                weight_decay=weight_decay,
                betas=betas,
            )
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                param_groups,
                lr=base_lr,
                weight_decay=weight_decay,
                betas=betas,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Scheduler
        scheduler_name = self.scheduler_config.get('name', 'reduce_lr_on_plateau').lower()

        if scheduler_name == 'reduce_lr_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.scheduler_config.get('mode', 'max'),
                factor=self.scheduler_config.get('factor', 0.5),
                patience=self.scheduler_config.get('patience', 5),
                min_lr=self.scheduler_config.get('min_lr', 1e-6),
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': self.scheduler_config.get('monitor', 'val_dice'),
                    'interval': 'epoch',
                    'frequency': 1,
                },
            }
        elif scheduler_name == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.get('T_max', 50),
                eta_min=self.scheduler_config.get('min_lr', 1e-6),
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                },
            }
        else:
            # No scheduler
            return optimizer

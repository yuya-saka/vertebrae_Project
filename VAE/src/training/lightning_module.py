"""
PyTorch Lightning Module for VQ-VAE Training
"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from ..models.vq_vae_3d import VQVAE3D, build_vqvae_3d


class VQVAELightningModule(pl.LightningModule):
    """
    VQ-VAE学習用のLightning Module

    Args:
        model_config: モデル設定の辞書
        learning_rate: 学習率
        optimizer: オプティマイザ名 ('adam' or 'adamw')
        weight_decay: 重み減衰
        scheduler: スケジューラ名 ('cosine' or 'plateau')
        scheduler_config: スケジューラ設定
        recon_loss_type: 再構成Loss ('l1' or 'l2')
        log_image_freq: 画像ロギングの頻度 (epoch)
    """

    def __init__(
        self,
        model_config: Dict,
        learning_rate: float = 1e-4,
        optimizer: str = 'adamw',
        weight_decay: float = 1e-4,
        scheduler: str = 'cosine',
        scheduler_config: Optional[Dict] = None,
        recon_loss_type: str = 'l1',
        log_image_freq: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()

        # モデルの構築
        self.model = build_vqvae_3d(model_config)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler
        self.scheduler_config = scheduler_config or {}
        self.recon_loss_type = recon_loss_type
        self.log_image_freq = log_image_freq

    def forward(self, x: torch.Tensor) -> Dict:
        """Forward pass"""
        return self.model(x)

    def _compute_reconstruction_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        再構成Lossを計算

        Args:
            recon: 再構成ボリューム (B, 1, D, H, W)
            target: 元のボリューム (B, 1, D, H, W)

        Returns:
            loss: スカラー
        """
        if self.recon_loss_type == 'l1':
            return F.l1_loss(recon, target)
        elif self.recon_loss_type == 'l2':
            return F.mse_loss(recon, target)
        else:
            raise ValueError(f"Unknown recon_loss_type: {self.recon_loss_type}")

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """訓練ステップ"""
        volume = batch  # (B, 1, D, H, W)

        # Forward pass
        output = self.model(volume)
        recon = output['recon']
        vq_loss = output['vq_loss']
        perplexity = output['perplexity']

        # 再構成Loss
        recon_loss = self._compute_reconstruction_loss(recon, volume)

        # 総合Loss
        total_loss = recon_loss + vq_loss

        # Logging
        self.log('train/recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/vq_loss', vq_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/perplexity', perplexity, on_step=True, on_epoch=True, prog_bar=True)

        # コードブック使用率
        if batch_idx % 100 == 0:
            usage_stats = self.model.vq_layer.get_codebook_usage(output['encoding_indices'])
            self.log('train/codebook_usage', usage_stats['usage_ratio'], on_step=True)

        return total_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """検証ステップ"""
        volume = batch  # (B, 1, D, H, W)

        # Forward pass
        output = self.model(volume)
        recon = output['recon']
        vq_loss = output['vq_loss']
        perplexity = output['perplexity']

        # 再構成Loss
        recon_loss = self._compute_reconstruction_loss(recon, volume)

        # 総合Loss
        total_loss = recon_loss + vq_loss

        # Logging
        self.log('val/recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/vq_loss', vq_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True)

        # 画像ロギング (epoch毎、最初のバッチのみ)
        if batch_idx == 0 and self.current_epoch % self.log_image_freq == 0:
            self._log_reconstruction_images(volume, recon)

        return total_loss

    def _log_reconstruction_images(self, original: torch.Tensor, recon: torch.Tensor):
        """
        再構成画像をWandBにロギング

        Args:
            original: 元のボリューム (B, 1, D, H, W)
            recon: 再構成ボリューム (B, 1, D, H, W)
        """
        import wandb

        # 最初のサンプルの中央スライスを可視化
        idx = 0
        depth = original.shape[2]
        mid_slice = depth // 2

        # 中央スライスを取得 (H, W)
        orig_slice = original[idx, 0, mid_slice, :, :].cpu().numpy()
        recon_slice = recon[idx, 0, mid_slice, :, :].cpu().numpy()

        # 誤差マップ
        error_slice = abs(orig_slice - recon_slice)

        # WandBにロギング
        self.logger.experiment.log({
            f"val/original_slice_epoch{self.current_epoch}": wandb.Image(orig_slice, caption="Original"),
            f"val/recon_slice_epoch{self.current_epoch}": wandb.Image(recon_slice, caption="Reconstructed"),
            f"val/error_slice_epoch{self.current_epoch}": wandb.Image(error_slice, caption="Error Map"),
        })

    def configure_optimizers(self):
        """オプティマイザとスケジューラの設定"""
        # Optimizer
        if self.optimizer_name == 'adam':
            optimizer = Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == 'adamw':
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        # Scheduler
        if self.scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.get('T_max', 100),
                eta_min=self.scheduler_config.get('eta_min', 1e-6),
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }

        elif self.scheduler_name == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_config.get('factor', 0.5),
                patience=self.scheduler_config.get('patience', 10),
                min_lr=self.scheduler_config.get('min_lr', 1e-6),
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/total_loss',
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }

        else:
            return optimizer

    def on_train_epoch_end(self):
        """訓練epoch終了時のコールバック"""
        # 学習率のロギング
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('train/learning_rate', current_lr, on_epoch=True)

    def get_reconstruction_error_map(self, volume: torch.Tensor) -> torch.Tensor:
        """
        再構成誤差マップを取得 (推論用)

        Args:
            volume: (B, 1, D, H, W)

        Returns:
            error_map: (B, 1, D, H, W)
        """
        self.eval()
        with torch.no_grad():
            error_map = self.model.get_reconstruction_error(volume, reduction='none')
        return error_map

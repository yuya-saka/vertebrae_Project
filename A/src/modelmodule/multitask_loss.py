"""
Multi-task loss functions for classification and segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Reference: https://arxiv.org/abs/1708.02002

    Args:
        alpha: Weighting factor for positive class (0-1)
        gamma: Focusing parameter (typically 2.0)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities (B, 1, H, W) or (B, H, W)
            target: Ground truth binary mask (B, 1, H, W) or (B, H, W)

        Returns:
            Focal loss value
        """
        pred = pred.view(-1)
        target = target.view(-1)

        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce

        return focal_loss.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.

    Args:
        smooth: Smoothing factor to avoid division by zero
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)

        Returns:
            Dice loss value
        """
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining classification and segmentation.

    Total Loss = w_class * Loss_class + w_seg * Loss_seg

    Args:
        w_class: Weight for classification loss (default: 1.0)
        w_seg: Weight for segmentation loss (default: 0.1)
        seg_loss_type: Type of segmentation loss ('focal', 'dice', 'focal_dice')
        focal_alpha: Alpha parameter for Focal Loss
        focal_gamma: Gamma parameter for Focal Loss
    """

    def __init__(
        self,
        w_class: float = 1.0,
        w_seg: float = 0.1,
        seg_loss_type: str = 'focal',
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.w_class = w_class
        self.w_seg = w_seg
        self.seg_loss_type = seg_loss_type

        # Select segmentation loss
        if seg_loss_type == 'focal':
            self.seg_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif seg_loss_type == 'dice':
            self.seg_criterion = DiceLoss()
        elif seg_loss_type == 'focal_dice':
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            self.dice_loss = DiceLoss()
        else:
            raise ValueError(f"Unknown seg_loss_type: {seg_loss_type}")

    def forward(
        self,
        pred_class: torch.Tensor,
        pred_seg: torch.Tensor,
        target_class: torch.Tensor,
        target_seg: torch.Tensor
    ) -> dict:
        """
        Compute multi-task loss.

        Args:
            pred_class: Classification predictions (B,)
            pred_seg: Segmentation predictions (B, 1, H, W)
            target_class: Classification labels (B,)
            target_seg: Segmentation masks (B, 1, H, W)

        Returns:
            Dict with 'total', 'class', 'seg' losses
        """
        # Classification loss (BCE)
        loss_class = F.binary_cross_entropy(pred_class, target_class)

        # Segmentation loss
        if self.seg_loss_type == 'focal_dice':
            loss_focal = self.focal_loss(pred_seg, target_seg)
            loss_dice = self.dice_loss(pred_seg, target_seg)
            loss_seg = (loss_focal + loss_dice) / 2.0
        else:
            loss_seg = self.seg_criterion(pred_seg, target_seg)

        # Total loss
        total_loss = self.w_class * loss_class + self.w_seg * loss_seg

        return {
            'total': total_loss,
            'class': loss_class.item(),
            'seg': loss_seg.item()
        }

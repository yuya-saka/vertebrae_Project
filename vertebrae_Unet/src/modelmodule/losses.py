"""
Loss functions for segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.

    Args:
        smooth: Smoothing factor to avoid division by zero
    """

    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice Loss.

        Args:
            predictions: Model predictions (B, 1, H, W) - logits
            targets: Ground truth masks (B, 1, H, W) - binary

        Returns:
            Dice loss value
        """
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(predictions)

        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Compute Dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )

        # Return Dice loss
        return 1.0 - dice


class TverskyLoss(nn.Module):
    """
    Tversky Loss for binary segmentation.

    Tversky loss is a generalization of Dice loss that allows for different
    weighting of false positives and false negatives. This is particularly
    useful for imbalanced datasets.

    Args:
        alpha: Weight for false positives (default: 0.7)
        beta: Weight for false negatives (default: 0.3)
               alpha + beta should equal 1.0
               Higher alpha penalizes false positives more
               Higher beta penalizes false negatives more
        smooth: Smoothing factor to avoid division by zero

    Reference:
        Salehi et al. (2017). Tversky loss function for image segmentation
        using 3D fully convolutional deep networks.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Tversky Loss.

        Args:
            predictions: Model predictions (B, 1, H, W) - logits
            targets: Ground truth masks (B, 1, H, W) - binary

        Returns:
            Tversky loss value
        """
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(predictions)

        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives, False Negatives
        TP = (predictions * targets).sum()
        FP = (predictions * (1 - targets)).sum()
        FN = ((1 - predictions) * targets).sum()

        # Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        # Return Tversky loss
        return 1.0 - tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for binary segmentation.

    Combines Tversky loss with focal loss concept to focus on hard examples.
    This is particularly effective for highly imbalanced segmentation tasks
    with small target regions (like vertebral fractures).

    Args:
        alpha: Weight for false positives (default: 0.7)
        beta: Weight for false negatives (default: 0.3)
        gamma: Focusing parameter (default: 0.75)
               Higher gamma focuses more on hard examples
               gamma = 0 reduces to Tversky loss
        smooth: Smoothing factor to avoid division by zero

    Reference:
        Abraham & Khan (2019). A Novel Focal Tversky Loss Function with
        Improved Attention U-Net for Lesion Segmentation.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        smooth: float = 1.0
    ):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Tversky Loss.

        Args:
            predictions: Model predictions (B, 1, H, W) - logits
            targets: Ground truth masks (B, 1, H, W) - binary

        Returns:
            Focal Tversky loss value
        """
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(predictions)

        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives, False Negatives
        TP = (predictions * targets).sum()
        FP = (predictions * (1 - targets)).sum()
        FN = ((1 - predictions) * targets).sum()

        # Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        # Focal Tversky loss
        focal_tversky = torch.pow(1.0 - tversky, self.gamma)

        return focal_tversky


class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation.

    Focal loss down-weights easy examples and focuses on hard negatives.
    This helps with class imbalance by reducing the loss contribution from
    easy (well-classified) examples.

    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
               alpha=0.25 means positive class gets 25% weight
        gamma: Focusing parameter (default: 2.0)
               Higher gamma focuses more on hard examples
               gamma=0 reduces to standard cross-entropy

    Reference:
        Lin et al. (2017). Focal Loss for Dense Object Detection.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.

        Args:
            predictions: Model predictions (B, 1, H, W) - logits
            targets: Ground truth masks (B, 1, H, W) - binary

        Returns:
            Focal loss value
        """
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(predictions)

        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Binary cross entropy
        bce = F.binary_cross_entropy(predictions, targets, reduction='none')

        # Focal weight
        p_t = predictions * targets + (1 - predictions) * (1 - targets)
        focal_weight = torch.pow(1.0 - p_t, self.gamma)

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal loss
        focal_loss = alpha_t * focal_weight * bce

        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined Dice Loss and Binary Cross Entropy Loss.

    Args:
        dice_weight: Weight for Dice loss
        bce_weight: Weight for BCE loss
        smooth: Smoothing factor for Dice loss
        pos_weight: Positive class weight for BCE loss (None for no weighting)
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1.0,
        pos_weight: float = None,
    ):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        self.dice_loss = DiceLoss(smooth=smooth)

        if pos_weight is not None:
            self.bce_loss = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight])
            )
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> tuple:
        """
        Compute combined loss.

        Args:
            predictions: Model predictions (B, 1, H, W) - logits
            targets: Ground truth masks (B, 1, H, W) - binary

        Returns:
            Tuple of (total_loss, dice_loss, bce_loss)
        """
        dice = self.dice_loss(predictions, targets)
        bce = self.bce_loss(predictions, targets)

        total_loss = self.dice_weight * dice + self.bce_weight * bce

        return total_loss, dice, bce


class FocalTverskyLossCombined(nn.Module):
    """
    Combined Focal Tversky Loss and Focal Loss.

    This combination is highly effective for imbalanced segmentation tasks
    with small target regions (like vertebral fractures < 1% of image).

    Args:
        focal_tversky_weight: Weight for Focal Tversky loss (default: 0.7)
        focal_weight: Weight for Focal loss (default: 0.3)
        tversky_alpha: Weight for false positives in Tversky (default: 0.7)
        tversky_beta: Weight for false negatives in Tversky (default: 0.3)
        tversky_gamma: Focusing parameter for Focal Tversky (default: 0.75)
        focal_alpha: Alpha parameter for Focal loss (default: 0.25)
        focal_gamma: Gamma parameter for Focal loss (default: 2.0)
        smooth: Smoothing factor

    Recommended settings for small fracture regions:
        - tversky_alpha=0.7, tversky_beta=0.3: Penalize FP more (reduce over-segmentation)
        - tversky_gamma=0.75: Focus on hard examples
        - focal_alpha=0.25: Give more weight to positive class
        - focal_gamma=2.0: Strong focus on hard negatives
    """

    def __init__(
        self,
        focal_tversky_weight: float = 0.7,
        focal_weight: float = 0.3,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
        tversky_gamma: float = 0.75,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth: float = 1.0,
    ):
        super(FocalTverskyLossCombined, self).__init__()
        self.focal_tversky_weight = focal_tversky_weight
        self.focal_weight = focal_weight

        self.focal_tversky_loss = FocalTverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma,
            smooth=smooth
        )

        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma
        )

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> tuple:
        """
        Compute combined Focal Tversky and Focal loss.

        Args:
            predictions: Model predictions (B, 1, H, W) - logits
            targets: Ground truth masks (B, 1, H, W) - binary

        Returns:
            Tuple of (total_loss, focal_tversky_loss, focal_loss)
        """
        ftl = self.focal_tversky_loss(predictions, targets)
        fl = self.focal_loss(predictions, targets)

        total_loss = self.focal_tversky_weight * ftl + self.focal_weight * fl

        return total_loss, ftl, fl

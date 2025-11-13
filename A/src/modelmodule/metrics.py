"""
Evaluation metrics for classification and segmentation.
"""

import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from typing import Tuple


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Dice coefficient.

    Args:
        pred: Predicted logits (B, 1, H, W)
        target: Ground truth binary mask (B, 1, H, W)
        threshold: Binarization threshold

    Returns:
        Dice coefficient (0-1)
    """
    # Convert logits to probabilities
    pred = torch.sigmoid(pred)

    pred_binary = (pred > threshold).float()
    target_binary = target.float()

    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-8)

    return dice.item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute IoU (Intersection over Union).

    Args:
        pred: Predicted logits (B, 1, H, W)
        target: Ground truth binary mask (B, 1, H, W)
        threshold: Binarization threshold

    Returns:
        IoU score (0-1)
    """
    # Convert logits to probabilities
    pred = torch.sigmoid(pred)

    pred_binary = (pred > threshold).float()
    target_binary = target.float()

    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection

    iou = intersection / (union + 1e-8)

    return iou.item()


def compute_pr_auc(pred_probs: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute PR-AUC (Precision-Recall Area Under Curve).

    Args:
        pred_probs: Predicted probabilities (N,)
        targets: Ground truth labels (N,)

    Returns:
        PR-AUC score (0-1)
    """
    if len(np.unique(targets)) < 2:
        # Only one class present
        return 0.0

    precision, recall, _ = precision_recall_curve(targets, pred_probs)
    pr_auc = auc(recall, precision)

    return pr_auc


def compute_metrics_batch(
    pred_class: torch.Tensor,
    pred_seg: torch.Tensor,
    target_class: torch.Tensor,
    target_seg: torch.Tensor
) -> dict:
    """
    Compute metrics for a batch.

    Args:
        pred_class: Classification logits (B,)
        pred_seg: Segmentation logits (B, 1, H, W)
        target_class: Classification labels (B,)
        target_seg: Segmentation masks (B, 1, H, W)

    Returns:
        Dict with all metrics
    """
    # Classification accuracy (binary threshold at 0.0 for logits)
    pred_class_binary = (pred_class > 0.0).float()
    class_acc = (pred_class_binary == target_class).float().mean().item()

    # Segmentation metrics
    dice = dice_coefficient(pred_seg, target_seg)
    iou = iou_score(pred_seg, target_seg)

    return {
        'class_acc': class_acc,
        'dice': dice,
        'iou': iou
    }

"""
Evaluation metrics for segmentation
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import auc as compute_auc


def dice_coefficient(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1.0) -> torch.Tensor:
    """
    Calculate Dice coefficient.

    Args:
        predictions: Model predictions (B, 1, H, W) - logits or probabilities
        targets: Ground truth masks (B, 1, H, W) - binary
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor

    Returns:
        Dice coefficient
    """
    # Apply sigmoid if needed and threshold
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)

    predictions = (predictions > threshold).float()

    # Flatten
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    # Calculate Dice
    intersection = (predictions * targets).sum()
    dice = (2.0 * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)

    return dice


def iou_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1.0) -> torch.Tensor:
    """
    Calculate IoU (Intersection over Union) score.

    Args:
        predictions: Model predictions (B, 1, H, W) - logits or probabilities
        targets: Ground truth masks (B, 1, H, W) - binary
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor

    Returns:
        IoU score
    """
    # Apply sigmoid if needed and threshold
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)

    predictions = (predictions > threshold).float()

    # Flatten
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    # Calculate IoU
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou


def precision_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate Precision.

    Args:
        predictions: Model predictions (B, 1, H, W) - logits or probabilities
        targets: Ground truth masks (B, 1, H, W) - binary
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor

    Returns:
        Precision score
    """
    # Apply sigmoid if needed and threshold
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)

    predictions = (predictions > threshold).float()

    # Flatten
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    # Calculate Precision: TP / (TP + FP)
    true_positive = (predictions * targets).sum()
    predicted_positive = predictions.sum()

    precision = (true_positive + smooth) / (predicted_positive + smooth)

    return precision


def recall_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate Recall.

    Args:
        predictions: Model predictions (B, 1, H, W) - logits or probabilities
        targets: Ground truth masks (B, 1, H, W) - binary
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor

    Returns:
        Recall score
    """
    # Apply sigmoid if needed and threshold
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)

    predictions = (predictions > threshold).float()

    # Flatten
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    # Calculate Recall: TP / (TP + FN)
    true_positive = (predictions * targets).sum()
    actual_positive = targets.sum()

    recall = (true_positive + smooth) / (actual_positive + smooth)

    return recall


def f1_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate F1 Score.

    Args:
        predictions: Model predictions (B, 1, H, W) - logits or probabilities
        targets: Ground truth masks (B, 1, H, W) - binary
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor

    Returns:
        F1 score
    """
    precision = precision_score(predictions, targets, threshold, smooth)
    recall = recall_score(predictions, targets, threshold, smooth)

    f1 = 2 * (precision * recall) / (precision + recall + smooth)

    return f1


def pr_auc_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Precision-Recall AUC (PRAUC) using trapezoidal rule.

    Args:
        predictions: Model predictions (B, 1, H, W) - probabilities [0, 1]
        targets: Ground truth masks (B, 1, H, W) - binary {0, 1}

    Returns:
        PRAUC score
    """
    # Apply sigmoid if needed
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)

    # Flatten and move to CPU
    predictions_flat = predictions.view(-1).cpu().numpy()
    targets_flat = targets.view(-1).cpu().numpy()

    # Check if there are any positive samples
    if targets_flat.sum() == 0:
        return 0.0

    # Sort by prediction score (descending)
    sorted_indices = np.argsort(-predictions_flat)
    sorted_targets = targets_flat[sorted_indices]

    # Calculate cumulative TP and FP
    tp_cumsum = np.cumsum(sorted_targets)
    fp_cumsum = np.cumsum(1 - sorted_targets)

    # Calculate precision and recall
    total_positives = targets_flat.sum()
    recall = tp_cumsum / total_positives
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Sort recall and precision for AUC calculation
    # Recall goes from 0 to 1, we need to sort it
    sorted_recall_indices = np.argsort(recall)
    sorted_recall = recall[sorted_recall_indices]
    sorted_precision = precision[sorted_recall_indices]

    # Calculate AUC using trapezoidal rule
    prauc = compute_auc(sorted_recall, sorted_precision)

    return float(prauc)


def calculate_all_metrics(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict:
    """
    Calculate all segmentation metrics.

    Args:
        predictions: Model predictions (B, 1, H, W) - logits or probabilities
        targets: Ground truth masks (B, 1, H, W) - binary
        threshold: Threshold for binarizing predictions

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'dice': dice_coefficient(predictions, targets, threshold).item(),
        'iou': iou_score(predictions, targets, threshold).item(),
        'precision': precision_score(predictions, targets, threshold).item(),
        'recall': recall_score(predictions, targets, threshold).item(),
        'f1': f1_score(predictions, targets, threshold).item(),
    }

    return metrics

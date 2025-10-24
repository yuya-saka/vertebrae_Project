"""
カスタム損失関数の実装
- FocalLoss: クラス不均衡対策（Lin et al., 2017）
- CustomDetectionLoss: YOLOv8のDetectionLossを継承し、Focal Lossに置き換え

Focal Loss: FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
- γ (gamma): Focusing parameter。易しいサンプルの重みを減少させる度合い
- α (alpha): Balancing factor。クラス不均衡の補正係数
"""

import torch
import torch.nn as nn
from ultralytics.utils.loss import v8DetectionLoss


class FocalLoss(nn.Module):
    """
    Focal Lossの実装（Lin et al., 2017）

    クラス不均衡問題に対処し、易しいサンプルの重みを減少させる。
    医療画像の骨折検出において、以下の効果が期待される：
    - 易しい背景領域（非骨折）の損失を抑制
    - 難しい骨折領域（Hard examples）の学習を強化
    - 偽陰性（骨折の見逃し）の削減

    Args:
        gamma (float): Focusing parameter。大きいほど易しいサンプルの重みが減少。
                      推奨値: 1.5～2.5（医療画像では2.0が一般的）
        alpha (float): Balancing factor。クラス不均衡の補正係数。
                      推奨値: 0.25～0.5（骨折30%:非骨折70%の場合は0.3程度）

    Example:
        >>> focal_loss = FocalLoss(gamma=2.0, alpha=0.3)
        >>> pred = torch.randn(8, 1, requires_grad=True)  # [batch_size, num_classes]
        >>> label = torch.empty(8, 1).random_(2)  # Binary labels [0 or 1]
        >>> loss = focal_loss(pred, label)
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, label):
        """
        Focal Lossの順伝播計算

        Args:
            pred (torch.Tensor): モデルの予測（logits）。Shape: [batch_size, ...]
            label (torch.Tensor): Ground truthラベル。Shape: [batch_size, ...]

        Returns:
            torch.Tensor: Focal Lossの値（スカラー）
        """
        # Binary Cross Entropy Loss（要素ごと）
        bce_loss = self.bce(pred, label)

        # 予測確率を計算（sigmoid適用）
        pred_prob = pred.sigmoid()

        # p_t: 正しいクラスの予測確率
        # label=1の場合: p_t = pred_prob
        # label=0の場合: p_t = 1 - pred_prob
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)

        # Focal Loss modulating factor: (1 - p_t)^gamma
        # 易しいサンプル（p_t ≈ 1）の重みを大幅に減少させる
        # 難しいサンプル（p_t ≈ 0）の重みを維持
        modulating_factor = (1.0 - p_t) ** self.gamma

        # Alpha balancing: クラス不均衡の補正
        # label=1（骨折）の場合: alpha_t = alpha
        # label=0（非骨折）の場合: alpha_t = 1 - alpha
        alpha_t = label * self.alpha + (1 - label) * (1 - self.alpha)

        # Focal Loss = alpha * (1-p_t)^gamma * BCE
        focal_loss = alpha_t * modulating_factor * bce_loss

        # 平均を取ってスカラー値を返す
        return focal_loss.mean()


class CustomDetectionLoss(v8DetectionLoss):
    """
    YOLOv8のDetectionLossを継承し、分類損失をFocal Lossに置き換える

    YOLOv8のデフォルト損失関数:
    - 分類損失: BCEWithLogitsLoss（Binary Cross Entropy）
    - BBox回帰損失: CIoU Loss
    - DFL損失: Distribution Focal Loss（BBox分布の最適化）

    このクラスは分類損失のみをFocal Lossに置き換える。
    BBox回帰損失とDFL損失は変更しない。

    Args:
        model: YOLOv8モデル
        gamma (float): Focal Lossのfocusing parameter（デフォルト: 2.0）
        alpha (float): Focal Lossのbalancing factor（デフォルト: 0.25）

    Example:
        >>> from ultralytics import YOLO
        >>> model = YOLO('yolov8n.pt')
        >>> custom_loss = CustomDetectionLoss(model, gamma=2.0, alpha=0.3)
    """

    def __init__(self, model, gamma=2.0, alpha=0.25):
        super().__init__(model)

        # BCEWithLogitsLossをFocalLossに置き換え
        # self.bceはYOLOv8の分類損失で使用される
        self.bce = FocalLoss(gamma=gamma, alpha=alpha)

        print(f"✅ CustomDetectionLoss initialized with Focal Loss")
        print(f"   - gamma (focusing parameter): {gamma}")
        print(f"   - alpha (balancing factor): {alpha}")
        print(f"   - Effect: Down-weights easy examples, focuses on hard negatives")

"""
Attention Gate for U-Net skip connections.

Based on: Attention U-Net (https://arxiv.org/abs/1804.03999)
"""

import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """
    Attention Gate for U-Net skip connections.

    The attention gate learns to focus on salient features from the encoder
    while suppressing irrelevant background regions.

    Args:
        F_g: Number of feature maps in gating signal (from decoder)
        F_l: Number of feature maps in skip connection (from encoder)
        F_int: Number of intermediate feature maps
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()

        # Gating signal transformation
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Skip connection transformation
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Attention coefficient generation
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Forward pass.

        Args:
            g: Gating signal from decoder (B, F_g, H, W)
            x: Skip connection from encoder (B, F_l, H, W)

        Returns:
            Attention-weighted feature map (B, F_l, H, W)
        """
        # Transform gating signal and skip connection
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Combine and compute attention coefficients
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # Apply attention weights to skip connection
        return x * psi

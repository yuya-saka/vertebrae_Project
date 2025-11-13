"""
Multi-Task U-Net with Attention Gates.

Y-shaped architecture with shared encoder and two task-specific heads:
- Classification head: GAP + FC
- Segmentation decoder: Attention U-Net
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple
from omegaconf import DictConfig

from .attention_gate import AttentionGate


class MultiTaskUNet(nn.Module):
    """
    Y-shaped Multi-Task U-Net.

    Architecture:
        - Shared Encoder: ResNet18/EfficientNet (pretrained)
        - Branch 1: Classification head (GAP + FC)
        - Branch 2: Segmentation decoder (Attention Gates)
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.encoder_name = cfg.encoder_name
        self.encoder_weights = cfg.encoder_weights

        # Build encoder
        if cfg.encoder_name == 'resnet18':
            self._build_resnet18_encoder(cfg)
        else:
            raise ValueError(f"Unsupported encoder: {cfg.encoder_name}")

        # Branch 1: Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(cfg.classifier.dropout),
            nn.Linear(512, 1)
        )

        # Branch 2: Segmentation decoder
        self.decoder_channels = cfg.decoder_channels  # [256, 128, 64, 32, 16]

        # Attention Gates
        self.att4 = AttentionGate(F_g=self.decoder_channels[0], F_l=256, F_int=128)
        self.att3 = AttentionGate(F_g=self.decoder_channels[1], F_l=128, F_int=64)
        self.att2 = AttentionGate(F_g=self.decoder_channels[2], F_l=64, F_int=32)
        self.att1 = AttentionGate(F_g=self.decoder_channels[3], F_l=64, F_int=16)

        # Decoder blocks
        self.up4 = self._make_decoder_block(512, self.decoder_channels[0])
        self.up3 = self._make_decoder_block(self.decoder_channels[0] + 256, self.decoder_channels[1])
        self.up2 = self._make_decoder_block(self.decoder_channels[1] + 128, self.decoder_channels[2])
        self.up1 = self._make_decoder_block(self.decoder_channels[2] + 64, self.decoder_channels[3])
        self.up0 = self._make_decoder_block(self.decoder_channels[3] + 64, self.decoder_channels[4])

        # Segmentation output head
        self.seg_head = nn.Conv2d(self.decoder_channels[4], 1, kernel_size=1)

    def _build_resnet18_encoder(self, cfg: DictConfig):
        """Build ResNet18 encoder."""
        backbone = models.resnet18(pretrained=(cfg.encoder_weights == 'imagenet'))

        # First conv layer (3 channels input)
        self.encoder_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if cfg.encoder_weights == 'imagenet':
            self.encoder_conv1.weight.data = backbone.conv1.weight.data

        self.encoder_bn1 = backbone.bn1
        self.encoder_relu = backbone.relu
        self.encoder_maxpool = backbone.maxpool

        # ResNet layers
        self.encoder_layer1 = backbone.layer1  # 64 channels
        self.encoder_layer2 = backbone.layer2  # 128 channels
        self.encoder_layer3 = backbone.layer3  # 256 channels
        self.encoder_layer4 = backbone.layer4  # 512 channels (bottleneck)

    def _make_decoder_block(self, in_channels: int, out_channels: int):
        """Create a decoder block with upsampling and convolution."""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input image (B, 3, 256, 256)

        Returns:
            p_class: Classification logits (B,)
            p_seg: Segmentation logits map (B, 1, 256, 256)
        """
        # Encoder forward pass
        x0 = self.encoder_conv1(x)  # (B, 64, 128, 128)
        x0 = self.encoder_bn1(x0)
        x0 = self.encoder_relu(x0)

        x1 = self.encoder_maxpool(x0)  # (B, 64, 64, 64)
        x1 = self.encoder_layer1(x1)   # (B, 64, 64, 64)

        x2 = self.encoder_layer2(x1)   # (B, 128, 32, 32)
        x3 = self.encoder_layer3(x2)   # (B, 256, 16, 16)
        x4 = self.encoder_layer4(x3)   # (B, 512, 8, 8) - bottleneck

        # Branch 1: Classification head
        p_class = self.classifier(x4).squeeze(1)  # (B,)

        # Branch 2: Segmentation decoder with Attention Gates
        d4 = self.up4(x4)  # (B, 256, 16, 16)
        x3_att = self.att4(g=d4, x=x3)
        d4 = torch.cat([d4, x3_att], dim=1)

        d3 = self.up3(d4)  # (B, 128, 32, 32)
        x2_att = self.att3(g=d3, x=x2)
        d3 = torch.cat([d3, x2_att], dim=1)

        d2 = self.up2(d3)  # (B, 64, 64, 64)
        x1_att = self.att2(g=d2, x=x1)
        d2 = torch.cat([d2, x1_att], dim=1)

        d1 = self.up1(d2)  # (B, 32, 128, 128)
        x0_att = self.att1(g=d1, x=x0)
        d1 = torch.cat([d1, x0_att], dim=1)

        d0 = self.up0(d1)  # (B, 16, 256, 256)

        p_seg = self.seg_head(d0)  # (B, 1, 256, 256)

        return p_class, p_seg

    def freeze_encoder(self):
        """Freeze encoder weights for fine-tuning."""
        for param in self.encoder_conv1.parameters():
            param.requires_grad = False
        for param in self.encoder_bn1.parameters():
            param.requires_grad = False
        for param in self.encoder_layer1.parameters():
            param.requires_grad = False
        for param in self.encoder_layer2.parameters():
            param.requires_grad = False
        for param in self.encoder_layer3.parameters():
            param.requires_grad = False
        for param in self.encoder_layer4.parameters():
            param.requires_grad = False

    def get_encoder_params(self):
        """Get encoder parameters for differential learning rate."""
        encoder_params = []
        encoder_params.extend(self.encoder_conv1.parameters())
        encoder_params.extend(self.encoder_bn1.parameters())
        encoder_params.extend(self.encoder_layer1.parameters())
        encoder_params.extend(self.encoder_layer2.parameters())
        encoder_params.extend(self.encoder_layer3.parameters())
        encoder_params.extend(self.encoder_layer4.parameters())
        return encoder_params

    def get_decoder_params(self):
        """Get decoder + classifier parameters."""
        decoder_params = []
        decoder_params.extend(self.classifier.parameters())
        decoder_params.extend(self.att4.parameters())
        decoder_params.extend(self.att3.parameters())
        decoder_params.extend(self.att2.parameters())
        decoder_params.extend(self.att1.parameters())
        decoder_params.extend(self.up4.parameters())
        decoder_params.extend(self.up3.parameters())
        decoder_params.extend(self.up2.parameters())
        decoder_params.extend(self.up1.parameters())
        decoder_params.extend(self.up0.parameters())
        decoder_params.extend(self.seg_head.parameters())
        return decoder_params

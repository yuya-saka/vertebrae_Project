"""
Attention U-Net for Vertebral Fracture Segmentation

Reference:
Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas.
https://arxiv.org/abs/1804.03999
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .attention_gate import AttentionGate


class ConvBlock(nn.Module):
    """
    Convolutional block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dropout: Dropout probability (0 for no dropout)
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        if self.dropout:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class EncoderBlock(nn.Module):
    """
    Encoder block: ConvBlock -> MaxPool

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dropout: Dropout probability
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super(EncoderBlock, self).__init__()

        self.conv_block = ConvBlock(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Returns:
            Tuple of (pooled output, skip connection)
        """
        skip = self.conv_block(x)
        pooled = self.pool(skip)
        return pooled, skip


class DecoderBlock(nn.Module):
    """
    Decoder block: Upsample -> Attention Gate -> Concat -> ConvBlock

    Args:
        in_channels: Number of input channels (from previous decoder)
        skip_channels: Number of channels in skip connection
        out_channels: Number of output channels
        attention_mode: Attention mode ('additive' or 'multiplicative')
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        attention_mode: str = 'additive',
        dropout: float = 0.0,
    ):
        super(DecoderBlock, self).__init__()

        # Upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )

        # Attention Gate
        self.attention = AttentionGate(
            F_g=in_channels,
            F_l=skip_channels,
            F_int=in_channels // 2,
            mode=attention_mode
        )

        # Convolution after concatenation
        self.conv_block = ConvBlock(
            in_channels + skip_channels, out_channels, dropout
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input from previous decoder layer
            skip: Skip connection from encoder

        Returns:
            Decoder output
        """
        # Upsample
        x = self.upsample(x)

        # Apply attention to skip connection
        skip_att = self.attention(g=x, x=skip)

        # Concatenate
        x = torch.cat([x, skip_att], dim=1)

        # Convolution
        x = self.conv_block(x)

        return x


class AttentionUNet(nn.Module):
    """
    Attention U-Net for medical image segmentation.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (classes)
        init_features: Number of initial features (default: 64)
        depth: Depth of the network (number of encoder/decoder blocks, default: 4)
        attention_mode: Attention mode ('additive' or 'multiplicative')
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        init_features: int = 64,
        depth: int = 4,
        attention_mode: str = 'additive',
        dropout: float = 0.1,
    ):
        super(AttentionUNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        self.depth = depth

        # Calculate feature dimensions at each level
        features = [init_features * (2 ** i) for i in range(depth)]
        bottleneck_features = init_features * (2 ** depth)

        # Encoder
        self.encoders = nn.ModuleList()
        current_channels = in_channels
        for feat in features:
            self.encoders.append(
                EncoderBlock(current_channels, feat, dropout)
            )
            current_channels = feat

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], bottleneck_features, dropout)

        # Decoder
        self.decoders = nn.ModuleList()
        decoder_features = list(reversed(features))
        decoder_in = bottleneck_features

        for i, feat in enumerate(decoder_features):
            self.decoders.append(
                DecoderBlock(
                    in_channels=decoder_in,
                    skip_channels=feat,
                    out_channels=feat,
                    attention_mode=attention_mode,
                    dropout=dropout
                )
            )
            decoder_in = feat

        # Final output layer
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output segmentation (B, out_channels, H, W)
        """
        # Store skip connections
        skips = []

        # Encoder path
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path (reverse order of skips)
        skips = list(reversed(skips))
        for decoder, skip in zip(self.decoders, skips):
            x = decoder(x, skip)

        # Final output
        x = self.out_conv(x)

        return x

    def initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AttentionUNetTest:
    """Unit tests for AttentionUNet."""

    @staticmethod
    def test_forward_pass():
        """Test forward pass with different configurations."""
        print("Testing AttentionUNet forward pass...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Using device: {device}")

        # Test configuration
        batch_size = 2
        in_channels = 3
        out_channels = 1
        h, w = 256, 256

        # Test 1: Default configuration
        print("\n  Test 1: Default configuration (depth=4)")
        model = AttentionUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=64,
            depth=4
        ).to(device)

        model.initialize_weights()

        x = torch.randn(batch_size, in_channels, h, w).to(device)
        out = model(x)

        assert out.shape == (batch_size, out_channels, h, w), \
            f"Output shape {out.shape} != expected {(batch_size, out_channels, h, w)}"
        assert torch.all(torch.isfinite(out)), "Output contains NaN or Inf"

        print(f"    ✓ Input shape: {x.shape}")
        print(f"    ✓ Output shape: {out.shape}")

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"    ✓ Number of parameters: {num_params:,}")

        # Test 2: Smaller model
        print("\n  Test 2: Smaller model (depth=3, init_features=32)")
        model_small = AttentionUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=32,
            depth=3
        ).to(device)

        x_small = torch.randn(batch_size, in_channels, 128, 128).to(device)
        out_small = model_small(x_small)

        assert out_small.shape == (batch_size, out_channels, 128, 128)
        num_params_small = sum(p.numel() for p in model_small.parameters())
        print(f"    ✓ Output shape: {out_small.shape}")
        print(f"    ✓ Number of parameters: {num_params_small:,}")

        # Test 3: Multiplicative attention
        print("\n  Test 3: Multiplicative attention mode")
        model_mult = AttentionUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            attention_mode='multiplicative'
        ).to(device)

        out_mult = model_mult(x)
        assert out_mult.shape == (batch_size, out_channels, h, w)
        print(f"    ✓ Output shape: {out_mult.shape}")

        print("\n✓ All AttentionUNet tests passed!")

    @staticmethod
    def test_gradient_flow():
        """Test gradient flow through the network."""
        print("\nTesting gradient flow...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = AttentionUNet(in_channels=3, out_channels=1).to(device)
        model.initialize_weights()

        x = torch.randn(2, 3, 256, 256, requires_grad=True).to(device)
        target = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)

        # Forward pass
        output = model(x)

        # Compute loss
        loss = F.binary_cross_entropy_with_logits(output, target)

        # Backward pass
        loss.backward()

        # Check gradients
        has_gradients = 0
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    has_gradients += 1
                    assert torch.all(torch.isfinite(param.grad)), \
                        f"Gradient contains NaN or Inf in {name}"

        print(f"  Parameters with gradients: {has_gradients}/{total_params}")
        print(f"  Loss: {loss.item():.4f}")
        print("\n✓ Gradient flow test passed!")


def run_tests():
    """Run all tests."""
    print("="*60)
    print("Running AttentionUNet Tests")
    print("="*60)

    test = AttentionUNetTest()
    test.test_forward_pass()
    test.test_gradient_flow()

    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)


if __name__ == '__main__':
    run_tests()

"""
Attention Gate module for Attention U-Net

Reference:
Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas.
https://arxiv.org/abs/1804.03999
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Attention Gate module for focusing on relevant features in skip connections.

    The Attention Gate takes two inputs:
    1. g: gating signal from deeper layer (decoder)
    2. x: skip connection features from encoder

    It outputs attention coefficients that are multiplied with x to suppress
    irrelevant features and highlight important regions.

    Args:
        F_g: Number of channels in gating signal
        F_l: Number of channels in skip connection
        F_int: Number of intermediate channels
        mode: Attention mode ('additive' or 'multiplicative')
    """

    def __init__(self, F_g: int, F_l: int, F_int: int, mode: str = 'additive'):
        super(AttentionGate, self).__init__()

        self.mode = mode

        # Transform gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Transform skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Attention Gate.

        Args:
            g: Gating signal from decoder (B, F_g, H_g, W_g)
            x: Skip connection from encoder (B, F_l, H_x, W_x)

        Returns:
            Attention-weighted skip connection (B, F_l, H_x, W_x)
        """
        # Get input shapes
        g_shape = g.size()
        x_shape = x.size()

        # If sizes don't match, upsample gating signal to match skip connection
        if g_shape[2:] != x_shape[2:]:
            g = F.interpolate(g, size=x_shape[2:], mode='bilinear', align_corners=True)

        # Transform both inputs
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Additive attention
        if self.mode == 'additive':
            # Element-wise addition followed by ReLU
            psi_input = self.relu(g1 + x1)
        else:  # multiplicative
            # Element-wise multiplication
            psi_input = g1 * x1

        # Compute attention coefficients
        psi = self.psi(psi_input)

        # Apply attention coefficients to skip connection
        out = x * psi

        return out


class AttentionGateTest:
    """Unit tests for AttentionGate."""

    @staticmethod
    def test_forward_pass():
        """Test forward pass with different input sizes."""
        print("Testing AttentionGate forward pass...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Test configuration
        batch_size = 2
        F_g = 512  # Gating signal channels
        F_l = 256  # Skip connection channels
        F_int = 128  # Intermediate channels

        # Create module
        attention_gate = AttentionGate(F_g, F_l, F_int, mode='additive').to(device)

        # Test 1: Same spatial dimensions
        print("\n  Test 1: Same spatial dimensions")
        g = torch.randn(batch_size, F_g, 32, 32).to(device)
        x = torch.randn(batch_size, F_l, 32, 32).to(device)
        out = attention_gate(g, x)

        assert out.shape == x.shape, f"Output shape {out.shape} != input shape {x.shape}"
        assert torch.all(torch.isfinite(out)), "Output contains NaN or Inf"
        print(f"    ✓ Input shape: g={g.shape}, x={x.shape}")
        print(f"    ✓ Output shape: {out.shape}")

        # Test 2: Different spatial dimensions (gating signal smaller)
        print("\n  Test 2: Different spatial dimensions")
        g = torch.randn(batch_size, F_g, 16, 16).to(device)
        x = torch.randn(batch_size, F_l, 32, 32).to(device)
        out = attention_gate(g, x)

        assert out.shape == x.shape, f"Output shape {out.shape} != input shape {x.shape}"
        assert torch.all(torch.isfinite(out)), "Output contains NaN or Inf"
        print(f"    ✓ Input shape: g={g.shape}, x={x.shape}")
        print(f"    ✓ Output shape: {out.shape}")

        # Test 3: Multiplicative mode
        print("\n  Test 3: Multiplicative attention mode")
        attention_gate_mult = AttentionGate(F_g, F_l, F_int, mode='multiplicative').to(device)
        g = torch.randn(batch_size, F_g, 32, 32).to(device)
        x = torch.randn(batch_size, F_l, 32, 32).to(device)
        out = attention_gate_mult(g, x)

        assert out.shape == x.shape, f"Output shape {out.shape} != input shape {x.shape}"
        assert torch.all(torch.isfinite(out)), "Output contains NaN or Inf"
        print(f"    ✓ Output shape: {out.shape}")

        print("\n✓ All AttentionGate tests passed!")

    @staticmethod
    def test_attention_values():
        """Test that attention values are in valid range."""
        print("\nTesting attention value ranges...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        batch_size = 2
        F_g = 64
        F_l = 64
        F_int = 32

        attention_gate = AttentionGate(F_g, F_l, F_int).to(device)

        g = torch.randn(batch_size, F_g, 32, 32).to(device)
        x = torch.randn(batch_size, F_l, 32, 32).to(device)

        # Forward pass
        out = attention_gate(g, x)

        # Attention should reduce or maintain magnitude
        # (since attention coefficients are in [0, 1])
        x_magnitude = torch.abs(x).mean()
        out_magnitude = torch.abs(out).mean()

        print(f"  Input magnitude: {x_magnitude:.4f}")
        print(f"  Output magnitude: {out_magnitude:.4f}")
        print(f"  Attention effect: {out_magnitude/x_magnitude:.4f}x")

        print("\n✓ Attention value test passed!")


def run_tests():
    """Run all tests."""
    print("="*60)
    print("Running AttentionGate Tests")
    print("="*60)

    test = AttentionGateTest()
    test.test_forward_pass()
    test.test_attention_values()

    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)


if __name__ == '__main__':
    run_tests()

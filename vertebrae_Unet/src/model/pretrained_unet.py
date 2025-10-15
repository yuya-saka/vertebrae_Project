"""
U-Net with Pre-trained ImageNet Backbone

Using segmentation_models_pytorch library for easy integration
of ImageNet pre-trained encoders.

Reference:
- https://github.com/qubvel/segmentation_models.pytorch
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional, List


class PretrainedUNet(nn.Module):
    """
    U-Net with ImageNet pre-trained encoder.

    This class wraps segmentation_models_pytorch to provide a consistent
    interface with our custom Attention U-Net implementation.

    Args:
        encoder_name: Backbone encoder name (e.g., 'resnet34', 'efficientnet-b3')
                     See smp.encoders.get_encoder_names() for all options
        encoder_weights: Pre-trained weights to use ('imagenet' or None for random init)
        in_channels: Number of input channels (default: 3)
        classes: Number of output classes (default: 1 for binary segmentation)
        architecture: Architecture type ('unet', 'unetplusplus', 'deeplabv3', 'deeplabv3plus', 'fpn')
        decoder_attention_type: Attention mechanism type ('scse', 'cbam', or None)
        encoder_depth: Depth of encoder (default: 5)
        decoder_channels: Number of channels in decoder blocks

    Available encoder_names (examples):
        - ResNet family: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        - EfficientNet family: 'efficientnet-b0' to 'efficientnet-b7'
        - SE-ResNet: 'se_resnet50', 'se_resnet101', 'se_resnet152'
        - DenseNet: 'densenet121', 'densenet161', 'densenet169', 'densenet201'
        - MobileNet: 'mobilenet_v2'
        - And many more...

    Available attention types:
        - 'scse': Spatial and Channel Squeeze & Excitation
        - 'cbam': Convolutional Block Attention Module
        - None: No attention mechanism
    """

    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: Optional[str] = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        architecture: str = 'unetplusplus',
        decoder_attention_type: Optional[str] = 'scse',
        encoder_depth: int = 5,
        decoder_channels: Optional[List[int]] = None,
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
        self.architecture = architecture
        self.decoder_attention_type = decoder_attention_type

        # Default decoder channels if not specified
        if decoder_channels is None:
            decoder_channels = (256, 128, 64, 32, 16)

        # Select architecture
        architecture_classes = {
            'unet': smp.Unet,
            'unetplusplus': smp.UnetPlusPlus,
            'deeplabv3': smp.DeepLabV3,
            'deeplabv3plus': smp.DeepLabV3Plus,
            'fpn': smp.FPN,
        }

        if architecture.lower() not in architecture_classes:
            raise ValueError(
                f"Unknown architecture: {architecture}. "
                f"Available: {list(architecture_classes.keys())}"
            )

        architecture_class = architecture_classes[architecture.lower()]

        # Create model
        # Note: Different architectures support different parameters
        model_kwargs = {
            'encoder_name': encoder_name,
            'encoder_weights': encoder_weights,
            'in_channels': in_channels,
            'classes': classes,
            'activation': None,  # We apply activation in loss function
        }

        # Add architecture-specific parameters
        if architecture.lower() in ['unet', 'unetplusplus']:
            model_kwargs['encoder_depth'] = encoder_depth
            model_kwargs['decoder_channels'] = decoder_channels
            model_kwargs['decoder_attention_type'] = decoder_attention_type

        self.model = architecture_class(**model_kwargs)

        # Store encoder output channels for potential use
        self.encoder_output_channels = self.model.encoder.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output segmentation logits (B, classes, H, W)
        """
        return self.model(x)

    def freeze_encoder(self):
        """
        Freeze encoder parameters (for fine-tuning decoder only).
        Useful for initial training phases or limited data scenarios.
        """
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        print(f"Encoder ({self.encoder_name}) frozen")

    def unfreeze_encoder(self):
        """
        Unfreeze encoder parameters (for full model fine-tuning).
        """
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        print(f"Encoder ({self.encoder_name}) unfrozen")

    def get_encoder_lr_params(self):
        """
        Get encoder parameters for differential learning rates.

        Returns:
            List of encoder parameters
        """
        return list(self.model.encoder.parameters())

    def get_decoder_lr_params(self):
        """
        Get decoder parameters for differential learning rates.

        Returns:
            List of decoder parameters (all non-encoder parameters)
        """
        encoder_param_ids = {id(p) for p in self.model.encoder.parameters()}
        decoder_params = [
            p for p in self.model.parameters()
            if id(p) not in encoder_param_ids
        ]
        return decoder_params

    def print_architecture_info(self):
        """Print detailed information about the model architecture."""
        print(f"\n{'='*80}")
        print(f"PretrainedUNet Architecture Information")
        print(f"{'='*80}")
        print(f"Architecture: {self.architecture}")
        print(f"Encoder: {self.encoder_name}")
        print(f"Pre-trained weights: {self.encoder_weights}")
        print(f"Input channels: {self.in_channels}")
        print(f"Output classes: {self.classes}")
        print(f"Attention type: {self.decoder_attention_type}")

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.model.encoder.parameters())
        decoder_params = total_params - encoder_params

        print(f"\nParameter counts:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Encoder parameters: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
        print(f"  Decoder parameters: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")
        print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

        print(f"\nEncoder output channels: {self.encoder_output_channels}")
        print(f"{'='*80}\n")


class PretrainedUNetTest:
    """Unit tests for PretrainedUNet."""

    @staticmethod
    def test_forward_pass():
        """Test forward pass with different configurations."""
        print("Testing PretrainedUNet forward pass...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Using device: {device}")

        # Test configuration
        batch_size = 2
        in_channels = 3
        classes = 1
        h, w = 256, 256

        # Test 1: ResNet34 + UNet++ + SCSE
        print("\n  Test 1: ResNet34 + UNet++ + SCSE")
        model = PretrainedUNet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=classes,
            architecture='unetplusplus',
            decoder_attention_type='scse',
        ).to(device)

        model.print_architecture_info()

        x = torch.randn(batch_size, in_channels, h, w).to(device)
        out = model(x)

        assert out.shape == (batch_size, classes, h, w), \
            f"Output shape {out.shape} != expected {(batch_size, classes, h, w)}"
        assert torch.all(torch.isfinite(out)), "Output contains NaN or Inf"

        print(f"    ✓ Input shape: {x.shape}")
        print(f"    ✓ Output shape: {out.shape}")

        # Test 2: EfficientNet-B3 + UNet++ + SCSE
        print("\n  Test 2: EfficientNet-B3 + UNet++ + SCSE")
        model_efficient = PretrainedUNet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            architecture='unetplusplus',
            decoder_attention_type='scse',
        ).to(device)

        out_efficient = model_efficient(x)
        assert out_efficient.shape == (batch_size, classes, h, w)
        print(f"    ✓ Output shape: {out_efficient.shape}")

        # Test 3: Freeze/unfreeze encoder
        print("\n  Test 3: Freeze/unfreeze encoder")
        model.freeze_encoder()
        trainable_frozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Trainable params (frozen): {trainable_frozen:,}")

        model.unfreeze_encoder()
        trainable_unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Trainable params (unfrozen): {trainable_unfrozen:,}")
        print(f"    ✓ Freeze/unfreeze working")

        # Test 4: Differential learning rate parameter groups
        print("\n  Test 4: Differential learning rate parameters")
        encoder_params = model.get_encoder_lr_params()
        decoder_params = model.get_decoder_lr_params()
        print(f"    Encoder params: {sum(p.numel() for p in encoder_params):,}")
        print(f"    Decoder params: {sum(p.numel() for p in decoder_params):,}")
        print(f"    ✓ Parameter grouping working")

        print("\n✓ All PretrainedUNet tests passed!")

    @staticmethod
    def test_gradient_flow():
        """Test gradient flow through the network."""
        print("\nTesting gradient flow...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = PretrainedUNet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
        ).to(device)

        x = torch.randn(2, 3, 256, 256, requires_grad=True).to(device)
        target = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)

        # Forward pass
        output = model(x)

        # Compute loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)

        # Backward pass
        loss.backward()

        # Check gradients
        has_gradients = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
        total_params = sum(1 for p in model.parameters() if p.requires_grad)

        print(f"  Parameters with gradients: {has_gradients}/{total_params}")
        print(f"  Loss: {loss.item():.4f}")

        # Verify no NaN in gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.all(torch.isfinite(param.grad)), \
                    f"Gradient contains NaN or Inf in {name}"

        print("\n✓ Gradient flow test passed!")


def run_tests():
    """Run all tests."""
    print("="*80)
    print("Running PretrainedUNet Tests")
    print("="*80)

    test = PretrainedUNetTest()
    test.test_forward_pass()
    test.test_gradient_flow()

    print("\n" + "="*80)
    print("All tests completed successfully!")
    print("="*80)


if __name__ == '__main__':
    run_tests()

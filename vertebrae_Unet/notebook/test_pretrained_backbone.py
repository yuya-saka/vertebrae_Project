"""
Test script for pre-trained backbone models
Verify functionality, memory usage, and performance
"""

import torch
import segmentation_models_pytorch as smp
import time


def test_model_forward_backward(encoder_name, attention_type='scse', batch_size=8):
    """
    Test forward and backward pass with given encoder.

    Args:
        encoder_name: Name of the encoder (e.g., 'resnet34')
        attention_type: Type of attention ('scse', 'cbam', None)
        batch_size: Batch size for testing
    """
    print(f"\n{'='*80}")
    print(f"Testing: {encoder_name} with {attention_type} attention")
    print(f"{'='*80}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    try:
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None,
            decoder_attention_type=attention_type,
        )
        model = model.to(device)
        print(f"✓ Model created successfully")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

    # Test input
    h, w = 256, 256
    x = torch.randn(batch_size, 3, h, w).to(device)
    target = torch.randint(0, 2, (batch_size, 1, h, w)).float().to(device)

    # Measure memory before forward
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory before forward: {mem_before:.3f} GB")

    # Forward pass
    try:
        start_time = time.time()
        with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
            output = model(x)
        forward_time = time.time() - start_time

        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Forward time: {forward_time:.3f} sec")
        print(f"  Throughput: {batch_size / forward_time:.1f} images/sec")

        assert output.shape == (batch_size, 1, h, w), f"Output shape mismatch: {output.shape}"
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

    # Measure memory after forward
    if device.type == 'cuda':
        mem_after_forward = torch.cuda.memory_allocated() / 1024**3
        peak_mem_forward = torch.cuda.max_memory_allocated() / 1024**3
        print(f"GPU memory after forward: {mem_after_forward:.3f} GB")
        print(f"Peak GPU memory (forward): {peak_mem_forward:.3f} GB")

    # Backward pass
    try:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)

        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time

        print(f"✓ Backward pass successful")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Backward time: {backward_time:.3f} sec")

        # Check gradients
        has_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total_param_count = sum(1 for _ in model.parameters())
        print(f"  Parameters with gradients: {has_grad}/{total_param_count}")

    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        return False

    # Measure memory after backward
    if device.type == 'cuda':
        mem_after_backward = torch.cuda.memory_allocated() / 1024**3
        peak_mem_backward = torch.cuda.max_memory_allocated() / 1024**3
        print(f"GPU memory after backward: {mem_after_backward:.3f} GB")
        print(f"Peak GPU memory (total): {peak_mem_backward:.3f} GB")

    # Clean up
    del model, x, target, output, loss
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print(f"✓ Test completed successfully for {encoder_name}")
    return True


def test_multiple_batch_sizes(encoder_name='resnet34', attention_type='scse'):
    """Test different batch sizes to find optimal setting."""
    print(f"\n{'='*80}")
    print(f"Testing multiple batch sizes for {encoder_name}")
    print(f"{'='*80}")

    batch_sizes = [4, 8, 16, 24, 32]
    results = []

    for bs in batch_sizes:
        print(f"\nTesting batch size: {bs}")
        try:
            success = test_model_forward_backward(encoder_name, attention_type, batch_size=bs)
            if success:
                results.append((bs, "✓ Success"))
            else:
                results.append((bs, "✗ Failed"))
                break  # Stop if failed
        except RuntimeError as e:
            if "out of memory" in str(e):
                results.append((bs, "✗ OOM"))
                print(f"Out of memory at batch size {bs}")
                break
            else:
                results.append((bs, f"✗ Error: {e}"))
                break

    print(f"\n{'='*80}")
    print("Batch Size Test Results:")
    print(f"{'='*80}")
    for bs, result in results:
        print(f"Batch size {bs:2d}: {result}")

    # Find recommended batch size
    successful_batch_sizes = [bs for bs, result in results if "Success" in result]
    if successful_batch_sizes:
        recommended = max(successful_batch_sizes)
        print(f"\n✓ Recommended batch size: {recommended}")
    else:
        print(f"\n✗ No successful batch size found")


def list_available_encoders():
    """List all available encoders in segmentation_models_pytorch."""
    print(f"\n{'='*80}")
    print("Available Encoders")
    print(f"{'='*80}")

    encoders = smp.encoders.get_encoder_names()

    # Categorize encoders
    resnet_family = [e for e in encoders if 'resnet' in e.lower()]
    efficientnet_family = [e for e in encoders if 'efficientnet' in e.lower()]
    other_encoders = [e for e in encoders if e not in resnet_family + efficientnet_family]

    print(f"\nResNet family ({len(resnet_family)}):")
    for encoder in sorted(resnet_family):
        print(f"  - {encoder}")

    print(f"\nEfficientNet family ({len(efficientnet_family)}):")
    for encoder in sorted(efficientnet_family):
        print(f"  - {encoder}")

    print(f"\nOther encoders ({len(other_encoders)}):")
    for encoder in sorted(other_encoders)[:20]:  # Show first 20
        print(f"  - {encoder}")
    if len(other_encoders) > 20:
        print(f"  ... and {len(other_encoders) - 20} more")

    print(f"\nTotal available encoders: {len(encoders)}")


def main():
    """Main test function."""
    print("="*80)
    print("Pre-trained Backbone Test Script")
    print("="*80)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\n✓ CUDA is available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"\n⚠ CUDA is not available, using CPU")

    # List available encoders
    list_available_encoders()

    # Test recommended encoders
    encoders_to_test = [
        ('resnet34', 'scse'),
        ('resnet50', 'scse'),
        ('efficientnet-b3', 'scse'),
    ]

    print(f"\n{'='*80}")
    print("Testing Recommended Encoders")
    print(f"{'='*80}")

    results = []
    for encoder_name, attention_type in encoders_to_test:
        success = test_model_forward_backward(
            encoder_name=encoder_name,
            attention_type=attention_type,
            batch_size=8
        )
        results.append((encoder_name, attention_type, success))

        # Clean up between tests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1)

    # Summary
    print(f"\n{'='*80}")
    print("Test Summary")
    print(f"{'='*80}")
    for encoder_name, attention_type, success in results:
        status = "✓ Pass" if success else "✗ Fail"
        print(f"{encoder_name:20s} + {attention_type:10s}: {status}")

    # Test batch sizes for ResNet34 (recommended)
    if torch.cuda.is_available():
        test_multiple_batch_sizes(encoder_name='resnet34', attention_type='scse')

    print(f"\n{'='*80}")
    print("All tests completed!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

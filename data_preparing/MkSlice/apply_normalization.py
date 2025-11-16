import numpy as np
import torch
import os
import math

def apply_normalization(input_array, output_size, interpolation_mode='trilinear', gpu_id=None):
    """
    Resamples a 3D volume to a target size using GPU-accelerated interpolation with PyTorch.

    Args:
        input_array (np.ndarray): The input 3D numpy array (assumed to be in D, H, W order).
        output_size (int or tuple): The target output size. If int, creates a cubic volume.
        interpolation_mode (str): 'trilinear' for image data, 'nearest' for segmentation masks.
        gpu_id (int, optional): GPU device ID to use. If None, uses CUDA_VISIBLE_DEVICES or default GPU 0.

    Returns:
        np.ndarray: The resampled 3D numpy array.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but GPU normalization was requested.")

    # GPU device selection
    if gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
    else:
        # Use CUDA_VISIBLE_DEVICES if set, otherwise use default GPU 0
        device = torch.device("cuda")

    if isinstance(output_size, int):
        target_shape = (output_size, output_size, output_size)
    else:
        # Ensure it's a tuple for the interpolate function
        target_shape = tuple(output_size)

    # PyTorch's interpolate function expects the input tensor in (N, C, D, H, W) format.
    # We add a batch (N=1) and a channel (C=1) dimension.
    tensor = torch.from_numpy(input_array.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)

    # align_corners=True is recommended for trilinear mode to better align with
    # libraries like OpenCV and Scikit-image. It has no effect on 'nearest'.
    align_corners = True if interpolation_mode == 'trilinear' else None
    
    normalized_tensor = torch.nn.functional.interpolate(
        tensor,
        size=target_shape,
        mode=interpolation_mode,
        align_corners=align_corners
    )

    # After interpolation, remove the batch and channel dimensions and move the tensor
    # back to the CPU before converting it to a NumPy array.
    return normalized_tensor.squeeze(0).squeeze(0).cpu().numpy()


def gpu_rotate_3d(input_array, angle, axes=(0, 1), reshape=True, gpu_id=None):
    """
    GPU-accelerated 3D rotation using PyTorch affine transformation.

    Args:
        input_array (np.ndarray): The input 3D numpy array.
        angle (float): Rotation angle in degrees.
        axes (tuple): Axes to rotate around. (0,1), (1,2), or (0,2).
        reshape (bool): Whether to reshape output to fit entire rotated volume.
        gpu_id (int, optional): GPU device ID to use.

    Returns:
        np.ndarray: The rotated 3D numpy array.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but GPU rotation was requested.")

    # GPU device selection
    if gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cuda")

    # Convert to tensor and move to GPU
    tensor = torch.from_numpy(input_array.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)

    # Convert angle to radians
    theta_rad = math.radians(angle)
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)

    # Create appropriate affine transformation matrix based on axes
    # PyTorch uses (D, H, W) format, corresponding to axes (0, 1, 2)
    if axes == (0, 1):  # Rotate in sagittal-coronal plane (XY plane in medical imaging)
        # Create 3D affine matrix for rotation around Z axis
        affine_matrix = torch.tensor([
            [cos_theta, -sin_theta, 0, 0],
            [sin_theta, cos_theta, 0, 0],
            [0, 0, 1, 0]
        ], dtype=torch.float32, device=device).unsqueeze(0)
    elif axes == (1, 2):  # Rotate in coronal-axial plane
        # Create 3D affine matrix for rotation around X axis
        affine_matrix = torch.tensor([
            [1, 0, 0, 0],
            [0, cos_theta, -sin_theta, 0],
            [0, sin_theta, cos_theta, 0]
        ], dtype=torch.float32, device=device).unsqueeze(0)
    elif axes == (0, 2):  # Rotate in sagittal-axial plane
        # Create 3D affine matrix for rotation around Y axis
        affine_matrix = torch.tensor([
            [cos_theta, 0, sin_theta, 0],
            [0, 1, 0, 0],
            [-sin_theta, 0, cos_theta, 0]
        ], dtype=torch.float32, device=device).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported axes: {axes}")

    # Calculate output size if reshape=True
    if reshape:
        D, H, W = input_array.shape
        if axes == (0, 1):
            # Calculate new dimensions after rotation
            new_D = int(math.ceil(abs(D * cos_theta) + abs(H * sin_theta)))
            new_H = int(math.ceil(abs(D * sin_theta) + abs(H * cos_theta)))
            new_W = W
            output_size = (new_D, new_H, new_W)
        elif axes == (1, 2):
            new_D = D
            new_H = int(math.ceil(abs(H * cos_theta) + abs(W * sin_theta)))
            new_W = int(math.ceil(abs(H * sin_theta) + abs(W * cos_theta)))
            output_size = (new_D, new_H, new_W)
        elif axes == (0, 2):
            new_D = int(math.ceil(abs(D * cos_theta) + abs(W * sin_theta)))
            new_H = H
            new_W = int(math.ceil(abs(D * sin_theta) + abs(W * cos_theta)))
            output_size = (new_D, new_H, new_W)
    else:
        output_size = input_array.shape

    # Generate sampling grid
    grid = torch.nn.functional.affine_grid(
        affine_matrix,
        (1, 1, *output_size),
        align_corners=False
    )

    # Apply affine transformation
    rotated_tensor = torch.nn.functional.grid_sample(
        tensor,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )

    # Convert back to numpy
    return rotated_tensor.squeeze(0).squeeze(0).cpu().numpy()
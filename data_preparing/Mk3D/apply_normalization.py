import numpy as np
import torch
import math

def apply_normalization(input_array, output_size, interpolation_mode='trilinear', gpu_id=None):
    """
    3DボリュームをGPUを使用してリサイズ（リサンプリング）する関数。
    
    Args:
        input_array (np.ndarray): 入力3D配列 (D, H, W)
        output_size (tuple): 目標サイズ (target_D, target_H, target_W)
        interpolation_mode (str): 'trilinear' (CT画像用) or 'nearest' (マスク用)
        gpu_id (int, optional): 使用するGPU ID
        
    Returns:
        np.ndarray: リサイズ後の3D配列
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but GPU normalization was requested.")

    # GPUデバイス選択
    if gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cuda")

    # 目標形状のタプル化
    if isinstance(output_size, int):
        target_shape = (output_size, output_size, output_size)
    else:
        target_shape = tuple(output_size)

    # Numpy -> Tensor変換 (Float32)
    # PyTorchのinterpolateは (N, C, D, H, W) 形式を要求するため次元を追加
    tensor = torch.from_numpy(input_array.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)

    # 補間モードの設定
    # align_cornersはtrilinearの場合True推奨 (座標ズレ防止)
    align_corners = True if interpolation_mode == 'trilinear' else None
    
    # リサイズ実行
    with torch.no_grad(): # 勾配計算不要
        resized_tensor = torch.nn.functional.interpolate(
            tensor,
            size=target_shape,
            mode=interpolation_mode,
            align_corners=align_corners
        )

    # Tensor -> Numpy変換
    return resized_tensor.squeeze(0).squeeze(0).cpu().numpy()

# ※ 必要であれば、ここに回転用の関数(gpu_rotate_3dなど)を残しても問題ありません
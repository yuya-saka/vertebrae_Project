
import os
import numpy as np
import nibabel as nib
import cv2
import argparse
from tqdm import tqdm

def apply_hu_window(image, min_hu, max_hu):
    """HU値をクリッピングし、[0, 255]の範囲に正規化する。"""
    channel = np.clip(image, min_hu, max_hu)
    channel = (channel - min_hu) / (max_hu - min_hu + 1e-6)
    return (channel * 255).astype(np.uint8)

def convert_nii_to_png(nii_path, output_path, hu_windows):
    """単一のNIFTIファイルを3チャンネルPNGに変換する。"""
    try:
        nii_img = nib.load(nii_path)
        img_data = nii_img.get_fdata()

        if img_data.ndim == 3:
            img_data = img_data[:, :, img_data.shape[2] // 2]

        channels = []
        for window_name in ['bone', 'soft_tissue', 'wide']:
            window = hu_windows[window_name]
            channel = apply_hu_window(img_data, window['min'], window['max'])
            channels.append(channel)
        
        img_3c = np.stack(channels, axis=-1)
        
        # OpenCVはBGR順で保存するため、RGBに変換
        img_bgr = cv2.cvtColor(img_3c, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(output_path, img_bgr)
        return True
    except Exception as e:
        print(f"Error converting {nii_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert NIFTI files to 3-channel PNG images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing NIFTI files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save PNG files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 固定のHUウィンドウ設定（YOLOプロジェクトの設定に合わせる）
    hu_windows_config = {
        'bone': {'min': 0, 'max': 1800},
        'soft_tissue': {'min': -100, 'max': 300},
        'wide': {'min': -200, 'max': 500}
    }

    nii_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.nii', '.nii.gz'))]

    for filename in tqdm(nii_files, desc="Converting NII to PNG"):
        nii_path = os.path.join(args.input_dir, filename)
        png_filename = os.path.splitext(filename)[0].replace('.nii', '') + '.png'
        output_path = os.path.join(args.output_dir, png_filename)
        convert_nii_to_png(nii_path, output_path, hu_windows_config)

    print(f"\nConversion complete. {len(nii_files)} files converted.")

if __name__ == "__main__":
    main()


import os
import numpy as np
import torch
import nibabel as nib
import cv2
from torch.utils.data import Dataset

class YOLODataset(Dataset):
    """
    YOLO形式のデータセットを読み込むPyTorch Datasetクラス。
    NIFTI画像と対応するテキストラベルを処理し、3チャンネルのHUウィンドウを適用する。
    """
    def __init__(self, label_dir, image_paths, image_size=256, hu_windows=None, transform=None):
        """
        Args:
            label_dir (str): ラベルファイル（.txt）が格納されているディレクトリ。
            image_paths (list[str]): 画像ファイル（.nii.gz）のフルパスのリスト。
            image_size (int): 出力画像のサイズ。
            hu_windows (dict): HUウィンドウ設定。例: {'bone': {'min': 0, 'max': 1800}, ...}
            transform (callable, optional): 画像に適用する変換。
        """
        self.label_dir = label_dir
        self.image_paths = image_paths
        self.image_size = image_size
        self.hu_windows = hu_windows
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # ---- 1. 画像ファイルの読み込み ----
        img_path = self.image_paths[idx]
        
        try:
            nii_img = nib.load(img_path)
            img_data = nii_img.get_fdata()
        except Exception as e:
            print(f"Error loading NIFTI file {img_path}: {e}")
            # Return a dummy tensor if file is corrupt
            return torch.zeros((3, self.image_size, self.image_size)), torch.zeros((0, 5))

        # 3Dデータの場合、中央のスライスを使用 (必要に応じて変更)
        if img_data.ndim == 3:
            img_data = img_data[:, :, img_data.shape[2] // 2]

        # ---- 2. HUウィンドウを適用して3チャンネル画像を生成 ----
        channels = []
        for window_name in ['bone', 'soft_tissue', 'wide']:
            window = self.hu_windows[window_name]
            channel = self._apply_hu_window(img_data, window['min'], window['max'])
            channels.append(channel)
        
        # チャンネルをスタックして3チャンネル画像を作成
        img_3c = np.stack(channels, axis=-1)

        # ---- 3. 画像のリサイズと正規化 ----
        img_resized = cv2.resize(img_3c, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        
        # (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_resized.astype(np.float32)).permute(2, 0, 1)

        # ---- 4. ラベルの読み込み ----
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0].replace('.nii', '') + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        # class, x_center, y_center, width, height
                        values = [float(p) for p in parts]
                        boxes.append(values)
        
        labels = torch.tensor(boxes, dtype=torch.float32)

        # ---- 5. Augmentation (オプション) ----
        if self.transform:
            # Note: Albumentations or similar library is needed for bbox-safe transforms
            # For now, we assume simple transforms that don't affect bboxes
            img_tensor = self.transform(img_tensor)

        return img_tensor, labels

    def _apply_hu_window(self, image, min_hu, max_hu):
        """HU値をクリッピングし、[0, 1]の範囲に正規化する。"""
        channel = np.clip(image, min_hu, max_hu)
        channel = (channel - min_hu) / (max_hu - min_hu + 1e-6) # ゼロ除算を防止
        return channel

if __name__ == '__main__':
    # --- このファイルが直接実行された場合のテストコード ---
    # 設定はダミーです。実際のパスと設定に合わせてください。
    
    # ダミーのNIFTIファイルとラベルファイルを作成
    DUMMY_ROOT = 'dummy_data'
    DUMMY_IMG_DIR = os.path.join(DUMMY_ROOT, 'images')
    DUMMY_LBL_DIR = os.path.join(DUMMY_ROOT, 'labels')
    os.makedirs(DUMMY_IMG_DIR, exist_ok=True)
    os.makedirs(DUMMY_LBL_DIR, exist_ok=True)

    # 1. ダミーNIFTIファイル作成 (HU値: -1000 to 2000)
    dummy_nii_path = os.path.join(DUMMY_IMG_DIR, 'test_slice.nii.gz')
    dummy_nii_data = np.random.randint(-1000, 2000, size=(128, 128), dtype=np.int16)
    dummy_nii_img = nib.Nifti1Image(dummy_nii_data, np.eye(4))
    nib.save(dummy_nii_img, dummy_nii_path)

    # 2. ダミーラベルファイル作成
    with open(os.path.join(DUMMY_LBL_DIR, 'test_slice.txt'), 'w') as f:
        f.write('0 0.5 0.5 0.2 0.3\n') # class x_center y_center width height

    # データセットの設定
    hu_windows_config = {
        'bone': {'min': 0, 'max': 1800},
        'soft_tissue': {'min': -100, 'max': 300},
        'wide': {'min': -200, 'max': 500}
    }

    # 画像パスのリストを作成
    image_paths = [os.path.abspath(dummy_nii_path)]

    # データセットのインスタンス化
    dataset = YOLODataset(
        label_dir=DUMMY_LBL_DIR,
        image_paths=image_paths,
        image_size=256,
        hu_windows=hu_windows_config
    )

    # 1つのサンプルを取得して形状を確認
    if len(dataset) > 0:
        img_tensor, labels_tensor = dataset[0]
        print(f"--- Dataset Test ---")
        print(f"Image tensor shape: {img_tensor.shape}")
        print(f"Labels tensor shape: {labels_tensor.shape}")
        print(f"Image tensor dtype: {img_tensor.dtype}")
        print(f"Labels tensor dtype: {labels_tensor.dtype}")
        print(f"Number of boxes: {len(labels_tensor)}")
        if len(labels_tensor) > 0:
            print(f"First box: {labels_tensor[0]}")

        # Check image value range
        print(f"Image tensor min value: {img_tensor.min():.4f}")
        print(f"Image tensor max value: {img_tensor.max():.4f}")
        
        # Check if the output is valid
        assert img_tensor.shape == (3, 256, 256), "Image shape is incorrect"
        assert labels_tensor.shape[0] > 0, "Labels are missing"
        assert labels_tensor.shape[1] == 5, "Label format is incorrect"
        assert img_tensor.min() >= 0.0 and img_tensor.max() <= 1.0, "Image values are not in [0, 1] range"
        print("\n✅ Test Passed!")
    else:
        print("❌ Test Failed: Dataset is empty.")

    # クリーンアップ
    import shutil
    shutil.rmtree(DUMMY_ROOT)

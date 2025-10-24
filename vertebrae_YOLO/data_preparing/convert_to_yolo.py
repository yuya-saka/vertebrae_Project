"""
YOLO形式データセット変換スクリプト（マルチインスタンス対応）

データセットの特性:
- 骨折マスクは値0～6で構成（0=背景, 1～6=各骨折インスタンス）
- 各非ゼロ値は異なる骨折領域を示す（専門医による個別アノテーション）
- 1椎体あたり最大5つの独立した骨折が存在

入力:
- data/slice_train/axial/ 
- data/slice_train/axial_mask/ 

[修正点]
1.  load_nifti_slice: 
    - [修正] `np.asarray(nii.dataobj)` から `nii.get_fdata(dtype=np.float32)` に変更。
    - `get_fdata()` を使うことで、アフィン行列に基づきデータを自動で再配向（傾き補正）します。
    - 手動の `np.fliplr` を削除。
2.  normalize_and_pad_image:
    - リサイズ処理を `PIL` から `scipy.ndimage.zoom` に変更。
    - `PIL` は `np.uint8` への変換が必要なため、-1000~3000のHU値（CT値）が
      失われていました。`zoom` は `float` 型のままリサイズできます。
    - パディングの値を `0` から `image.min()` に変更。
      (CT値の最小値（空気など）で埋めるため)
    - リサイズ後の厳密なサイズ担保ロジックを追加。
3.  convert_case:
    - [追加] マスク読み込み時、`get_fdata()` の補間による小数誤差を
      `np.round().astype(np.int32)` で補正し、整数性を担保します。
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
from scipy.ndimage import zoom


class YOLOConverter:
    """マスク画像をYOLO形式に変換するクラス"""

    def __init__(
        self,
        view: str,
        split: str,
        base_input_prefix: str = "data/slice",
        base_output_dir: str = "data/yolo_format",
        min_bbox_area: int = 36,
        target_size: tuple = (256, 256)
    ):
        """
        Args:
            view (str): データセットの視点 (例: 'axial', 'sagittal').
            split (str): データセットの分割 (例: 'train', 'test', 'val').
            base_input_prefix (str): 入力データのプレフィックス (例: 'data/slice').
            base_output_dir (str): 出力先のベースディレクトリ.
            min_bbox_area (int): 最小BBox面積閾値（ノイズ除去用）.
            target_size (tuple): 出力画像サイズ.
        """
        self.view = view
        self.split = split
        self.min_bbox_area = min_bbox_area
        self.target_size = target_size

        self.input_image_dir = Path(f"{base_input_prefix}_{split}") / view
        self.input_mask_dir = Path(f"{base_input_prefix}_{split}") / f"{view}_mask"

        self.output_dir = Path(base_output_dir)
        self.images_dir = self.output_dir / "images" / view / split
        self.labels_dir = self.output_dir / "labels" / view / split

        self.stats = {
            'total_slices': 0,
            'total_bboxes': 0,
            'slices_with_fracture': 0,
            'bboxes_per_slice': defaultdict(int),
            'bbox_sizes': [],
            'extreme_bboxes': [],
            'skipped_bboxes': 0,
        }

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output images will be saved to: {self.images_dir}")
        print(f"Output labels will be saved to: {self.labels_dir}")

    def load_nifti_slice(self, nii_path: Path) -> np.ndarray:
        """
        NIfTIファイルから2Dスライスを読み込み、自動再配向（傾き補正）を適用
        """
        nii = nib.load(str(nii_path))
        data = nii.get_fdata(dtype=np.float32)

        if data.ndim == 3 and data.shape[2] == 1:
            data = data[:, :, 0]
        elif data.ndim > 2:
            data = data[:, :, 0] if data.shape[2] == 1 else data.squeeze()

        return data

    def extract_bboxes_from_mask(self, mask: np.ndarray) -> list:
        """
        マスク画像から複数のBBoxを抽出（マルチインスタンス対応）
        """
        bboxes = []
        height, width = mask.shape
        
        MIN_TARGET_SIZE = 10

        for mask_value in range(1, 7):
            binary_mask = (mask == mask_value)

            if not binary_mask.any():
                continue

            y_coords, x_coords = np.where(binary_mask)
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()

            bbox_width_px = x_max - x_min + 1
            bbox_height_px = y_max - y_min + 1
            
            x_center_px = (x_min + x_max) / 2
            y_center_px = (y_min + y_max) / 2
            
            if bbox_width_px < MIN_TARGET_SIZE:
                bbox_width_px = MIN_TARGET_SIZE
            
            if bbox_height_px < MIN_TARGET_SIZE:
                bbox_height_px = MIN_TARGET_SIZE
            
            area = bbox_width_px * bbox_height_px

            if not self._is_valid_bbox(bbox_width_px, bbox_height_px, area):
                self.stats['skipped_bboxes'] += 1
                continue

            x_center = x_center_px / width
            y_center = y_center_px / height
            bbox_width = bbox_width_px / width
            bbox_height = bbox_height_px / height

            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                    0 < bbox_width <= 1 and 0 < bbox_height <= 1):
                self.stats['skipped_bboxes'] += 1
                continue

            bboxes.append({
                'class': 0,
                'x_center': x_center,
                'y_center': y_center,
                'width': bbox_width,
                'height': bbox_height,
                'area': area,
            })

            self.stats['bbox_sizes'].append(area)

            if area <= 100 or area > 10000:
                self.stats['extreme_bboxes'].append({
                    'area': area,
                    'size': (bbox_width_px, bbox_height_px)
                })

        return bboxes

    def _is_valid_bbox(self, width_px: int, height_px: int, area: int) -> bool:
        """BBoxの品質チェック"""
        if area < self.min_bbox_area:
            return False
        if width_px <= 0 or height_px <= 0:
            return False
        if min(width_px, height_px) == 0: return False
        aspect_ratio = max(width_px, height_px) / min(width_px, height_px)
        if aspect_ratio > 20:
            return False
        return True

    def save_yolo_annotation(self, bboxes: list, output_path: Path):
        """YOLO形式でアノテーションを保存"""
        with open(output_path, 'w') as f:
            for bbox in bboxes:
                f.write(f"{bbox['class']} {bbox['x_center']:.6f} {bbox['y_center']:.6f} "
                        f"{bbox['width']:.6f} {bbox['height']:.6f}\n")

    def normalize_and_pad_image(self, image: np.ndarray) -> np.ndarray:
        """
        画像をリサイズまたはパディングでターゲットサイズに調整 (HU値を保持)
        """
        h, w = image.shape[:2]
        bg_value = image.min()

        if h == self.target_size[0] and w == self.target_size[1]:
            return image

        if h > self.target_size[0] or w > self.target_size[1]:
            zoom_h = self.target_size[0] / h
            zoom_w = self.target_size[1] / w
            
            resized_image = zoom(
                image, 
                (zoom_h, zoom_w), 
                order=1, # Bilinear interpolation
                mode='constant', 
                cval=bg_value
            )
            
            rh, rw = resized_image.shape
            th, tw = self.target_size
            
            final_image = np.full(self.target_size, bg_value, dtype=image.dtype)
            
            h_crop = min(rh, th)
            w_crop = min(rw, tw)
            
            h_offset = (th - h_crop) // 2
            w_offset = (tw - w_crop) // 2
            
            final_image[h_offset:h_offset+h_crop, w_offset:w_offset+w_crop] = resized_image[0:h_crop, 0:w_crop]
            
            return final_image
            
        padded = np.full(self.target_size, bg_value, dtype=image.dtype)
        pad_h = (self.target_size[0] - h) // 2
        pad_w = (self.target_size[1] - w) // 2
        padded[pad_h:pad_h+h, pad_w:pad_w+w] = image
        return padded

    # [追加] マスク専用のリサイズ・パディング関数
    def normalize_and_pad_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        マスクをリサイズまたはパディングでターゲットサイズに調整（最近傍補間）
        """
        h, w = mask.shape[:2]
        bg_value = 0  # マスクの背景は常に0

        if h == self.target_size[0] and w == self.target_size[1]:
            return mask

        if h > self.target_size[0] or w > self.target_size[1]:
            zoom_h = self.target_size[0] / h
            zoom_w = self.target_size[1] / w
            
            # order=0 は最近傍補間（Nearest Neighbor）。マスクの値が崩れないようにする
            resized_mask = zoom(
                mask, 
                (zoom_h, zoom_w), 
                order=0,  # Nearest neighbor interpolation
                mode='constant', 
                cval=bg_value
            )
            
            rh, rw = resized_mask.shape
            th, tw = self.target_size
            
            final_mask = np.full(self.target_size, bg_value, dtype=mask.dtype)
            
            h_crop = min(rh, th)
            w_crop = min(rw, tw)
            
            h_offset = (th - h_crop) // 2
            w_offset = (tw - w_crop) // 2
            
            final_mask[h_offset:h_offset+h_crop, w_offset:w_offset+w_crop] = resized_mask[0:h_crop, 0:w_crop]
            
            return final_mask
            
        padded = np.full(self.target_size, bg_value, dtype=mask.dtype)
        pad_h = (self.target_size[0] - h) // 2
        pad_w = (self.target_size[1] - w) // 2
        padded[pad_h:pad_h+h, pad_w:pad_w+w] = mask
        return padded

    def convert_case(self, case_name: str):
        """1症例のデータをYOLO形式に変換"""
        case_image_dir = self.input_image_dir / case_name
        case_mask_dir = self.input_mask_dir / case_name

        if not case_image_dir.exists() or not case_mask_dir.exists():
            print(f"Warning: {case_name} not found in both image and mask directories")
            return

        vertebra_dirs = sorted([d for d in case_image_dir.iterdir() if d.is_dir()])

        for vertebra_dir in vertebra_dirs:
            vertebra_name = vertebra_dir.name
            vertebra_mask_dir = case_mask_dir / vertebra_name

            if not vertebra_mask_dir.exists():
                continue

            slice_files = sorted(vertebra_dir.glob("slice_*.nii"))

            for slice_file in slice_files:
                slice_idx = slice_file.stem.split('_')[1]
                mask_file = vertebra_mask_dir / f"mask_{slice_idx}.nii"

                if not mask_file.exists():
                    continue

                try:
                    image_original = self.load_nifti_slice(slice_file)
                    mask_float = self.load_nifti_slice(mask_file)
                    mask_original = np.round(mask_float).astype(np.int32)
                except Exception as e:
                    print(f"Error loading {slice_file}: {e}")
                    continue

                # [修正] 画像とマスクの両方を先にリサイズ・パディングする
                image_processed = self.normalize_and_pad_image(image_original)
                mask_processed = self.normalize_and_pad_mask(mask_original)

                # [修正] 変形後のマスクからBBoxを抽出する
                bboxes = self.extract_bboxes_from_mask(mask_processed)

                output_name = f"{case_name}_{vertebra_name}_slice_{slice_idx}"
                
                # NIfTI形式で保存
                output_nii_path = self.images_dir / f"{output_name}.nii"
                nii_img = nib.Nifti1Image(image_processed.astype(np.float32), np.eye(4))
                nib.save(nii_img, str(output_nii_path))

                # アノテーション保存
                self.save_yolo_annotation(bboxes, self.labels_dir / f"{output_name}.txt")

                # 統計更新
                self.stats['total_slices'] += 1
                if bboxes:
                    self.stats['slices_with_fracture'] += 1
                    self.stats['total_bboxes'] += len(bboxes)
                    self.stats['bboxes_per_slice'][len(bboxes)] += 1

    def convert_all(self):
        """全症例を変換"""
        if not self.input_image_dir.exists():
            print(f"Error: Input image directory not found at {self.input_image_dir}")
            return
            
        case_dirs = sorted([d.name for d in self.input_image_dir.iterdir() if d.is_dir()])

        if not case_dirs:
            print(f"No case directories found in {self.input_image_dir}")
            return

        print(f"Found {len(case_dirs)} cases for {self.split} set.")
        print(f"Converting to YOLO format...")

        for case_name in tqdm(case_dirs, desc=f"Converting {self.split} cases"):
            self.convert_case(case_name)

        print("\nConversion completed!")
        self.print_statistics()

    def print_statistics(self):
        """統計情報を出力"""
        print("\n" + "="*60)
        print(f"YOLO Dataset Statistics ({self.split} - {self.view})")
        print("="*60)

        if self.stats['total_slices'] == 0:
            print("No slices were processed. Please check input directories.")
            print("="*60)
            return

        print(f"\n[Dataset Overview]")
        print(f"   Total slices: {self.stats['total_slices']}")
        print(f"   Slices with fracture: {self.stats['slices_with_fracture']} "
              f"({self.stats['slices_with_fracture']/self.stats['total_slices']*100:.1f}%)")
        print(f"   Slices without fracture: {self.stats['total_slices'] - self.stats['slices_with_fracture']}")
        print(f"   Total bboxes: {self.stats['total_bboxes']}")
        print(f"   Skipped bboxes (quality filter): {self.stats['skipped_bboxes']}")

        print(f"\n[BBoxes per Slice Distribution]")
        for num_bbox, count in sorted(self.stats['bboxes_per_slice'].items()):
            print(f"   {num_bbox} bbox(es): {count} slices")

        if self.stats['bbox_sizes']:
            bbox_sizes = np.array(self.stats['bbox_sizes'])
            print(f"\n[BBox Size Statistics (in pixels²)]")
            print(f"   Min: {bbox_sizes.min()}")
            print(f"   Max: {bbox_sizes.max()}")
            print(f"   Median: {np.median(bbox_sizes):.1f}")
            print(f"   Mean: {bbox_sizes.mean():.1f} ± {bbox_sizes.std():.1f}")

        if self.stats['extreme_bboxes']:
            print(f"\n[Extreme BBoxes for Manual Review]")
            print(f"   Count: {len(self.stats['extreme_bboxes'])}")
            extreme_samples = self.stats['extreme_bboxes'][:10]
            for i, bbox in enumerate(extreme_samples, 1):
                print(f"     {i}. Area={bbox['area']}px², Size={bbox['size']}")

        print("\n" + "="*60)


def main():
    """メイン関数"""
    VIEW = 'axial'
    SPLIT = 'train'
    BASE_INPUT = 'data/slice'
    BASE_OUTPUT = 'data/yolo_format'

    print(f"Starting conversion for: VIEW={VIEW}, SPLIT={SPLIT}")
    
    converter = YOLOConverter(
        view=VIEW,
        split=SPLIT,
        base_input_prefix=BASE_INPUT,
        base_output_dir=BASE_OUTPUT,
        min_bbox_area=36,
        target_size=(256, 256)
    )

    converter.convert_all()


if __name__ == "__main__":
    main()

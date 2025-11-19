import os
import glob
import re
import time
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from apply_normalization import apply_normalization

# 実行コマンド
# uv run python main.py --gpu 0or1or2

# テスト用
# uv run python main.py --gpu 1 --start-id 1001 --end-id 1003

# ==========================================
# 設定 (Configuration)
# ==========================================
# 物理的な切り出しサイズ (mm) -> これを128voxelにすることで1.0mm等方性にする
TARGET_FOV_MM = (128.0, 128.0, 128.0) 
TARGET_SIZE = (128, 128, 128)

# CT値のウィンドウ設定 (Bone Window)
HU_MIN = 0
HU_MAX = 1900

current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent.parent  #3階層上がプロジェクトルート
# パス設定 (実行環境に合わせて変更可能)
INPUT_ROOT = PROJECT_ROOT / "input_nii/"   # 元データの場所
OUTPUT_ROOT = PROJECT_ROOT / "data/3d_data/"  # 出力先

# 椎体定義
VERT_IDS_SEG = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
VERT_IDS_ALT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
VERT_NAMES = ["L5", "L4", "L3", "L2", "L1", "T12", "T11", "T10", "T9", "T8", "T7", "T6", "T5", "T4"]

# ==========================================
# 処理関数 (Helper Functions)
# ==========================================

def normalize_intensity(vol):
    """CT値をBone Windowでクリップし、0.0-1.0に正規化"""
    vol = np.clip(vol, HU_MIN, HU_MAX)
    vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)
    return vol.astype(np.float32)

def create_weak_label(gt_mask):
    """詳細なマスクから3Dバウンディングボックス（弱ラベル）を生成"""
    weak_mask = np.zeros_like(gt_mask)
    coords = np.argwhere(gt_mask > 0)
    
    if coords.size > 0:
        # 存在する範囲の最小・最大を取得
        min_z, min_y, min_x = coords.min(axis=0)
        max_z, max_y, max_x = coords.max(axis=0)
        # その範囲を箱状に1で埋める
        weak_mask[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = 1.0
        
    return weak_mask.astype(np.float32)

def robust_crop(data, center_idx, crop_size_vox):
    """
    画像範囲外を考慮して安全に切り出す関数（ゼロパディング付き）。
    物理座標計算で求めたボクセル数分だけ切り出す。
    
    Args:
        data: 3D numpy array (Z, Y, X)
        center_idx: 重心座標 (z, y, x)
        crop_size_vox: 切り出しサイズ (dz, dy, dx)
    """
    z, y, x = center_idx
    dz, dy, dx = crop_size_vox
    
    # 切り出し開始・終了位置の計算
    z_start = int(z - dz / 2)
    y_start = int(y - dy / 2)
    x_start = int(x - dx / 2)
    
    z_end = z_start + int(dz)
    y_end = y_start + int(dy)
    x_end = x_start + int(dx)
    
    D, H, W = data.shape
    
    # 有効範囲の計算
    valid_z_start = max(0, z_start)
    valid_y_start = max(0, y_start)
    valid_x_start = max(0, x_start)
    
    valid_z_end = min(D, z_end)
    valid_y_end = min(H, y_end)
    valid_x_end = min(W, x_end)
    
    # データ切り出し
    crop = data[valid_z_start:valid_z_end, valid_y_start:valid_y_end, valid_x_start:valid_x_end]
    
    # パディング量の計算 (足りない分)
    pad_z_pre = max(0, -z_start)
    pad_y_pre = max(0, -y_start)
    pad_x_pre = max(0, -x_start)
    
    pad_z_post = max(0, z_end - D)
    pad_y_post = max(0, y_end - H)
    pad_x_post = max(0, x_end - W)
    
    # パディング適用 (不足分を0埋め)
    if any([pad_z_pre, pad_z_post, pad_y_pre, pad_y_post, pad_x_pre, pad_x_post]):
        crop = np.pad(crop, (
            (pad_z_pre, pad_z_post),
            (pad_y_pre, pad_y_post),
            (pad_x_pre, pad_x_post)
        ), mode='constant', constant_values=0)
        
    return crop

def save_nii_debug(vol, path):
    """可視化確認用にNIfTIとして保存（Affineは単位行列）"""
    # 1.0mm等方性になったため、Affineは単位行列でOK（方向確認用）
    affine = np.eye(4) 
    nii = nib.Nifti1Image(vol, affine)
    nib.save(nii, path)

def process_subject(sbj_id, inp_path, seg_path, ans_path, gpu_id=0):
    """1症例に対する処理フロー"""
    print(f"[PROCESSING] ID: {sbj_id}")
    
    # NIfTIロード
    inp_nii = nib.load(inp_path)
    seg_nii = nib.load(seg_path)
    ans_nii = nib.load(ans_path)
    
    # データ取得 (nibabelは通常 (X, Y, Z) だが get_fdata() でNumPyになると (X, Y, Z) のまま)
    # ※ 注意: 医用画像処理では (Z, Y, X) で扱うことが多いが、
    # nibabelのロード直後は (dim0, dim1, dim2) = (Sagittal, Coronal, Axial) 順などデータによる。
    # ここでは、単純に読み込んだ配列の軸順序に従って処理を行い、
    # Spacing情報もそれに対応させる。
    inp = inp_nii.get_fdata()
    seg = seg_nii.get_fdata()
    ans = ans_nii.get_fdata()
    
    # 解像度(Spacing)を取得 (x, y, z)
    spacing = inp_nii.header.get_zooms() 
    
    # 【重要】物理サイズ(128mm)を満たすのに必要なボクセル数を計算
    # inp.shape が (X, Y, Z) なら spacing も (sx, sy, sz) で対応しているはず
    req_voxels = [TARGET_FOV_MM[i] / spacing[i] for i in range(3)]
    req_voxels = tuple(map(int, req_voxels))
    
    print(f"  Spacing: {spacing}")
    print(f"  Req Voxels (128mm): {req_voxels} -> Target: {TARGET_SIZE}")

    # 保存先ディレクトリ定義
    dirs = {
        "vae": os.path.join(OUTPUT_ROOT, "train_vae"),
        "det_vol": os.path.join(OUTPUT_ROOT, "train_det/vol"),
        "det_weak": os.path.join(OUTPUT_ROOT, "train_det/mask_weak_label"),
        "det_gt": os.path.join(OUTPUT_ROOT, "train_det/mask_gt_label"),
        "vis_vol": os.path.join(OUTPUT_ROOT, "visualization/vol"),
        "vis_weak": os.path.join(OUTPUT_ROOT, "visualization/mask_weak"),
        "vis_gt": os.path.join(OUTPUT_ROOT, "visualization/mask_gt"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 各椎体(T4-L5)をループ処理
    count_processed = 0
    for i, v_id in enumerate(VERT_IDS_SEG):
        v_name = VERT_NAMES[i]
        
        # セグメンテーションから椎体領域を特定
        coords = np.argwhere((seg == v_id) | (seg == VERT_IDS_ALT[i]))
        
        if coords.size == 0:
            continue # その椎体がなければスキップ
            
        # 重心 (Center of Mass) を計算
        center = coords.mean(axis=0)
        
        # 1. Robust Crop (パディング付き切り出し)
        # 物理的に128mm分を切り出す
        crop_inp = robust_crop(inp, center, req_voxels)
        crop_ans = robust_crop(ans, center, req_voxels)
        
        # 2. GPU Resize (128x128x128へ) -> 結果的に1.0mm等方性になる
        # CT画像: Trilinear補間
        vol_resized = apply_normalization(
            crop_inp, 
            output_size=TARGET_SIZE, 
            interpolation_mode='trilinear', 
            gpu_id=gpu_id
        )
        
        # マスク画像: Nearest補間 (値が変わらないように)
        ans_resized = apply_normalization(
            crop_ans, 
            output_size=TARGET_SIZE, 
            interpolation_mode='nearest', 
            gpu_id=gpu_id
        )
        
        # 3. Normalization (Bone Window適用 & 0-1正規化)
        vol_norm = normalize_intensity(vol_resized)
        
        # 4. Weak Label Generation (詳細マスクから箱を作る)
        weak_label = create_weak_label(ans_resized)
        
        # --- 保存処理 ---
        base_name = f"{sbj_id}_{v_name}"
        has_fracture = np.max(ans_resized) > 0
        
        # A. 学習用 (.npy) - 高速読み込み用
        np.save(os.path.join(dirs["det_vol"], f"vol_{base_name}.npy"), vol_norm)
        np.save(os.path.join(dirs["det_weak"], f"weak_{base_name}.npy"), weak_label)
        np.save(os.path.join(dirs["det_gt"], f"gt_{base_name}.npy"), ans_resized)
        
        # 骨折がない場合のみ、VAE用(正常学習用)に保存
        if not has_fracture:
            np.save(os.path.join(dirs["vae"], f"vol_{base_name}.npy"), vol_norm)
            
        # B. 可視化確認用 (.nii.gz) - 3D Slicer等で確認用
        save_nii_debug(vol_norm, os.path.join(dirs["vis_vol"], f"vol_{base_name}.nii.gz"))
        save_nii_debug(weak_label, os.path.join(dirs["vis_weak"], f"weak_{base_name}.nii.gz"))
        if has_fracture:
             save_nii_debug(ans_resized, os.path.join(dirs["vis_gt"], f"gt_{base_name}.nii.gz"))

        count_processed += 1
        
    print(f"  Processed {count_processed} vertebrae.")

# ==========================================
# メイン実行ブロック
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process 3D vertebrae data for VAE & Detection')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--start-id', type=int, default=1001, help='Start Subject ID')
    parser.add_argument('--end-id', type=int, default=1100, help='End Subject ID')
    args = parser.parse_args()
    
    print(f"[START] Processing IDs {args.start_id} to {args.end_id}")
    start_time = time.time()
    
    # input_nii 以下の inp*.nii.gz を検索
    search_pattern = os.path.join(INPUT_ROOT, "inp*.nii.gz")
    files = sorted(glob.glob(search_pattern))

    print(f"Found {len(files)} input files in {INPUT_ROOT}")

    processed_count = 0
    for fpath in files:
       # ファイル名からID抽出 (例: inp1001.nii.gz -> 1001)
       basename = os.path.basename(fpath)
       match = re.search(r'(\d{4,})', basename)
       if not match:
          continue

       sbj_id = int(match.group(1))

       # 指定範囲外ならスキップ
       if not (args.start_id <= sbj_id < args.end_id):
          continue

       # 対応する seg, ans ファイルのパスを推定 (.nii)
       seg_path = os.path.join(INPUT_ROOT, f"seg{sbj_id}.nii")
       ans_path = os.path.join(INPUT_ROOT, f"ans{sbj_id}.nii")

       # ファイルセットが揃っているか確認
       if os.path.exists(seg_path) and os.path.exists(ans_path):
           try:
               process_subject(sbj_id, fpath, seg_path, ans_path, gpu_id=args.gpu)
               processed_count += 1
           except Exception as e:
               print(f"[ERROR] Failed processing ID {sbj_id}: {e}")
       else:
           pass

            
    total_time = time.time() - start_time
    print(f"\n[DONE] Processed {processed_count} subjects in {total_time:.1f} seconds.")
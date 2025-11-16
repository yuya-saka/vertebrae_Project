from mk_train_img import mk_train_img
import glob
import os
import re
import argparse

###
###　課題：　正規化後のサイズを統一する。　答え矩形リストを出力する。
###  椎体４個で５分くらい掛かる
####   複数ファイルで実行する。　出力フォルダを分ける。
##
## 答えリストのフォーマットを合わせる
## 集計プログラムのチェック
## TotalSegの実力チェック
##　学習プログラムを作る
##　推論プログラムを作る
##

# uv run python main.py --gpu 0or1or2 --planes standard
# uv run python main.py --gpu 0 --planes cross
# uv run python main.py --gpu 1 --planes all

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process vertebrae CT scans and create training slices')
parser.add_argument('--gpu', type=int, default=None, help='GPU device ID to use (default: auto-detect from CUDA_VISIBLE_DEVICES or use GPU 0)')
parser.add_argument('--start-id', type=int, default=1001, help='Starting subject ID (default: 1001)')
parser.add_argument('--end-id', type=int, default=1100, help='Ending subject ID (exclusive, default: 1100)')
parser.add_argument('--planes', type=str, default='standard',
                    choices=['standard', 'cross', 'all'],
                    help='Planes to process: standard (0-2: Sagittal/Coronal/Axial), cross (3-8: 6 cross planes), all (0-8: all 9 planes)')
args = parser.parse_args()

# Display GPU configuration
if args.gpu is not None:
    print(f"[CONFIG] Using GPU: {args.gpu}")
else:
    import os as os_env
    cuda_visible = os_env.environ.get('CUDA_VISIBLE_DEVICES', '0')
    print(f"[CONFIG] Using GPU from CUDA_VISIBLE_DEVICES: {cuda_visible} (or default GPU 0)")

# Display plane configuration
plane_descriptions = {
    'standard': '3 standard planes (0-2: Sagittal, Coronal, Axial)',
    'cross': '6 cross planes (3-8: SgCr1, SgCr2, CrAx1, CrAx2, AxSg1, AxSg2)',
    'all': 'All 9 planes (0-8: standard + cross)'
}
print(f"[CONFIG] Processing planes: {args.planes} - {plane_descriptions[args.planes]}")

inp_nii = "/mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/input_nii/"
out_path = "/mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/data/output/"

# ファイルパスをIDをキーにした辞書にまとめる
file_map = {}
file_patterns = {"inp": "inp*", "ans": "ans*", "seg": "seg*"}

for key, pattern in file_patterns.items():
    for filepath in glob.glob(os.path.join(inp_nii, pattern)):
        # ファイル名から数字IDを抽出 (例: inp1001.nii -> 1001)
        match = re.search(r'(\d{4,})', os.path.basename(filepath))
        if match:
            file_id = int(match.group(1))
            if file_id not in file_map:
                file_map[file_id] = {}
            file_map[file_id][key] = filepath

print("Found file sets for IDs:", sorted(file_map.keys()))
print(f"[CONFIG] Processing IDs from {args.start_id} to {args.end_id-1}")

for i in range(args.start_id, args.end_id):
    if i in file_map and all(k in file_map[i] for k in ["inp", "ans", "seg"]):
        paths = file_map[i]
        inp_path = paths["inp"]
        ans_path = paths["ans"]
        seg_path = paths["seg"]

        print(f"\n{'='*60}")
        print(f"Processing ID: {i}")
        print(f"{'='*60}")
        print(f"Input:  {inp_path}")
        print(f"Answer: {ans_path}")
        print(f"Segmentation: {seg_path}")

        an_opath = os.path.join(out_path, "ans_nii")
        cn_opath = os.path.join(out_path, "ct_nii")
        si_opath = os.path.join(out_path, "slice_image")
        si_opath_ans = os.path.join(out_path, "slice_image_ans")
        si_opath_rect = os.path.join(out_path, "slice_image_rect")
        al_opath = os.path.join(out_path, "ans_list")

        os.makedirs(out_path, exist_ok=True)
        os.makedirs(an_opath, exist_ok=True)
        os.makedirs(cn_opath, exist_ok=True)
        os.makedirs(si_opath, exist_ok=True)
        os.makedirs(si_opath_ans, exist_ok=True)
        os.makedirs(si_opath_rect, exist_ok=True)
        os.makedirs(al_opath, exist_ok=True)

        mk_train_img(i, inp_nii, inp_path, seg_path, ans_path,
                     os.path.join(an_opath, f"ans_ni{i:04}_"),
                     os.path.join(cn_opath, f"seg_ni{i:04}_"),
                     si_opath,
                     si_opath_ans,
                     si_opath_rect,
                     os.path.join(al_opath, f"ans_li{i:04}_"),
                     os.path.join(al_opath, "ans_li_"),
                     gpu_id=args.gpu,
                     plane_mode=args.planes)
    else:
        # Optional: print a message if files for an ID are missing
        # print(f"Skipping ID {i}: missing one or more required files.")
        pass

print(f"\n{'='*60}")
print("[INFO] All processing complete!")
print(f"{'='*60}")
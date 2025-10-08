import os
import re
import logging
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import numpy as np
import nibabel as nib

"""
高速版 NIfTI カットスクリプト
改善点
* nibabel の遅延読み込み (mmap) で必要領域のみをロード
* ProcessPoolExecutor によるマルチプロセス並列化
* ファイルリストを辞書キャッシュして線形探索を 1 回に集約
* dtype を保持して余分な float64 変換を回避
"""

# ロギング設定
def setup_logger(log_dir: str = "./logs"):
    """ロガーの初期化"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"nifti_cut_{timestamp}.log")
    
    # ロガーの設定
    logger = logging.getLogger("NIfTI_Cutter")
    logger.setLevel(logging.DEBUG)
    
    # ファイルハンドラー
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    # コンソールハンドラー
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # フォーマッター
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"ログファイル: {log_file}")
    return logger

logger = setup_logger()

# パース関連
# -----------------------------
def parse_cut_file(file_path: str):
    """cut_li*.txt を辞書のリストに変換"""
    logger.debug(f"カットファイル読み込み開始: {file_path}")
    cut_info = []
    try:
        with open(file_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip().rstrip(',')
                if not line:
                    continue
                vals = list(map(int, line.split(',')))
                cut_info.append({
                    "vertebra_num": vals[0],
                    "slb": vals[1],
                    "slb2": vals[2],
                    "slice_count": vals[3],
                    "base_size": vals[4],
                    "x_range": vals[5:7],
                    "y_range": vals[7:9],
                    "z_range": vals[9:11]
                })
        logger.info(f"カットファイル読み込み完了: {file_path} ({len(cut_info)}件)")
    except Exception as e:
        logger.error(f"カットファイル読み込みエラー: {file_path} - {e}")
        raise
    return cut_info

# -----------------------------
# NIfTI 切り出し
# -----------------------------
def _clip(start: int, end: int, maxv: int, margin: int = 1):
    """範囲を margin 分だけ広げてクリップ"""
    return max(start - margin, 0), min(end + margin, maxv)

def apply_cut_to_nifti(args):
    """ワーカー関数: 指定領域を切り出して保存"""
    input_path, output_path, cut_info = args
    try:
        img = nib.load(input_path, mmap=True)  # lazy + memmap
        xs, xe = _clip(*cut_info["x_range"], img.shape[0])
        ys, ye = _clip(*cut_info["y_range"], img.shape[1])
        zs, ze = _clip(*cut_info["z_range"], img.shape[2])
        
        # 必要領域だけを実体化
        cut_data = np.asanyarray(img.dataobj[xs:xe, ys:ye, zs:ze])
        nib.save(nib.Nifti1Image(cut_data, img.affine, img.header), output_path)
        
        logger.debug(f"切り出し完了: {os.path.basename(output_path)}")
        return output_path, True, None
    except Exception as e:
        logger.error(f"切り出しエラー: {input_path} -> {output_path} - {e}")
        return output_path, False, str(e)

# -----------------------------
# ディレクトリ処理
# -----------------------------
def process_directory(input_dir: str, output_base_dir: str, max_workers: int | None = None):
    logger.info(f"処理開始: {input_dir}")
    logger.info(f"出力先: {output_base_dir}")
    
    if not os.path.exists(input_dir):
        logger.error(f"入力ディレクトリが見つかりません: {input_dir}")
        return
    
    files = os.listdir(input_dir)
    cut_files = [f for f in files if f.startswith("cut_li") and f.endswith(".txt")]
    nii_files = [f for f in files if f.endswith(".nii") or f.endswith(".nii.gz")]
    
    logger.info(f"カットファイル: {len(cut_files)}個")
    logger.info(f"NIfTIファイル: {len(nii_files)}個")
    
    # 数字 -> 対応する nii ファイル群
    file_map: dict[str, list[str]] = defaultdict(list)
    for f in nii_files:
        match = re.search(r"\d+", f)
        if match:
            num = match.group()
            file_map[num].append(f)
    
    # 並列実行用ジョブ作成
    jobs: list[tuple[str, str, dict]] = []
    for cut_file in cut_files:
        match = re.search(r"\d+", cut_file)
        if not match:
            logger.warning(f"数字が見つかりません: {cut_file}")
            continue
            
        number = match.group()
        cut_info_list = parse_cut_file(os.path.join(input_dir, cut_file))
        
        for cut_info in cut_info_list:
            for nii_file in file_map[number]:
                in_path = os.path.join(input_dir, nii_file)
                out_dir = os.path.join(output_base_dir, f"inp{number}", str(cut_info["vertebra_num"]))
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"cut_{nii_file}")
                jobs.append((in_path, out_path, cut_info))
    
    logger.info(f"ジョブ総数: {len(jobs)}個")
    worker_count = max_workers or os.cpu_count()
    logger.info(f"ワーカー数: {worker_count}")
    
    # マルチプロセス実行
    success_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=worker_count) as ex:
        futures = [ex.submit(apply_cut_to_nifti, j) for j in jobs]
        
        for i, future in enumerate(as_completed(futures), 1):
            output_path, success, error = future.result()
            if success:
                success_count += 1
            else:
                error_count += 1
            
            if i % 10 == 0 or i == len(jobs):
                logger.info(f"進捗: {i}/{len(jobs)} (成功: {success_count}, エラー: {error_count})")
    
    logger.info(f"処理完了 - 成功: {success_count}, エラー: {error_count}")

# -----------------------------
# エントリポイント
# -----------------------------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("NIfTI カット処理開始")
    logger.info("=" * 60)

    current_file = Path(__file__).resolve()
    PROJECT_ROOT = current_file.parent.parent.parent #3階層上がプロジェクトルート
    train_dir = PROJECT_ROOT / "data/train"
    output_train_dir = PROJECT_ROOT / "data/processed_train"

    try:
        process_directory(train_dir, output_train_dir)
        logger.info("=" * 60)
        logger.info("すべてのカット処理が完了しました")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"処理中に致命的なエラーが発生しました: {e}", exc_info=True)
        raise
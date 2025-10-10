### 最初にどの軸をスライスするか、またどこのフォルダにつくるのか、要確認!!!!!!!!!!!!!!!!!!!!!

"""
NIfTI マスクスライス抽出スクリプト（テストデータ用）
機能:
* アノテーション画像から軸方向のマスクスライスを抽出
* CT画像のスライスと1対1で対応
* data/slice_test/axial_mask/に保存
"""
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict

# ロギング設定
def setup_logger(log_dir: str = "./logs"):
    """ロガーの初期化"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"mask_slice_extraction_test_{timestamp}.log"

    # ロガーの設定
    logger = logging.getLogger("MaskSliceExtractor_Test")
    logger.setLevel(logging.DEBUG)

    # 既存のハンドラーをクリア
    logger.handlers.clear()

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

# ----------------------- 設定 -----------------------
# 実行前に必ず確認する
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent.parent.parent  # 4階層上がプロジェクトルート
input_base_dir  = PROJECT_ROOT / "data/processed_test"
output_base_dir = PROJECT_ROOT / "data/slice_test/axial_mask"
target_cases    = list(range(1003, 1085))

def process_vertebra_directory(
    vertebra_dir: Path,
    case_number: int,
    inp_dir_name: str,
    annotation_img,
    output_base_dir: Path
) -> List[Dict]:
    """椎体ディレクトリからマスクスライスを抽出"""
    folder_results = []

    try:
        # アノテーションデータを二値化（0以外を1に）
        annotation_data = annotation_img.get_fdata()
        annotation_bin = (annotation_data != 0).astype(np.uint8)

        axis_name = "axial"
        H, W, D = map(int, annotation_bin.shape)

        logger.debug(f"  椎体: {vertebra_dir.name}, 形状: ({H}, {W}, {D})")

        output_dir = output_base_dir / inp_dir_name / vertebra_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        fracture_slice_count = 0

        # スライス毎に処理
        for z in range(D):
            mask_slice = annotation_bin[:, :, z].astype(np.uint8)
            has_fract = bool(mask_slice.any())
            fracture_label = int(has_fract)

            if has_fract:
                fracture_slice_count += 1

            # マスクスライスをNIfTI形式で保存
            mask_nifti = nib.Nifti1Image(mask_slice, affine=annotation_img.affine)
            mask_path = output_dir / f"mask_{z:03d}.nii"
            nib.save(mask_nifti, str(mask_path))

            folder_results.append({
                "MaskPath": str(mask_path),
                "Vertebra": vertebra_dir.name,
                "SliceIndex": z,
                "Fracture_Label": fracture_label,
                "Case": case_number,
                "Axis": axis_name,
                "Mask_H": H,
                "Mask_W": W,
                "Mask_D": D,
                "InputMaskPath": str(vertebra_dir / f"cut_ans{case_number}.nii"),
            })

        logger.info(f"    {vertebra_dir.name}: {D}マスクスライス抽出 (骨折: {fracture_slice_count}スライス)")

    except Exception as e:
        logger.error(f"  椎体処理エラー: {vertebra_dir.name} - {e}", exc_info=True)

    return folder_results

def process_case_directory(inp_dir: Path, output_base_dir: Path, target_cases: List[int]) -> bool:
    """ケースディレクトリを処理"""
    if not inp_dir.is_dir():
        return False

    try:
        case_number = int(inp_dir.name[3:])
    except ValueError:
        logger.warning(f"ディレクトリ名から番号を抽出できません: {inp_dir.name}")
        return False

    if case_number not in target_cases:
        logger.debug(f"スキップ: {inp_dir.name} (対象外)")
        return False

    logger.info(f"処理開始: {inp_dir.name} (Case {case_number})")

    folder_results = []
    vertebra_count = 0
    skipped_count = 0

    for vertebra_dir in sorted(inp_dir.iterdir()):
        if not vertebra_dir.is_dir():
            continue

        annotation_path = vertebra_dir / f"cut_ans{case_number}.nii"

        if not annotation_path.exists():
            logger.warning(f"  アノテーションファイルが見つかりません: {annotation_path}")
            skipped_count += 1
            continue

        try:
            # アノテーション画像読み込み
            annotation_img = nib.load(str(annotation_path))

            # 椎体処理
            vertebra_results = process_vertebra_directory(
                vertebra_dir, case_number, inp_dir.name,
                annotation_img, output_base_dir
            )
            folder_results.extend(vertebra_results)
            vertebra_count += 1

        except Exception as e:
            logger.error(f"  椎体 {vertebra_dir.name} の読み込みエラー: {e}")
            skipped_count += 1

    # CSV 保存
    if folder_results:
        df = pd.DataFrame(folder_results)
        csv_path = output_base_dir / inp_dir.name / f"mask_labels_{inp_dir.name}.csv"
        df.to_csv(csv_path, index=False)

        fracture_count = df['Fracture_Label'].sum()
        total_slices = len(df)

        logger.info(f"完了: {inp_dir.name}")
        logger.info(f"  椎体数: {vertebra_count}, スキップ: {skipped_count}")
        logger.info(f"  総マスクスライス数: {total_slices}, 骨折スライス: {fracture_count}")
        logger.info(f"  CSV保存: {csv_path}")
        return True
    else:
        logger.warning(f"{inp_dir.name}: 処理可能なデータがありませんでした")
        return False

# ------------------- メイン処理 --------------------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("NIfTI マスクスライス抽出処理開始（テストデータ）")
    logger.info("=" * 60)
    logger.info(f"入力ディレクトリ: {input_base_dir}")
    logger.info(f"出力ディレクトリ: {output_base_dir}")
    logger.info(f"対象ケース数: {len(target_cases)} ({min(target_cases)} - {max(target_cases)})")

    input_path = Path(input_base_dir)
    output_path = Path(output_base_dir)

    if not input_path.exists():
        logger.error(f"入力ディレクトリが存在しません: {input_base_dir}")
        exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    # 処理統計
    total_cases = 0
    success_cases = 0
    failed_cases = 0

    try:
        for inp_dir in sorted(input_path.glob("inp*")):
            total_cases += 1
            if process_case_directory(inp_dir, output_path, target_cases):
                success_cases += 1
            else:
                failed_cases += 1
            logger.info("-" * 60)

        logger.info("=" * 60)
        logger.info("全ケース処理完了")
        logger.info(f"  処理対象: {total_cases}ケース")
        logger.info(f"  成功: {success_cases}ケース")
        logger.info(f"  失敗/スキップ: {failed_cases}ケース")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("処理が中断されました (Ctrl+C)")
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}", exc_info=True)
        raise

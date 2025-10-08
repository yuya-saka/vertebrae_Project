import os
import shutil
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging  # loggingモジュールをインポート
from tqdm import tqdm  # tqdmをインポートしてプログレスバーを表示

# --- ログ設定 ---
# ログのレベル、フォーマット、時刻の形式を設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("スクリプトを開始します。")

# 入力ディレクトリと出力ディレクトリの設定
input_all = Path("/mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/input_nii")
output_dir_train = Path("/mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/vertebrae_Unet/data/train")
output_dir_test = Path("/mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/vertebrae_Unet/data/test")

# 出力ディレクトリを作成
logging.info(f"出力ディレクトリを作成します: {output_dir_train}")
output_dir_train.mkdir(exist_ok=True)
logging.info(f"出力ディレクトリを作成します: {output_dir_test}")
output_dir_test.mkdir(exist_ok=True)

# 最適化されたファイル名から番号（xxxx）を抽出（正規表現使用）
number_pattern = re.compile(r'\d+')
def extract_number(filename):
    match = number_pattern.search(filename)
    return match.group() if match else ''

# ファイルを番号ごとにグループ化（最適化）
logging.info(f"入力ディレクトリをスキャン中: {input_all}")
files = list(input_all.iterdir())
logging.info(f"{len(files)}個のファイルが見つかりました。症例番号ごとにグループ化します。")

groups = {}
for file_path in files:
    if file_path.is_file():
        number = extract_number(file_path.name)
        if number:
            if number not in groups:
                groups[number] = []
            groups[number].append(file_path)

logging.info(f"{len(groups)}個の症例グループが見つかりました。")

# グループをシャッフルして8症例をテストに割り当て
logging.info("症例グループをシャッフルし、テストデータと訓練データに分割します。")
group_keys = list(groups.keys())
random.shuffle(group_keys)

# 8症例をテストデータに分ける
test_keys = set(group_keys[:8])
logging.info(f"テスト用に{len(test_keys)}症例を選択しました。")
logging.debug(f"テスト症例の番号: {sorted(list(test_keys))}") # 詳細なデバッグ情報

# 並列処理でファイルコピーを高速化
def copy_file(args):
    src_path, dest_path = args
    shutil.copy2(src_path, dest_path)  # copy2は高速でメタデータも保持
    return dest_path

# コピータスクを準備
logging.info("ファイルコピーのタスクリストを準備しています。")
copy_tasks = []
for number, file_paths in groups.items():
    target_dir = output_dir_test if number in test_keys else output_dir_train
    
    for file_path in file_paths:
        dest_path = target_dir / file_path.name
        copy_tasks.append((file_path, dest_path))

logging.info(f"合計 {len(copy_tasks)}個のファイルをコピーします。")

# 並列処理でファイルコピーを実行（tqdmでプログレスバー表示）
with ThreadPoolExecutor(max_workers=4) as executor:
    # tqdmでexecutor.mapをラップして進捗を表示
    results = list(tqdm(executor.map(copy_file, copy_tasks), total=len(copy_tasks), desc="ファイルコピー中"))

logging.info("すべてのファイルコピーが完了しました。")

# --- 最終結果の表示 ---
print("\n" + "="*40)
print("処理結果サマリー")
print("="*40)
print(f"テストデータの分割が完了しました。")
print(f"処理された合計ファイル数: {len(results)}")
print(f"テストデータに割り当てられた症例数: {len(test_keys)} 症例")
print(f"訓練データに割り当てられた症例数: {len(group_keys) - len(test_keys)} 症例")
print("="*40)

logging.info("スクリプトが正常に終了しました。")
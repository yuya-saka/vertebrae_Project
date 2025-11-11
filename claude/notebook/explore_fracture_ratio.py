"""
椎体骨折検出プロジェクト - 骨折画像比率探索
Fracture Image Ratio Exploration for vertebrae slice images

実行方法:
1. VSCodeでこのファイルを開く
2. Python拡張機能がインストールされていることを確認
3. セル単位で実行: Ctrl+Enter (またはCmd+Enter)
4. または全体実行: uv run python claude/notebook/explore_fracture_ratio.py

各セルは # %% で区切られており、VSCodeやJupyter拡張で実行可能

このノートブックの目的:
- 骨折スライスと非骨折スライスの比率を分析
- クラス不均衡の定量評価（医療AIで重要）
- 患者ごと、椎体ごとの骨折発生率を可視化
"""

# %% [markdown]
# # 椎体骨折スライスの比率分析
#
# このノートブックでは、3方向(axial, coronal, sagittal)から切り出した
# 椎体スライス画像における**骨折スライスの比率**を分析します。
#
# **医療AIにおける重要性**:
# - クラス不均衡（骨折は稀）が学習に与える影響の把握
# - サンプリング戦略の決定
# - 評価指標の選択（Accuracy vs Precision/Recall）

# %% セットアップ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

# 日本語フォント設定 - seabornより前に実行
import matplotlib.font_manager as fm

# Noto Sans CJK JPフォントを直接登録
noto_font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if noto_font_path.exists():
    fm.fontManager.addfont(str(noto_font_path))

plt.rcParams['axes.unicode_minus'] = False

# seabornを設定（これがフォント設定を上書きする可能性がある）
sns.set_style("whitegrid")

# seabornの後に再度フォント設定を適用
if noto_font_path.exists():
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP'] + plt.rcParams['font.sans-serif']
    print("✅ Noto Sans CJK JP フォントを使用")
else:
    # フォールバック
    try:
        import japanize_matplotlib
        print("✅ japanize_matplotlib を使用")
    except ImportError:
        print("⚠️ 日本語フォントが見つかりません。")
        print("   pip install japanize-matplotlib を実行してください。")

# データパス
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent.parent  # 3階層上がプロジェクトルート
SLICE_TRAIN = PROJECT_ROOT / "data/slice_train"
SLICE_TEST = PROJECT_ROOT / "data/slice_test"

# 3方向の設定
DIRECTIONS = ["axial", "coronal", "sagittal"]
DIRECTION_NAMES_JP = {
    "axial": "横断面 (Axial)",
    "coronal": "冠状断 (Coronal)",
    "sagittal": "矢状断 (Sagittal)"
}

print("✅ セットアップ完了")
print(f"Project Root: {PROJECT_ROOT}")
print(f"Training data: {SLICE_TRAIN}")
print(f"Test data: {SLICE_TEST}")

# %% データ読み込み関数
def load_all_csv_files(data_dir: Path, direction: str = "axial") -> pd.DataFrame:
    """
    指定方向のすべてのfracture_labels CSVファイルを読み込む

    Args:
        data_dir: データディレクトリ（SLICE_TRAIN or SLICE_TEST）
        direction: 方向 (axial/coronal/sagittal)

    Returns:
        統合されたDataFrame
    """
    direction_path = data_dir / direction

    if not direction_path.exists():
        print(f"⚠️  {direction_path} が見つかりません")
        return pd.DataFrame()

    # すべてのfracture_labels CSVファイルを検索
    csv_files = list(direction_path.glob("*/fracture_labels_*.csv"))

    if len(csv_files) == 0:
        print(f"⚠️  {direction} のCSVファイルが見つかりません")
        return pd.DataFrame()

    # すべてのCSVを読み込んで結合
    df_list = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df_list.append(df)
        except Exception as e:
            print(f"⚠️  {csv_file.name} の読み込みエラー: {e}")

    combined_df = pd.concat(df_list, ignore_index=True)

    print(f"✅ {direction} のデータ読み込み完了: {len(combined_df)}スライス ({len(csv_files)}ファイル)")

    return combined_df

print("\n📂 データ読み込み関数定義完了")

# %% Train データの読み込み（全方向）
print("\n" + "="*60)
print("【Training データ読み込み】")
print("="*60)

train_data = {}
for direction in DIRECTIONS:
    print(f"\n読み込み中: {DIRECTION_NAMES_JP[direction]}")
    train_data[direction] = load_all_csv_files(SLICE_TRAIN, direction)

# %% Test データの読み込み（全方向）
print("\n" + "="*60)
print("【Test データ読み込み】")
print("="*60)

test_data = {}
for direction in DIRECTIONS:
    print(f"\n読み込み中: {DIRECTION_NAMES_JP[direction]}")
    test_data[direction] = load_all_csv_files(SLICE_TEST, direction)

# %% 基本統計: Train全体
print("\n" + "="*60)
print("【Train データ: 骨折スライス比率】")
print("="*60)

for direction in DIRECTIONS:
    df = train_data[direction]

    if len(df) == 0:
        continue

    total_slices = len(df)
    fracture_slices = df['Fracture_Label'].sum()
    non_fracture_slices = total_slices - fracture_slices
    fracture_ratio = (fracture_slices / total_slices) * 100 if total_slices > 0 else 0

    print(f"\n[{DIRECTION_NAMES_JP[direction]}]")
    print(f"  総スライス数: {total_slices:,}")
    print(f"  骨折スライス数: {fracture_slices:,} ({fracture_ratio:.2f}%)")
    print(f"  非骨折スライス数: {non_fracture_slices:,} ({100-fracture_ratio:.2f}%)")
    print(f"  不均衡比率: 1:{non_fracture_slices/fracture_slices:.2f} (骨折:非骨折)" if fracture_slices > 0 else "  不均衡比率: N/A")

# %% 基本統計: Test全体
print("\n" + "="*60)
print("【Test データ: 骨折スライス比率】")
print("="*60)

for direction in DIRECTIONS:
    df = test_data[direction]

    if len(df) == 0:
        continue

    total_slices = len(df)
    fracture_slices = df['Fracture_Label'].sum()
    non_fracture_slices = total_slices - fracture_slices
    fracture_ratio = (fracture_slices / total_slices) * 100 if total_slices > 0 else 0

    print(f"\n[{DIRECTION_NAMES_JP[direction]}]")
    print(f"  総スライス数: {total_slices:,}")
    print(f"  骨折スライス数: {fracture_slices:,} ({fracture_ratio:.2f}%)")
    print(f"  非骨折スライス数: {non_fracture_slices:,} ({100-fracture_ratio:.2f}%)")
    print(f"  不均衡比率: 1:{non_fracture_slices/fracture_slices:.2f} (骨折:非骨折)" if fracture_slices > 0 else "  不均衡比率: N/A")

# %% 可視化1: 全体の骨折比率（3方向比較）
def plot_overall_fracture_ratio(train_data: Dict, test_data: Dict):
    """
    Train/Testの骨折比率を3方向で比較表示
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('骨折スライス比率 - 全体統計', fontsize=16, fontweight='bold')

    for idx, direction in enumerate(DIRECTIONS):
        # Train
        df_train = train_data[direction]
        if len(df_train) > 0:
            fracture_count = df_train['Fracture_Label'].sum()
            non_fracture_count = len(df_train) - fracture_count

            labels = ['非骨折', '骨折']
            sizes = [non_fracture_count, fracture_count]
            colors = ['#90caf9', '#ef5350']
            explode = (0, 0.1)

            axes[0, idx].pie(sizes, explode=explode, labels=labels, colors=colors,
                            autopct='%1.1f%%', shadow=True, startangle=90)
            axes[0, idx].set_title(f'Train - {DIRECTION_NAMES_JP[direction]}')
        else:
            axes[0, idx].text(0.5, 0.5, 'データなし', ha='center', va='center')
            axes[0, idx].set_title(f'Train - {DIRECTION_NAMES_JP[direction]}')

        # Test
        df_test = test_data[direction]
        if len(df_test) > 0:
            fracture_count = df_test['Fracture_Label'].sum()
            non_fracture_count = len(df_test) - fracture_count

            labels = ['非骨折', '骨折']
            sizes = [non_fracture_count, fracture_count]
            colors = ['#90caf9', '#ef5350']
            explode = (0, 0.1)

            axes[1, idx].pie(sizes, explode=explode, labels=labels, colors=colors,
                            autopct='%1.1f%%', shadow=True, startangle=90)
            axes[1, idx].set_title(f'Test - {DIRECTION_NAMES_JP[direction]}')
        else:
            axes[1, idx].text(0.5, 0.5, 'データなし', ha='center', va='center')
            axes[1, idx].set_title(f'Test - {DIRECTION_NAMES_JP[direction]}')

    plt.tight_layout()
    plt.show()

print("\n📊 全体の骨折比率を可視化中...")
plot_overall_fracture_ratio(train_data, test_data)

# %% 患者ごとの骨折スライス数と比率
def analyze_patient_fracture_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    患者ごとの骨折スライス数と比率を計算
    """
    patient_stats = []

    for case in df['Case'].unique():
        case_df = df[df['Case'] == case]
        total_slices = len(case_df)
        fracture_slices = case_df['Fracture_Label'].sum()
        fracture_ratio = (fracture_slices / total_slices) * 100 if total_slices > 0 else 0

        patient_stats.append({
            'Case': case,
            'TotalSlices': total_slices,
            'FractureSlices': fracture_slices,
            'NonFractureSlices': total_slices - fracture_slices,
            'FractureRatio': fracture_ratio
        })

    return pd.DataFrame(patient_stats).sort_values('Case')

print("\n" + "="*60)
print("【患者ごとの骨折スライス統計 - Train (Axial)】")
print("="*60)

patient_stats_train = analyze_patient_fracture_ratio(train_data['axial'])
print(patient_stats_train.to_string(index=False))

print(f"\n【サマリー】")
print(f"骨折スライスを持つ患者数: {(patient_stats_train['FractureSlices'] > 0).sum()} / {len(patient_stats_train)}")
print(f"平均骨折スライス数: {patient_stats_train['FractureSlices'].mean():.1f}")
print(f"平均骨折比率: {patient_stats_train['FractureRatio'].mean():.2f}%")

# %% 可視化2: 患者ごとの骨折スライス数分布
def plot_patient_fracture_distribution(patient_stats: pd.DataFrame, title_suffix: str = ""):
    """
    患者ごとの骨折スライス数をヒストグラムで表示
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 骨折スライス数の分布
    axes[0].hist(patient_stats['FractureSlices'], bins=20, alpha=0.7,
                 color='#ef5350', edgecolor='black')
    axes[0].set_xlabel('骨折スライス数')
    axes[0].set_ylabel('患者数')
    axes[0].set_title(f'患者ごとの骨折スライス数分布{title_suffix}')
    axes[0].grid(alpha=0.3)
    axes[0].axvline(patient_stats['FractureSlices'].mean(),
                    color='blue', linestyle='--', linewidth=2, label=f'平均: {patient_stats["FractureSlices"].mean():.1f}')
    axes[0].legend()

    # 骨折比率の分布
    axes[1].hist(patient_stats['FractureRatio'], bins=20, alpha=0.7,
                 color='#66bb6a', edgecolor='black')
    axes[1].set_xlabel('骨折比率 (%)')
    axes[1].set_ylabel('患者数')
    axes[1].set_title(f'患者ごとの骨折比率分布{title_suffix}')
    axes[1].grid(alpha=0.3)
    axes[1].axvline(patient_stats['FractureRatio'].mean(),
                    color='blue', linestyle='--', linewidth=2, label=f'平均: {patient_stats["FractureRatio"].mean():.2f}%')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

print("\n📊 患者ごとの骨折スライス分布を可視化中...")
plot_patient_fracture_distribution(patient_stats_train, " (Train - Axial)")

# %% 椎体番号ごとの骨折比率
def analyze_vertebra_fracture_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    椎体番号ごとの骨折スライス数と比率を計算
    """
    vertebra_stats = []

    for vertebra in sorted(df['Vertebra'].unique(), key=lambda x: int(x) if str(x).isdigit() else 999):
        vertebra_df = df[df['Vertebra'] == vertebra]
        total_slices = len(vertebra_df)
        fracture_slices = vertebra_df['Fracture_Label'].sum()
        fracture_ratio = (fracture_slices / total_slices) * 100 if total_slices > 0 else 0

        # 骨折がある患者数
        fracture_cases = vertebra_df[vertebra_df['Fracture_Label'] == 1]['Case'].nunique()
        total_cases = vertebra_df['Case'].nunique()

        vertebra_stats.append({
            'Vertebra': vertebra,
            'TotalSlices': total_slices,
            'FractureSlices': fracture_slices,
            'FractureRatio': fracture_ratio,
            'FractureCases': fracture_cases,
            'TotalCases': total_cases,
            'CaseFractureRatio': (fracture_cases / total_cases) * 100 if total_cases > 0 else 0
        })

    return pd.DataFrame(vertebra_stats)

print("\n" + "="*60)
print("【椎体番号ごとの骨折統計 - Train (Axial)】")
print("="*60)

vertebra_stats_train = analyze_vertebra_fracture_ratio(train_data['axial'])
print(vertebra_stats_train.to_string(index=False))

# %% 可視化3: 椎体番号ごとの骨折比率
def plot_vertebra_fracture_ratio(vertebra_stats: pd.DataFrame, title_suffix: str = ""):
    """
    椎体番号ごとの骨折比率を棒グラフで表示
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # スライスレベルの骨折比率
    axes[0].bar(vertebra_stats['Vertebra'].astype(str), vertebra_stats['FractureRatio'],
                color='#ef5350', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('椎体番号')
    axes[0].set_ylabel('骨折比率 (%)')
    axes[0].set_title(f'椎体ごとの骨折スライス比率{title_suffix}')
    axes[0].grid(alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=45)

    # 患者数を表示（各バーの上）
    for i, row in vertebra_stats.iterrows():
        axes[0].text(i, row['FractureRatio'] + 1,
                     f"{int(row['FractureSlices'])}/{int(row['TotalSlices'])}",
                     ha='center', va='bottom', fontsize=8)

    # ケースレベルの骨折比率
    axes[1].bar(vertebra_stats['Vertebra'].astype(str), vertebra_stats['CaseFractureRatio'],
                color='#66bb6a', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('椎体番号')
    axes[1].set_ylabel('骨折患者比率 (%)')
    axes[1].set_title(f'椎体ごとの骨折患者比率{title_suffix}')
    axes[1].grid(alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=45)

    # 患者数を表示（各バーの上）
    for i, row in vertebra_stats.iterrows():
        axes[1].text(i, row['CaseFractureRatio'] + 1,
                     f"{int(row['FractureCases'])}/{int(row['TotalCases'])}",
                     ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

print("\n📊 椎体ごとの骨折比率を可視化中...")
plot_vertebra_fracture_ratio(vertebra_stats_train, " (Train - Axial)")

# %% 3方向比較: 骨折比率
def compare_fracture_ratio_across_directions(data_dict: Dict, dataset_name: str = "Train"):
    """
    3方向の骨折比率を比較
    """
    comparison_data = []

    for direction in DIRECTIONS:
        df = data_dict[direction]
        if len(df) == 0:
            continue

        total_slices = len(df)
        fracture_slices = df['Fracture_Label'].sum()
        fracture_ratio = (fracture_slices / total_slices) * 100 if total_slices > 0 else 0

        comparison_data.append({
            'Direction': DIRECTION_NAMES_JP[direction],
            'TotalSlices': total_slices,
            'FractureSlices': fracture_slices,
            'NonFractureSlices': total_slices - fracture_slices,
            'FractureRatio': fracture_ratio
        })

    comparison_df = pd.DataFrame(comparison_data)

    print(f"\n【{dataset_name} - 3方向比較】")
    print(comparison_df.to_string(index=False))

    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 骨折比率の比較
    axes[0].bar(comparison_df['Direction'], comparison_df['FractureRatio'],
                color=['#42a5f5', '#66bb6a', '#ffa726'], alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('骨折比率 (%)')
    axes[0].set_title(f'{dataset_name} - 方向ごとの骨折比率')
    axes[0].grid(alpha=0.3, axis='y')

    for i, row in comparison_df.iterrows():
        axes[0].text(i, row['FractureRatio'] + 0.5,
                     f"{row['FractureRatio']:.2f}%",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    # スライス数の比較（積み上げ棒グラフ）
    axes[1].bar(comparison_df['Direction'], comparison_df['NonFractureSlices'],
                label='非骨折', color='#90caf9', alpha=0.7, edgecolor='black')
    axes[1].bar(comparison_df['Direction'], comparison_df['FractureSlices'],
                bottom=comparison_df['NonFractureSlices'],
                label='骨折', color='#ef5350', alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('スライス数')
    axes[1].set_title(f'{dataset_name} - 方向ごとのスライス数')
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

print("\n" + "="*60)
print("【3方向比較分析】")
print("="*60)

compare_fracture_ratio_across_directions(train_data, "Train")
compare_fracture_ratio_across_directions(test_data, "Test")

# %% 骨折スライス連続性の分析
def analyze_fracture_continuity(df: pd.DataFrame) -> Dict:
    """
    骨折スライスの連続性を分析

    骨折が連続して現れるか、散発的に現れるかを確認
    """
    continuity_stats = {
        'isolated_fractures': 0,  # 単独の骨折スライス
        'continuous_sequences': [],  # 連続する骨折スライスの長さ
    }

    for case in df['Case'].unique():
        for vertebra in df[df['Case'] == case]['Vertebra'].unique():
            vertebra_df = df[(df['Case'] == case) & (df['Vertebra'] == vertebra)]
            vertebra_df = vertebra_df.sort_values('SliceIndex')

            # 骨折ラベルのシーケンスを取得
            labels = vertebra_df['Fracture_Label'].values

            # 連続する骨折を検出
            in_sequence = False
            sequence_length = 0

            for label in labels:
                if label == 1:
                    if not in_sequence:
                        in_sequence = True
                        sequence_length = 1
                    else:
                        sequence_length += 1
                else:
                    if in_sequence:
                        if sequence_length == 1:
                            continuity_stats['isolated_fractures'] += 1
                        else:
                            continuity_stats['continuous_sequences'].append(sequence_length)
                        in_sequence = False
                        sequence_length = 0

            # 最後のシーケンスを処理
            if in_sequence:
                if sequence_length == 1:
                    continuity_stats['isolated_fractures'] += 1
                else:
                    continuity_stats['continuous_sequences'].append(sequence_length)

    return continuity_stats

print("\n" + "="*60)
print("【骨折スライス連続性分析 - Train (Axial)】")
print("="*60)

continuity_stats = analyze_fracture_continuity(train_data['axial'])

print(f"単独の骨折スライス数: {continuity_stats['isolated_fractures']}")
print(f"連続する骨折シーケンス数: {len(continuity_stats['continuous_sequences'])}")

if len(continuity_stats['continuous_sequences']) > 0:
    print(f"連続骨折の平均長: {np.mean(continuity_stats['continuous_sequences']):.2f} スライス")
    print(f"連続骨折の最大長: {max(continuity_stats['continuous_sequences'])} スライス")
    print(f"連続骨折の最小長: {min(continuity_stats['continuous_sequences'])} スライス")

    # ヒストグラム
    plt.figure(figsize=(10, 6))
    plt.hist(continuity_stats['continuous_sequences'], bins=20, alpha=0.7,
             color='#ef5350', edgecolor='black')
    plt.xlabel('連続する骨折スライス数')
    plt.ylabel('頻度')
    plt.title('連続する骨折スライスの長さ分布 (Train - Axial)')
    plt.grid(alpha=0.3)
    plt.axvline(np.mean(continuity_stats['continuous_sequences']),
                color='blue', linestyle='--', linewidth=2,
                label=f'平均: {np.mean(continuity_stats["continuous_sequences"]):.2f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% まとめと推奨事項
print("\n" + "="*60)
print("✅ 骨折比率分析完了")
print("="*60)

print("\n【主要な知見】")
print("1. クラス不均衡の程度:")
for direction in DIRECTIONS:
    df = train_data[direction]
    if len(df) > 0:
        fracture_slices = df['Fracture_Label'].sum()
        total_slices = len(df)
        non_fracture_slices = total_slices - fracture_slices
        if fracture_slices > 0:
            ratio = non_fracture_slices / fracture_slices
            print(f"   {DIRECTION_NAMES_JP[direction]}: 1:{ratio:.2f} (骨折:非骨折)")

print("\n2. 医療AI学習への影響:")
print("   - 大きなクラス不均衡が存在 → 対策が必要")
print("   - 対策例:")
print("     * 重み付き損失関数（Class Weights）")
print("     * Focal Loss の使用")
print("     * オーバーサンプリング / アンダーサンプリング")
print("     * データ拡張（骨折スライスに対して重点的に）")

print("\n3. 評価指標の選択:")
print("   - Accuracy は不適切（常に非骨折と予測しても高精度）")
print("   - 推奨指標:")
print("     * Precision / Recall")
print("     * F1-Score")
print("     * AUC-ROC / AUC-PR")
print("     * Sensitivity / Specificity")

print("\n4. 椎体ごとの特徴:")
print("   - 特定の椎体で骨折が多い可能性")
print("   - → 椎体位置を特徴量として活用できる可能性")

print("\n" + "="*60)
print("🎯 このノートブックで分析した内容:")
print("="*60)
print("✓ 全体の骨折スライス比率（Train/Test, 3方向）")
print("✓ 患者ごとの骨折スライス数と比率")
print("✓ 椎体番号ごとの骨折発生率")
print("✓ 骨折スライスの連続性")
print("✓ 3方向（axial/coronal/sagittal）の比較")

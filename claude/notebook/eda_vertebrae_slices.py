"""
椎体骨折検出プロジェクト - スライス画像EDA
Exploratory Data Analysis for vertebrae slice images

実行方法:
1. VSCodeでこのファイルを開く
2. Python拡張機能がインストールされていることを確認
3. セル単位で実行: Ctrl+Enter (またはCmd+Enter)
4. または全体実行: uv run python claude/notebook/eda_vertebrae_slices.py

各セルは # %% で区切られており、VSCodeやJupyter拡張で実行可能

データ構造:
data/
├── slice_train/
│   ├── axial/          # 横断面画像
│   │   └── {patient_id}/
│   │       └── {vertebra_id}/
│   │           └── slice_XXX.nii
│   ├── axial_mask/     # 横断面マスク
│   │   └── {patient_id}/
│   │       └── {vertebra_id}/
│   │           └── mask_XXX.nii
│   ├── coronal/        # 冠状断画像
│   ├── coronal_mask/   # 冠状断マスク
│   ├── sagittal/       # 矢状断画像
│   └── sagittal_mask/  # 矢状断マスク
└── slice_test/
    └── (同様の構造)

注意:
- 画像ファイル名: slice_XXX.nii
- マスクファイル名: mask_XXX.nii (対応するスライスと同じ番号)
"""

# %% [markdown]
# # 椎体スライス画像の探索的データ分析 (EDA)
#
# このノートブックでは、3方向(axial, coronal, sagittal)から切り出した
# 椎体スライス画像の特徴を分析します。

# %% セットアップ
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Tuple
import seaborn as sns

# 日本語フォント設定
import matplotlib.font_manager as fm

# Noto Sans CJK JPフォントを直接登録
noto_font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if noto_font_path.exists():
    fm.fontManager.addfont(str(noto_font_path))
    # font.familyではなくfont.sans-serifを設定
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

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# データパス
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent.parent  #3階層上がプロジェクトルート
SLICE_TRAIN = PROJECT_ROOT / "data/slice_train"
SLICE_TEST = PROJECT_ROOT / "data/slice_test"

# 3方向の設定
DIRECTIONS = ["axial", "coronal", "sagittal"]
DIRECTION_NAMES_JP = {
    "axial": "横断面 (Axial)",
    "coronal": "冠状断 (Coronal)",
    "sagittal": "矢状断 (Sagittal)"
}

# HU値のウィンドウレベル設定 (簡易に変更可能)
HU_PRESETS = {
    "bone": {"center": 400, "width": 1500, "name": "骨条件"},
    "soft_tissue": {"center": 40, "width": 400, "name": "軟部組織"},
    "lung": {"center": -600, "width": 1500, "name": "肺条件"},
    "custom": {"center": 0, "width": 2000, "name": "カスタム"}
}

print("✅ セットアップ完了")
print(f"Training data: {SLICE_TRAIN}")
print(f"Test data: {SLICE_TEST}")

# %% HU値変換関数
def apply_hu_window(image: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    HU値にウィンドウレベルを適用

    Args:
        image: 入力画像 (HU値)
        center: ウィンドウセンター
        width: ウィンドウ幅

    Returns:
        0-255にスケーリングされた画像
    """
    min_hu = center - width / 2
    max_hu = center + width / 2

    windowed = np.clip(image, min_hu, max_hu)
    normalized = (windowed - min_hu) / (max_hu - min_hu) * 255

    return normalized.astype(np.uint8)

def load_nifti_slice(nifti_path: Path) -> np.ndarray:
    """NIFTI形式のスライス画像を読み込む"""
    img = nib.load(str(nifti_path))
    data = img.get_fdata()

    # 2Dスライスを想定
    if data.ndim == 3:
        # 最も薄い次元を絞り込む
        data = np.squeeze(data)

    return data

print("✅ HU値変換関数定義完了")

# %% データ構造の確認
def scan_data_structure(data_dir: Path, max_patients: int = 5) -> Dict:
    """
    データ構造をスキャン

    Returns:
        患者数、椎体数、スライス数などの統計情報
    """
    stats = {
        "directions": {},
        "total_patients": 0,
        "vertebrae_list": set()
    }

    for direction in DIRECTIONS:
        dir_path = data_dir / direction
        if not dir_path.exists():
            print(f"⚠️  {direction} ディレクトリが見つかりません")
            continue

        patients = sorted([p for p in dir_path.iterdir() if p.is_dir()])
        stats["total_patients"] = max(stats["total_patients"], len(patients))

        vertebrae_counts = []
        slice_counts = []

        # サンプリングして確認
        for patient_dir in patients[:max_patients]:
            vertebrae = sorted([v for v in patient_dir.iterdir() if v.is_dir()])
            vertebrae_counts.append(len(vertebrae))

            for vertebra_dir in vertebrae:
                stats["vertebrae_list"].add(vertebra_dir.name)
                slices = list(vertebra_dir.glob("slice_*.nii"))
                slice_counts.append(len(slices))

        stats["directions"][direction] = {
            "patients": len(patients),
            "avg_vertebrae_per_patient": np.mean(vertebrae_counts) if vertebrae_counts else 0,
            "avg_slices_per_vertebra": np.mean(slice_counts) if slice_counts else 0,
            "total_slices_sampled": sum(slice_counts)
        }

    stats["vertebrae_list"] = sorted(list(stats["vertebrae_list"]))

    return stats

print("\n📊 Training データ構造を確認中...")
train_stats = scan_data_structure(SLICE_TRAIN)

print("\n📊 Test データ構造を確認中...")
test_stats = scan_data_structure(SLICE_TEST)

# 結果表示
print("\n" + "="*60)
print("【Training データ統計】")
print("="*60)
print(f"総患者数: {train_stats['total_patients']}")
print(f"椎体番号: {train_stats['vertebrae_list']}")

for direction in DIRECTIONS:
    if direction in train_stats["directions"]:
        d_stats = train_stats["directions"][direction]
        print(f"\n[{DIRECTION_NAMES_JP[direction]}]")
        print(f"  - 患者あたり平均椎体数: {d_stats['avg_vertebrae_per_patient']:.1f}")
        print(f"  - 椎体あたり平均スライス数: {d_stats['avg_slices_per_vertebra']:.1f}")
        print(f"  - サンプル総スライス数: {d_stats['total_slices_sampled']}")

print("\n" + "="*60)
print("【Test データ統計】")
print("="*60)
print(f"総患者数: {test_stats['total_patients']}")

for direction in DIRECTIONS:
    if direction in test_stats["directions"]:
        d_stats = test_stats["directions"][direction]
        print(f"\n[{DIRECTION_NAMES_JP[direction]}]")
        print(f"  - 患者あたり平均椎体数: {d_stats['avg_vertebrae_per_patient']:.1f}")
        print(f"  - 椎体あたり平均スライス数: {d_stats['avg_slices_per_vertebra']:.1f}")

# %% 画像サンプルの読み込みと基本統計
def analyze_sample_images(data_dir: Path,
                          direction: str = "axial",
                          n_samples: int = 30) -> pd.DataFrame:
    """
    サンプル画像を読み込んで基本統計を計算

    Args:
        data_dir: データディレクトリ
        direction: 方向 (axial/coronal/sagittal)
        n_samples: サンプル数

    Returns:
        統計情報のDataFrame
    """
    dir_path = data_dir / direction
    stats_list = []

    patients = sorted([p for p in dir_path.iterdir() if p.is_dir()])

    for patient_dir in patients[:3]:  # 最初の3患者
        vertebrae = sorted([v for v in patient_dir.iterdir() if v.is_dir()])

        for vertebra_dir in vertebrae[:2]:  # 各患者の最初の2椎体
            slices = sorted(vertebra_dir.glob("*.nii"))

            for slice_path in slices[:n_samples]:
                try:
                    img_data = load_nifti_slice(slice_path)

                    stats_list.append({
                        "patient": patient_dir.name,
                        "vertebra": vertebra_dir.name,
                        "slice": slice_path.stem,
                        "shape": img_data.shape,
                        "min_hu": float(np.min(img_data)),
                        "max_hu": float(np.max(img_data)),
                        "mean_hu": float(np.mean(img_data)),
                        "std_hu": float(np.std(img_data)),
                        "median_hu": float(np.median(img_data))
                    })
                except Exception as e:
                    print(f"⚠️  {slice_path}: {e}")

    return pd.DataFrame(stats_list)

print("\n📈 画像統計を計算中 (Axial)...")
axial_stats = analyze_sample_images(SLICE_TRAIN, "axial", n_samples=5)

print("\n【画像統計サマリー】")
print(axial_stats.describe())

print("\n【HU値の範囲】")
print(f"最小HU値: {axial_stats['min_hu'].min():.1f}")
print(f"最大HU値: {axial_stats['max_hu'].max():.1f}")
print(f"平均HU値: {axial_stats['mean_hu'].mean():.1f} ± {axial_stats['mean_hu'].std():.1f}")

# %% 骨折マスクの統計
def analyze_fracture_masks(data_dir: Path,
                           direction: str = "axial",
                           n_samples: int = 50) -> pd.DataFrame:
    """
    骨折マスクの統計を分析

    Args:
        data_dir: データディレクトリ
        direction: 方向
        n_samples: サンプル数

    Returns:
        マスク統計のDataFrame
    """
    mask_dir_path = data_dir / f"{direction}_mask"
    stats_list = []

    patients = sorted([p for p in mask_dir_path.iterdir() if p.is_dir()])

    sample_count = 0
    for patient_dir in patients:
        if sample_count >= n_samples:
            break

        vertebrae = sorted([v for v in patient_dir.iterdir() if v.is_dir()])

        for vertebra_dir in vertebrae:
            if sample_count >= n_samples:
                break

            mask_files = sorted(vertebra_dir.glob("mask_*.nii"))

            for mask_path in mask_files:
                if sample_count >= n_samples:
                    break

                try:
                    mask_data = load_nifti_slice(mask_path)
                    unique_values = np.unique(mask_data)
                    has_fracture = len(unique_values) > 1 or (len(unique_values) == 1 and unique_values[0] != 0)
                    fracture_ratio = np.sum(mask_data > 0) / mask_data.size if mask_data.size > 0 else 0

                    stats_list.append({
                        "patient": patient_dir.name,
                        "vertebra": vertebra_dir.name,
                        "slice": mask_path.stem,
                        "has_fracture": has_fracture,
                        "fracture_pixel_ratio": fracture_ratio,
                        "unique_values": len(unique_values),
                        "mask_values": str(unique_values.tolist())
                    })

                    sample_count += 1

                except Exception as e:
                    print(f"⚠️  {mask_path}: {e}")

    return pd.DataFrame(stats_list)

print("\n🔍 骨折マスクを分析中...")
mask_stats = analyze_fracture_masks(SLICE_TRAIN, "axial", n_samples=100)

print("\n【骨折マスク統計】")
print(f"総サンプル数: {len(mask_stats)}")
print(f"骨折あり: {mask_stats['has_fracture'].sum()} ({mask_stats['has_fracture'].mean()*100:.1f}%)")
print(f"骨折なし: {(~mask_stats['has_fracture']).sum()} ({(~mask_stats['has_fracture']).mean()*100:.1f}%)")
print(f"\n骨折ピクセル割合統計:")
print(mask_stats[mask_stats['has_fracture']]['fracture_pixel_ratio'].describe())

# %% 画像可視化関数 (HU値調整可能)
def find_fracture_vertebra(data_dir: Path, direction: str = "axial"):
    """
    骨折がある最初の椎体を見つける

    Args:
        data_dir: データディレクトリ
        direction: 方向

    Returns:
        (patient_id, vertebra_id) のタプル、見つからない場合はNone
    """
    mask_dir = data_dir / f"{direction}_mask"
    patients = sorted([p for p in mask_dir.iterdir() if p.is_dir()])

    for patient_dir in patients:
        vertebrae = sorted([v for v in patient_dir.iterdir() if v.is_dir()])

        for vertebra_dir in vertebrae:
            mask_files = sorted(vertebra_dir.glob("mask_*.nii"))

            # マスクファイルをチェック
            for mask_path in mask_files:
                try:
                    mask_data = load_nifti_slice(mask_path)
                    if np.any(mask_data != 0):
                        # 骨折が見つかった
                        return patient_dir.name, vertebra_dir.name
                except:
                    continue

    return None, None

def visualize_slices_with_hu_window(data_dir: Path,
                                    direction: str = "axial",
                                    patient_id: str = None,
                                    vertebra_id: str = None,
                                    n_slices: int = 6,
                                    hu_preset: str = "bone",
                                    custom_center: float = None,
                                    custom_width: float = None,
                                    show_mask: bool = True,
                                    only_fracture: bool = True):
    """
    スライス画像とマスクを可視化 (HU値ウィンドウ調整可能)

    Args:
        data_dir: データディレクトリ
        direction: 方向 (axial/coronal/sagittal)
        patient_id: 患者ID (Noneの場合は自動選択)
        vertebra_id: 椎体ID (Noneの場合は自動選択)
        n_slices: 表示スライス数
        hu_preset: HU値プリセット名 ("bone", "soft_tissue", "lung", "custom")
        custom_center: カスタムウィンドウセンター
        custom_width: カスタムウィンドウ幅
        show_mask: マスクを表示するかどうか
        only_fracture: Trueの場合、骨折がある椎体のみを自動選択
    """
    # HU値設定
    if hu_preset == "custom" and custom_center is not None and custom_width is not None:
        center = custom_center
        width = custom_width
        preset_name = f"Custom (C:{center}, W:{width})"
    else:
        preset = HU_PRESETS.get(hu_preset, HU_PRESETS["bone"])
        center = preset["center"]
        width = preset["width"]
        preset_name = preset["name"]

    # データパス
    img_dir = data_dir / direction
    mask_dir = data_dir / f"{direction}_mask"

    # 患者・椎体選択
    if patient_id is None or vertebra_id is None:
        if only_fracture:
            # 骨折がある椎体を自動検索
            found_patient, found_vertebra = find_fracture_vertebra(data_dir, direction)
            if found_patient is None:
                print("⚠️  骨折がある椎体が見つかりませんでした")
                return
            patient_id = patient_id or found_patient
            vertebra_id = vertebra_id or found_vertebra
            print(f"🔍 骨折がある椎体を検出: {patient_id}/{vertebra_id}")
        else:
            # 通常の選択
            if patient_id is None:
                patients = sorted([p for p in img_dir.iterdir() if p.is_dir()])
                patient_id = patients[0].name
            if vertebra_id is None:
                vertebrae = sorted([v for v in (img_dir / patient_id).iterdir() if v.is_dir()])
                vertebra_id = vertebrae[0].name

    patient_dir = img_dir / patient_id
    vertebra_dir = patient_dir / vertebra_id

    # スライス取得
    slice_paths = sorted(vertebra_dir.glob("slice_*.nii"))

    if len(slice_paths) == 0:
        print(f"⚠️  スライスが見つかりません: {vertebra_dir}")
        return

    # only_fractureが有効な場合、骨折があるスライスのみを選択
    if only_fracture and show_mask:
        fracture_slices = []
        for slice_path in slice_paths:
            mask_filename = slice_path.name.replace("slice_", "mask_")
            mask_path = mask_dir / patient_id / vertebra_id / mask_filename

            if mask_path.exists():
                try:
                    mask_data = load_nifti_slice(mask_path)
                    if np.any(mask_data != 0):
                        fracture_slices.append(slice_path)
                except:
                    continue

        if len(fracture_slices) > 0:
            slice_paths = fracture_slices
            print(f"🔍 骨折があるスライス: {len(fracture_slices)}枚")
        else:
            print(f"⚠️  この椎体には骨折スライスがありません")

    # 等間隔にサンプリング
    indices = np.linspace(0, len(slice_paths)-1, min(n_slices, len(slice_paths)), dtype=int)
    selected_slices = [slice_paths[i] for i in indices]

    # 可視化
    n_cols = 2 if show_mask else 1
    n_rows = len(selected_slices)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(f"{DIRECTION_NAMES_JP[direction]} - {patient_id} - 椎体{vertebra_id}\n"
                 f"HU Window: {preset_name}", fontsize=14, fontweight='bold')

    for idx, slice_path in enumerate(selected_slices):
        # 画像読み込み
        img_data = load_nifti_slice(slice_path)
        img_windowed = apply_hu_window(img_data, center, width)

        # 画像表示
        axes[idx, 0].imshow(img_windowed, cmap='gray')
        axes[idx, 0].set_title(f"{slice_path.stem}\n"
                               f"HU: [{img_data.min():.0f}, {img_data.max():.0f}]")
        axes[idx, 0].axis('off')

        # マスク表示
        if show_mask:
            # mask_XXX.nii の形式に変換
            mask_filename = slice_path.name.replace("slice_", "mask_")
            mask_path = mask_dir / patient_id / vertebra_id / mask_filename

            if mask_path.exists():
                mask_data = load_nifti_slice(mask_path)

                # オーバーレイ表示
                axes[idx, 1].imshow(img_windowed, cmap='gray')
                axes[idx, 1].imshow(mask_data, cmap='Reds', alpha=0.5 * (mask_data > 0))

                fracture_ratio = np.sum(mask_data > 0) / mask_data.size * 100
                axes[idx, 1].set_title(f"Mask (骨折領域: {fracture_ratio:.2f}%)")
                axes[idx, 1].axis('off')
            else:
                axes[idx, 1].text(0.5, 0.5, "マスクなし", ha='center', va='center')
                axes[idx, 1].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"✅ 表示完了: {patient_id} - 椎体{vertebra_id} ({len(selected_slices)}スライス)")

# 使用例
print("\n🖼️  画像可視化の準備完了")
print("="*60)
print("✨ デフォルトで骨折がある椎体・スライスのみを自動表示します")
print("\n基本的な使用方法:")
print("visualize_slices_with_hu_window(SLICE_TRAIN)")
print("  → 骨折がある椎体を自動検出して、骨折スライスのみを表示")
print("\n詳細設定:")
print("visualize_slices_with_hu_window(")
print("    SLICE_TRAIN,")
print("    direction='axial',  # 'axial', 'coronal', 'sagittal'")
print("    patient_id=None,  # None=自動検出、または 'inp1017' など指定")
print("    vertebra_id=None,  # None=自動検出、または '28' など指定")
print("    n_slices=6,")
print("    hu_preset='bone',  # 'bone', 'soft_tissue', 'lung', 'custom'")
print("    show_mask=True,")
print("    only_fracture=True  # False=すべてのスライスを表示")
print(")")
print("\nカスタムHU値:")
print("visualize_slices_with_hu_window(")
print("    SLICE_TRAIN, hu_preset='custom',")
print("    custom_center=400, custom_width=2000")
print(")")
print("\n💡 骨折なしデータも表示したい場合:")
print("visualize_slices_with_hu_window(SLICE_TRAIN, only_fracture=False)")

# %% サンプル表示: 骨条件
print("\n" + "="*60)
print("【サンプル表示 1: 骨条件 (Bone Window)】")
print("="*60)

visualize_slices_with_hu_window(
    SLICE_TRAIN,
    direction="axial",
    n_slices=4,
    hu_preset="bone",
    show_mask=True
)

# %% サンプル表示: 軟部組織条件
print("\n" + "="*60)
print("【サンプル表示 2: 軟部組織条件 (Soft Tissue Window)】")
print("="*60)

visualize_slices_with_hu_window(
    SLICE_TRAIN,
    direction="axial",
    n_slices=4,
    hu_preset="soft_tissue",
    show_mask=True
)

# %% 3方向比較表示
def compare_three_directions(data_dir: Path,
                             patient_id: str = None,
                             vertebra_id: str = None,
                             slice_idx: int = None,
                             hu_preset: str = "bone",
                             only_fracture: bool = True):
    """
    同じ椎体の3方向を比較表示

    Args:
        data_dir: データディレクトリ
        patient_id: 患者ID (Noneの場合は自動選択)
        vertebra_id: 椎体ID (Noneの場合は自動選択)
        slice_idx: スライスインデックス (Noneの場合は骨折があるスライスを選択)
        hu_preset: HU値プリセット
        only_fracture: Trueの場合、骨折がある椎体を自動選択
    """
    # 患者・椎体を自動選択
    if patient_id is None or vertebra_id is None:
        if only_fracture:
            found_patient, found_vertebra = find_fracture_vertebra(data_dir, "axial")
            if found_patient is None:
                print("⚠️  骨折がある椎体が見つかりませんでした")
                return
            patient_id = patient_id or found_patient
            vertebra_id = vertebra_id or found_vertebra
            print(f"🔍 骨折がある椎体を検出: {patient_id}/{vertebra_id}")

    preset = HU_PRESETS.get(hu_preset, HU_PRESETS["bone"])
    center = preset["center"]
    width = preset["width"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"3方向比較 - {patient_id} - 椎体{vertebra_id}\n"
                 f"HU Window: {preset['name']}", fontsize=16, fontweight='bold')

    for col, direction in enumerate(DIRECTIONS):
        img_dir = data_dir / direction / patient_id
        vertebra_dir = img_dir / vertebra_id

        slice_paths = sorted(vertebra_dir.glob("slice_*.nii"))

        if len(slice_paths) == 0:
            axes[0, col].text(0.5, 0.5, "データなし", ha='center', va='center')
            axes[1, col].text(0.5, 0.5, "データなし", ha='center', va='center')
            continue

        # 骨折があるスライスを探す
        if slice_idx is None and only_fracture:
            mask_dir = data_dir / f"{direction}_mask" / patient_id / vertebra_id
            fracture_slice_idx = None

            for idx, slice_path in enumerate(slice_paths):
                mask_filename = slice_path.name.replace("slice_", "mask_")
                mask_path = mask_dir / mask_filename

                if mask_path.exists():
                    try:
                        mask_data = load_nifti_slice(mask_path)
                        if np.any(mask_data != 0):
                            fracture_slice_idx = idx
                            break
                    except:
                        continue

            if fracture_slice_idx is not None:
                mid_idx = fracture_slice_idx
            else:
                mid_idx = len(slice_paths) // 2
        else:
            # 中央スライスを選択
            if slice_idx is None:
                mid_idx = len(slice_paths) // 2
            else:
                mid_idx = min(slice_idx, len(slice_paths) - 1)

        slice_path = slice_paths[mid_idx]

        # 画像読み込み
        img_data = load_nifti_slice(slice_path)
        img_windowed = apply_hu_window(img_data, center, width)

        # 画像表示
        axes[0, col].imshow(img_windowed, cmap='gray')
        axes[0, col].set_title(f"{DIRECTION_NAMES_JP[direction]}\n{slice_path.stem}")
        axes[0, col].axis('off')

        # マスク表示
        mask_filename = slice_path.name.replace("slice_", "mask_")
        mask_path = data_dir / f"{direction}_mask" / patient_id / vertebra_id / mask_filename
        if mask_path.exists():
            mask_data = load_nifti_slice(mask_path)

            axes[1, col].imshow(img_windowed, cmap='gray')
            axes[1, col].imshow(mask_data, cmap='Reds', alpha=0.5 * (mask_data > 0))

            fracture_ratio = np.sum(mask_data > 0) / mask_data.size * 100
            axes[1, col].set_title(f"骨折領域: {fracture_ratio:.2f}%")
            axes[1, col].axis('off')
        else:
            axes[1, col].text(0.5, 0.5, "マスクなし", ha='center', va='center')
            axes[1, col].axis('off')

    plt.tight_layout()
    plt.show()

print("\n🔄 3方向比較表示の準備完了")
print("✨ デフォルトで骨折がある椎体・スライスを自動表示します")
print("\n基本的な使用方法:")
print("compare_three_directions(SLICE_TRAIN)")
print("  → 骨折がある椎体を自動検出して、骨折スライスを3方向で比較表示")
print("\n詳細設定:")
print("compare_three_directions(")
print("    SLICE_TRAIN,")
print("    patient_id=None,  # None=自動検出")
print("    vertebra_id=None,  # None=自動検出")
print("    slice_idx=None,  # None=骨折スライスを自動選択")
print("    hu_preset='bone',")
print("    only_fracture=True  # False=通常の選択")
print(")")

# %% サンプル: 3方向比較
print("\n" + "="*60)
print("【3方向比較表示】")
print("="*60)

compare_three_directions(
    SLICE_TRAIN,
    hu_preset="bone"
)

# %% HU値分布の可視化
def plot_hu_distribution(data_dir: Path,
                         direction: str = "axial",
                         n_samples: int = 20):
    """
    HU値の分布をヒストグラムで可視化

    Args:
        data_dir: データディレクトリ
        direction: 方向
        n_samples: サンプル数
    """
    hu_values = []

    img_dir = data_dir / direction
    patients = sorted([p for p in img_dir.iterdir() if p.is_dir()])

    sample_count = 0
    for patient_dir in patients:
        if sample_count >= n_samples:
            break

        vertebrae = sorted([v for v in patient_dir.iterdir() if v.is_dir()])

        for vertebra_dir in vertebrae:
            if sample_count >= n_samples:
                break

            slices = sorted(vertebra_dir.glob("slice_*.nii"))

            if len(slices) > 0:
                # 中央スライスを使用
                slice_path = slices[len(slices) // 2]
                img_data = load_nifti_slice(slice_path)
                hu_values.extend(img_data.flatten().tolist())
                sample_count += 1

    hu_values = np.array(hu_values)

    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 全体分布
    axes[0].hist(hu_values, bins=100, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('HU値')
    axes[0].set_ylabel('頻度')
    axes[0].set_title(f'HU値分布 - {DIRECTION_NAMES_JP[direction]} (全体)')
    axes[0].grid(alpha=0.3)

    # クリップされた分布 (-200 ~ 1500)
    axes[1].hist(hu_values, bins=100, range=(-200, 1500), alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('HU値')
    axes[1].set_ylabel('頻度')
    axes[1].set_title(f'HU値分布 - {DIRECTION_NAMES_JP[direction]} (-200~1500)')
    axes[1].grid(alpha=0.3)

    # HU値プリセットの範囲を表示
    for preset_name, preset in HU_PRESETS.items():
        if preset_name != "custom":
            center = preset["center"]
            width = preset["width"]
            min_hu = center - width / 2
            max_hu = center + width / 2
            axes[1].axvspan(min_hu, max_hu, alpha=0.1, label=preset["name"])

    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print(f"\n【HU値統計 - {DIRECTION_NAMES_JP[direction]}】")
    print(f"サンプル数: {sample_count}")
    print(f"最小値: {hu_values.min():.1f}")
    print(f"最大値: {hu_values.max():.1f}")
    print(f"平均値: {hu_values.mean():.1f}")
    print(f"中央値: {np.median(hu_values):.1f}")
    print(f"標準偏差: {hu_values.std():.1f}")

# %% HU値分布の表示
print("\n" + "="*60)
print("【HU値分布の可視化】")
print("="*60)

plot_hu_distribution(SLICE_TRAIN, direction="axial", n_samples=30)

# %% 椎体ごとのスライス数分布
def plot_slice_count_distribution(data_dir: Path, direction: str = "axial"):
    """
    椎体ごとのスライス数分布を可視化

    Args:
        data_dir: データディレクトリ
        direction: 方向
    """
    slice_counts = defaultdict(list)

    img_dir = data_dir / direction
    patients = sorted([p for p in img_dir.iterdir() if p.is_dir()])

    for patient_dir in patients:
        vertebrae = sorted([v for v in patient_dir.iterdir() if v.is_dir()])

        for vertebra_dir in vertebrae:
            vertebra_id = vertebra_dir.name
            slices = list(vertebra_dir.glob("slice_*.nii"))
            slice_counts[vertebra_id].append(len(slices))

    # データフレーム化
    df_list = []
    for vertebra_id, counts in slice_counts.items():
        for count in counts:
            df_list.append({"vertebra": vertebra_id, "slice_count": count})

    df = pd.DataFrame(df_list)

    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 箱ひげ図
    vertebrae_sorted = sorted(df['vertebra'].unique(), key=lambda x: int(x) if x.isdigit() else 999)
    df['vertebra'] = pd.Categorical(df['vertebra'], categories=vertebrae_sorted, ordered=True)
    df.boxplot(column='slice_count', by='vertebra', ax=axes[0])
    axes[0].set_xlabel('椎体番号')
    axes[0].set_ylabel('スライス数')
    axes[0].set_title(f'椎体ごとのスライス数分布 - {DIRECTION_NAMES_JP[direction]}')
    axes[0].grid(alpha=0.3)
    plt.sca(axes[0])
    plt.xticks(rotation=45)

    # ヒストグラム
    axes[1].hist(df['slice_count'], bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('スライス数')
    axes[1].set_ylabel('頻度')
    axes[1].set_title(f'スライス数の全体分布 - {DIRECTION_NAMES_JP[direction]}')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n【スライス数統計 - {DIRECTION_NAMES_JP[direction]}】")
    print(df.groupby('vertebra')['slice_count'].describe())

# %% スライス数分布の表示
print("\n" + "="*60)
print("【椎体ごとのスライス数分布】")
print("="*60)

plot_slice_count_distribution(SLICE_TRAIN, direction="axial")

# %% まとめ
print("\n" + "="*60)
print("✅ EDA完了")
print("="*60)
print("\n主要な関数:")
print("1. visualize_slices_with_hu_window() - HU値調整可能なスライス可視化")
print("2. compare_three_directions() - 3方向比較表示")
print("3. plot_hu_distribution() - HU値分布の可視化")
print("4. plot_slice_count_distribution() - スライス数分布")
print("\nHU値プリセット:")
for preset_name, preset in HU_PRESETS.items():
    if preset_name != "custom":
        print(f"  - {preset_name}: {preset['name']} (C:{preset['center']}, W:{preset['width']})")

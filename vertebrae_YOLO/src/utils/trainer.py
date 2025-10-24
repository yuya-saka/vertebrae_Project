import os
import yaml
import glob
import shutil
from types import SimpleNamespace
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from tqdm import tqdm

# Add project root to path to allow imports from other modules
import sys
project_root_for_imports = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_for_imports not in sys.path:
    sys.path.insert(0, project_root_for_imports)

from vertebrae_YOLO.data_preparing.convert_nii_to_png import convert_nii_to_png

# =================================================================================
# ===== Step 1: カスタムクラスの定義 (ここから追記) =====
# =================================================================================

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Compose, Format, LetterBox
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.utils import colorstr

# Focal Loss用のカスタム損失関数をimport
from vertebrae_YOLO.src.utils.custom_loss import CustomDetectionLoss

# --- 1. カスタムデータセットクラス ---
# YOLODatasetを継承し、ラベルの有無で拡張を切り替える
class CustomYOLOv8Dataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 注意: self.transformsは親クラスの__init__で既に構築されている
        # ここでは何もする必要はない

    def __getitem__(self, index):
        # 元のデータセットから画像とラベル情報を取得
        labels = self.get_image_and_label(index)

        # ラベル（骨折）があるかどうかを判定
        has_labels = 'cls' in labels and len(labels['cls']) > 0

        # ラベルがない場合、確率的augmentationを一時的に無効化
        if not has_labels:
            # transformsパイプライン内の確率的augmentationを一時的に無効化
            # これは、RandomHSV, RandomFlip, Albumentationsなどの .p 属性を持つtransformsに適用
            original_probs = []
            for t in self.transforms.transforms:
                if hasattr(t, 'p'):
                    original_probs.append((t, t.p))
                    t.p = 0.0  # 確率を0に設定してaugmentationを無効化

            # 変換を適用
            labels = self.transforms(labels)

            # 確率を元に戻す
            for t, p in original_probs:
                t.p = p
        else:
            # ラベルがある場合は通常通り変換を適用
            labels = self.transforms(labels)

        return labels

# --- 2. カスタムトレーナークラス ---
# DetectionTrainerを継承し、CustomYOLOv8Datasetを使うように設定
class CustomDetectionTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        モデルをロードし、カスタム損失関数（Focal Loss）を設定

        Args:
            cfg: モデル設定
            weights: 事前学習済み重みのパス
            verbose: 詳細情報の表示

        Returns:
            model: Focal Lossが適用されたYOLOv8モデル
        """
        # 親クラスのget_modelを呼び出してモデルをロード
        model = super().get_model(cfg, weights, verbose)

        # Focal Loss設定をself属性から取得（Trainer.fit()でアタッチされる）
        use_focal_loss = getattr(self, 'use_focal_loss', False)

        if use_focal_loss:
            gamma = getattr(self, 'focal_gamma', 2.0)
            alpha = getattr(self, 'focal_alpha', 0.3)

            print(f"\n{'='*60}")
            print(f"🔥 Applying Focal Loss to YOLOv8 Detection Loss")
            print(f"{'='*60}")
            print(f"  Focal Loss Parameters:")
            print(f"    - gamma (focusing parameter): {gamma}")
            print(f"    - alpha (balancing factor): {alpha}")
            print(f"  Expected Effects:")
            print(f"    ✓ Down-weights easy examples (non-fracture backgrounds)")
            print(f"    ✓ Focuses on hard examples (subtle fractures)")
            print(f"    ✓ Addresses class imbalance")
            print(f"{'='*60}\n")

            # YOLOv8の損失関数をFocal Lossに置き換え
            # 注意: v8DetectionLossを継承したCustomDetectionLossを使用
            model.loss = CustomDetectionLoss(model, gamma=gamma, alpha=alpha)
        else:
            print(f"\nℹ️  Using default YOLOv8 loss (BCEWithLogitsLoss)")
            print(f"   Set use_focal_loss=true in config.yaml to enable Focal Loss\n")

        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        デフォルトのbuild_yolo_datasetの代わりにCustomYOLOv8Datasetを返すようにオーバーライド
        """
        from ultralytics.utils.torch_utils import de_parallel

        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        dataset = CustomYOLOv8Dataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == 'train',  # trainモードのときだけaugmentを有効にする
            hyp=self.hyp_dict, # self.args.hypの代わりに、アタッチされた辞書を使用
            rect=mode == 'val',
            cache=self.args.cache,
            stride=gs,
            pad=0.0,
            prefix=colorstr(f'{mode}: '),
            data=self.data
        )
        return dataset

# =================================================================================
# ===== Step 2: 既存のTrainerクラスの改造 =====
# =================================================================================

class Trainer:
    """
    Ultralytics YOLOv8の学習プロセスを管理するトレーナークラス。
    - (改造) CustomDetectionTrainerを使用して、条件付きデータ拡張を実行
    - NIFTIをPNGに変換し、永続的なキャッシュディレクトリに保存
    - data.yamlを動的に生成
    - クリーンアップ
    """
    def __init__(self, model, cfg: DictConfig, project_root: str):
        self.model = model
        self.cfg = cfg
        self.project_root = project_root
        self.run_dir = os.path.join(os.getcwd(), "run_specific_data", f"fold_{self.cfg.split.fold_id}")
        self.cache_dir = os.path.join(self.project_root, "processed_data", self.cfg.constants.view)

        # hyp.yamlを読み込んで辞書として保持
        self.hyp_dict = None
        if hasattr(self.cfg.training, 'hyp') and self.cfg.training.hyp:
            hyp_path = os.path.join(self.project_root, self.cfg.training.hyp)
            if os.path.exists(hyp_path):
                with open(hyp_path) as f:
                    hyp_dict_raw = yaml.safe_load(f)
                    self.hyp_dict = SimpleNamespace(**hyp_dict_raw) # SimpleNamespaceに変換
            else:
                print(f"  WARNING: Custom hyperparameter file not found at {hyp_path}, using defaults.")

    def _prepare_data(self) -> str:
        print("--- Preparing data for Ultralytics ---")
        
        train_img_dir = os.path.join(self.cache_dir, 'images', 'train', f"fold_{self.cfg.split.fold_id}")
        val_img_dir = os.path.join(self.cache_dir, 'images', 'val', f"fold_{self.cfg.split.fold_id}")
        train_lbl_dir = os.path.join(self.cache_dir, 'labels', 'train', f"fold_{self.cfg.split.fold_id}")
        val_lbl_dir = os.path.join(self.cache_dir, 'labels', 'val', f"fold_{self.cfg.split.fold_id}")
        
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(train_lbl_dir, exist_ok=True)
        os.makedirs(val_lbl_dir, exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)

        base_image_dir = os.path.join(self.project_root, self.cfg.constants.data_dir, 'images', self.cfg.constants.view, self.cfg.constants.split)
        base_label_dir = os.path.join(self.project_root, self.cfg.constants.data_dir, 'labels', self.cfg.constants.view, self.cfg.constants.split)

        self._process_split(self.cfg.split.train_patients, base_image_dir, base_label_dir, train_img_dir, train_lbl_dir, "train")
        self._process_split(self.cfg.split.val_patients, base_image_dir, base_label_dir, val_img_dir, val_lbl_dir, "validation")
        print("Data preparation is complete.")

        data_yaml_path = os.path.join(self.run_dir, 'data.yaml')
        data_yaml_content = {
            'train': os.path.abspath(train_img_dir),
            'val': os.path.abspath(val_img_dir),
            'nc': self.cfg.model.num_classes,
            'names': ['fracture']
        }
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml_content, f)
        print(f"data.yaml created at: {data_yaml_path}")
        
        return data_yaml_path

    def _process_split(self, patient_ids, src_img_dir, src_lbl_dir, dest_img_dir, dest_lbl_dir, split_name):
        print(f"  Processing {split_name} split...")
        hu_windows = self.cfg.constants.hu_windows
        files_converted = 0
        
        for patient_id in tqdm(patient_ids, desc=f"  Converting {split_name} NIFTI to PNG"):
            glob_pattern = os.path.join(src_img_dir, f"{patient_id}*.nii*")
            nii_files = glob.glob(glob_pattern)
            if not nii_files:
                continue

            for nii_path in nii_files:
                png_filename = os.path.splitext(os.path.basename(nii_path))[0].replace('.nii', '') + '.png'
                output_png_path = os.path.join(dest_img_dir, png_filename)
                if not os.path.exists(output_png_path):
                    if convert_nii_to_png(nii_path, output_png_path, hu_windows):
                        files_converted += 1
            
            for lbl_path in glob.glob(os.path.join(src_lbl_dir, f"{patient_id}*.txt")):
                dest_path = os.path.join(dest_lbl_dir, os.path.basename(lbl_path))
                if not os.path.exists(dest_path):
                    os.symlink(os.path.abspath(lbl_path), dest_path)
        
        if files_converted > 0:
            print(f"  {files_converted} new NIFTI files converted to PNG for this split.")
        else:
            print(f"  All PNG files for this split already exist in cache.")

    def fit(self):
        """
        (改造) CustomDetectionTrainerをセットアップして学習を実行する
        """
        data_yaml_path = self._prepare_data()

        print("\n--- Initializing Custom Trainer ---")

        # 1. YOLOv8のデフォルト設定をロード
        overrides = vars(DEFAULT_CFG).copy()

        # 2. 我々のconfig.yamlの内容で設定を上書き (手動)
        training_cfg = self.cfg.training
        if hasattr(training_cfg, 'lr0'): overrides['lr0'] = training_cfg.lr0
        if hasattr(training_cfg, 'weight_decay'): overrides['weight_decay'] = training_cfg.weight_decay
        if hasattr(training_cfg, 'optimizer'): overrides['optimizer'] = training_cfg.optimizer
        if hasattr(training_cfg, 'warmup_epochs'): overrides['warmup_epochs'] = training_cfg.warmup_epochs
        if hasattr(training_cfg, 'epochs'): overrides['epochs'] = training_cfg.epochs
        if hasattr(training_cfg, 'patience'): overrides['patience'] = training_cfg.patience
        if hasattr(training_cfg, 'device'): overrides['device'] = training_cfg.device

        # Focal Loss設定は overrides に含めない（Ultralyticsの引数検証を回避）
        # 代わりに、trainer インスタンスに直接アタッチする
        use_focal_loss = getattr(training_cfg, 'use_focal_loss', False)
        focal_gamma = getattr(training_cfg, 'focal_gamma', 2.0)
        focal_alpha = getattr(training_cfg, 'focal_alpha', 0.3)

        overrides['imgsz'] = self.cfg.constants.image_size
        overrides['batch'] = self.cfg.constants.batch_size
        overrides['data'] = data_yaml_path
        overrides['project'] = self.cfg.logging.project_name
        overrides['name'] = f"{self.cfg.logging.experiment_name}/fold_{self.cfg.split.fold_id}"
        overrides['exist_ok'] = True
        overrides['seed'] = self.cfg.seed

        # モデルのパスを設定
        overrides['model'] = self.model.model_name

        try:
            # 3. カスタムトレーナーをインスタンス化して学習を開始
            trainer = CustomDetectionTrainer(overrides=overrides)
            trainer.hyp_dict = self.hyp_dict  # hyp辞書をアタッチ

            # Focal Loss設定をtrainerに直接アタッチ（argsには含めない）
            trainer.use_focal_loss = use_focal_loss
            trainer.focal_gamma = focal_gamma
            trainer.focal_alpha = focal_alpha

            trainer.train()
        finally:
            self._cleanup()

    def _cleanup(self):
        print("--- Cleaning up run-specific data ---")
        if os.path.exists(self.run_dir):
            shutil.rmtree(self.run_dir)
            print(f"Removed run-specific directory: {self.run_dir}")
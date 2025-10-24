
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import glob

# Add the project root to the Python path to allow absolute imports
import sys
# Assuming the script is in /path/to/project/vertebrae_YOLO/run/scripts/train
# This moves up 4 levels to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from vertebrae_YOLO.src.models.yolo_baseline import YOLOBaseline
from vertebrae_YOLO.src.utils.trainer import Trainer

@hydra.main(config_path="../../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """メインの学習実行関数。"""
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))

    # --- 1. 再現性のためのシード設定 ---
    # numpyもインポート
    import numpy as np
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # --- 2. モデルの準備 ---
    print("\n--- Initializing Model ---")
    model = YOLOBaseline(
        variant=cfg.model.variant,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained
    )
    print(f"Model: {cfg.model.name} ({cfg.model.variant}) initialized.")

    # --- 3. 学習の実行 ---
    print("\n--- Initializing Trainer ---")
    trainer = Trainer(
        model=model,
        cfg=cfg,
        project_root=project_root
    )
    
    trainer.fit()

    print("\n--- Training Finished ---")

if __name__ == "__main__":
    main()

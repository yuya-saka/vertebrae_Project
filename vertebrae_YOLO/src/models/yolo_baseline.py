
import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLOBaseline(nn.Module):
    """
    YOLOv8ベースラインモデルをラップするPyTorchモジュール。
    UltralyticsのYOLOオブジェクトを内部で保持し、推論と学習のインターフェースを提供。
    """
    def __init__(self, variant='yolov8n', num_classes=1, pretrained=True):
        """
        Args:
            variant (str): YOLOv8のモデルバリアント (例: 'yolov8n', 'yolov8s')。
            num_classes (int): 検出対象のクラス数。
            pretrained (bool): 事前学習済み重みを使用するかどうか。
        """
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes
        self.pretrained = pretrained

        # 事前学習済みモデル（.pt）またはスクラッチモデル（.yaml）をロード
        model_name = f'{self.variant}.pt' if self.pretrained else f'{self.variant}.yaml'
        self.model_name = model_name
        self.yolo = YOLO(model_name)

        # モデルのクラス数を学習時に指定するため、ここでは何もしない

    def forward(self, x):
        """
        推論のためのフォワードパス。
        学習時は、このモジュールではなく、trainerがyolo.train()を呼び出すことを想定。
        
        Args:
            x (torch.Tensor): 入力画像テンソル (B, C, H, W)。
        
        Returns:
            (list): Ultralyticsの推論結果オブジェクトのリスト。
        """
        # yolo(x)は推論を実行し、結果オブジェクトを返す
        return self.yolo(x)

    def train_model(self, **kwargs):
        """
        Ultralyticsの学習機能を呼び出すためのラッパーメソッド。
        
        Args:
            **kwargs: yolo.train()に渡す引数（例: data, epochs, imgsz）。
        """
        # yolo.train()は学習を実行し、結果オブジェクトを返す
        return self.yolo.train(**kwargs)


if __name__ == '__main__':
    # --- このファイルが直接実行された場合のテストコード ---
    
    # 1. モデルのインスタンス化
    print("--- Model Instantiation Test ---")
    try:
        model = YOLOBaseline(variant='yolov8n', num_classes=1, pretrained=True)
        print("✅ Model instantiated successfully!")
        print(f"Model variant: {model.variant}")
        print(f"Num classes overridden to: {model.yolo.overrides['nc']}")
    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        model = None

    if model:
        # 2. ダミー入力で推論をテスト
        print("\n--- Inference Test ---")
        try:
            # (B, C, H, W) のダミーテンソルを作成
            dummy_input = torch.randn(1, 3, 256, 256)
            
            # 推論モードに設定
            model.eval()
            
            with torch.no_grad():
                results = model(dummy_input)
            
            print(f"✅ Inference successful!")
            print(f"Result type: {type(results)}")
            print(f"Number of results: {len(results)}")
            # 推論結果はResultオブジェクトのリスト
            if len(results) > 0:
                print(f"First result object: {results[0]}")
                print(f"Boxes in first result: {results[0].boxes}")

        except Exception as e:
            print(f"❌ Inference failed: {e}")

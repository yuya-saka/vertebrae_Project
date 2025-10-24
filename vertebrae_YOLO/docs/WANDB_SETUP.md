# W&B（Weights & Biases）設定ガイド

YOLOv8椎体骨折検出プロジェクトでW&Bを使用する方法

---

## 📋 目次

1. [W&Bとは](#wandbとは)
2. [初回セットアップ](#初回セットアップ)
3. [使用方法](#使用方法)
4. [設定のカスタマイズ](#設定のカスタマイズ)
5. [トラブルシューティング](#トラブルシューティング)

---

## W&Bとは

Weights & Biasesは機械学習実験の管理・可視化ツールです。

**主な機能:**
- 学習曲線のリアルタイム可視化
- ハイパーパラメータの管理
- モデルの比較
- チームでの実験共有
- モデルチェックポイントの保存

**TensorBoardとの違い:**
| 項目 | TensorBoard | W&B |
|------|------------|-----|
| インストール | ローカル | クラウドベース |
| 実験管理 | 基本的 | 高度（タグ、検索、比較） |
| チーム共有 | 困難 | 簡単 |
| モデル保存 | ローカルのみ | クラウド保存可能 |

---

## 初回セットアップ

### 1. W&Bアカウント作成

1. https://wandb.ai/signup にアクセス
2. アカウントを作成（GitHubアカウントでも登録可能）

### 2. APIキー取得

1. ログイン後、https://wandb.ai/authorize にアクセス
2. APIキーをコピー

### 3. W&Bログイン

```bash
# W&Bにログイン
wandb login

# APIキーを入力（ペースト）
# または環境変数で設定
export WANDB_API_KEY=your_api_key_here
```

### 4. 依存関係の確認

```bash
# wandbがインストールされているか確認
pip list | grep wandb

# インストールされていない場合
pip install wandb
```

---

## 使用方法

### 方法1: コマンドラインで指定（推奨）

設定ファイルを変更せずに、実行時にW&Bを有効化：

```bash
cd vertebrae_YOLO/run/scripts/train

# W&Bを使用して学習
python train.py logging=wandb

# Fold 1で学習
python train.py logging=wandb split=fold_1

# プロジェクト名をカスタマイズ
python train.py logging=wandb logging.project_name=my_vertebrae_project

# 実験名をカスタマイズ
python train.py logging=wandb logging.experiment_name=baseline_v2
```

### 方法2: デフォルト設定を変更

常にW&Bを使用する場合は、[config.yaml](../run/conf/config.yaml) を編集：

```yaml
defaults:
  - model: yolo_baseline
  - data: yolo_data
  - split: fold_0
  - logging: wandb  # tensorboard → wandb に変更
```

その後、通常通り実行：

```bash
python train.py
```

### 3. W&Bダッシュボードで確認

学習開始後、ターミナルに表示されるURLをクリック：

```
wandb: 🚀 View run at https://wandb.ai/your-username/vertebrae_yolo/runs/xxxxx
```

ダッシュボードで以下を確認可能：
- 学習曲線（loss、mAP、学習率など）
- システムメトリクス（GPU使用率、メモリなど）
- ハイパーパラメータ
- コード（自動保存）

---

## 設定のカスタマイズ

### W&B詳細設定

[run/conf/logging/wandb.yaml](../run/conf/logging/wandb.yaml) を編集：

```yaml
logging:
  logger: wandb
  project_name: vertebrae_yolo
  experiment_name: yolo_baseline

  wandb:
    entity: your-team-name  # チーム名（個人の場合はnull）
    tags:
      - yolov8
      - vertebrae
      - fracture_detection
      - fold_0
    notes: "YOLOv8 baseline for vertebrae fracture detection"
    save_code: true  # コードを自動保存
    save_model: true  # モデルをW&Bに保存
```

### コマンドラインで詳細設定を変更

```bash
# タグを追加
python train.py logging=wandb logging.wandb.tags=[yolov8,experiment1]

# チーム名を指定
python train.py logging=wandb logging.wandb.entity=my-team

# メモを追加
python train.py logging=wandb logging.wandb.notes="Testing new augmentation"
```

---

## 実験管理のベストプラクティス

### 1. プロジェクト名の命名規則

```yaml
project_name: vertebrae_yolo  # プロジェクト全体
```

### 2. 実験名の命名規則

```yaml
# 例: {モデル}_{データ設定}_{特徴}
experiment_name: yolo_baseline_3ch_fold0
experiment_name: yolo_efficientnet_augmented_fold1
experiment_name: yolo_lstm_final_fold2
```

### 3. タグの活用

```yaml
tags:
  - yolov8n  # モデルバリアント
  - 3ch_hu  # データ処理方法
  - fold_0  # Fold番号
  - baseline  # 実験タイプ
  - v1  # バージョン
```

### 4. 5-fold CV実験の管理

各Foldで同じプロジェクト名、異なる実験名を使用：

```bash
# Fold 0-4を順番に実行
for fold in 0 1 2 3 4; do
    python train.py \
        logging=wandb \
        split=fold_${fold} \
        logging.experiment_name=baseline_fold${fold}
done
```

W&Bダッシュボードで5つの実験を並べて比較可能。

---

## 便利な機能

### 1. 複数実験の比較

W&Bダッシュボード上で：
1. 複数の実験を選択
2. "Compare" ボタンをクリック
3. 学習曲線を重ねて表示

### 2. ハイパーパラメータの並列座標プロット

1. "Sweeps" タブを開く
2. 実験結果をハイパーパラメータ別に可視化
3. 最適なパラメータ組み合わせを発見

### 3. モデルの保存と共有

```yaml
logging:
  wandb:
    save_model: true  # 有効化
```

学習終了後、ベストモデルがW&Bクラウドに自動保存され、チームメンバーと共有可能。

---

## トラブルシューティング

### 1. ログインエラー

```bash
# エラー: wandb login failed
# 解決策: APIキーを再入力
wandb login --relogin
```

### 2. ネットワークエラー

```bash
# オフラインモードで実行（ローカルにのみログ保存）
python train.py logging=wandb logging.wandb.mode=offline

# 後でオンラインに同期
wandb sync output/train/wandb/offline-run-xxxxx
```

### 3. W&Bを一時的に無効化

```bash
# 環境変数でW&Bを無効化
export WANDB_MODE=disabled
python train.py logging=wandb

# または TensorBoard に戻す
python train.py logging=tensorboard
```

### 4. ログが多すぎる場合

```yaml
# ログ頻度を減らす
logging:
  wandb:
    log_freq: 50  # 50ステップごとにログ（デフォルト: 10）
```

---

## 参考リンク

- [W&B公式ドキュメント](https://docs.wandb.ai/)
- [PyTorch Lightning + W&B](https://docs.wandb.ai/guides/integrations/lightning)
- [W&B Examples](https://github.com/wandb/examples)

---

**最終更新**: 2025/10/20

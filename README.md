# CINIC-10 ResNet50 画像分類

CINIC-10 の画像を ResNet50 で10クラス分類する実験です。

最初の実験では、短いファイバで最も簡単だと思われる
`25cm_straight/0` のデータを使います。

## このモデルで行うこと

このモデルは、画像を入力として受け取り、その画像が次の10クラスのどれに属するかを予測します。

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

処理の流れは次の通りです。

1. 画像を `224 x 224` にリサイズする
2. ImageNet と同じ平均・標準偏差で正規化する
3. ResNet50 に画像を入力する
4. ResNet50 の最後の層を10クラス分類用に置き換える
5. 各クラスのスコアを出力する
6. 最もスコアが高いクラスを予測結果とする

`--pretrained` を付けると、ImageNet で事前学習された ResNet50 を使います。
そのため、最初から画像特徴をある程度理解しているモデルを、CINIC-10 用に調整して学習できます。

## モデル構成

モデルは [model.py](model.py) にあります。

主な構成は次の通りです。

```text
入力画像
  ↓
Resize(224)
  ↓
Normalize(ImageNet mean/std)
  ↓
ResNet50
  ↓
Linear(model.fc.in_features, 10)
  ↓
10クラスの予測スコア
```

学習では `CrossEntropyLoss` を使います。

最適化には次を使います。

```text
optimizer: SGD
momentum: 0.9
scheduler: StepLR(step_size=7, gamma=0.1)
```

## 使用するデータ

最初に使うデータセット:

```text
/data/dataset/cinic10/25cm_straight/0
```

ディレクトリ構成:

```text
25cm_straight/0/
├── train/
├── valid/
└── test/
```

各フォルダの中に10クラス分のディレクトリがあります。

## 学習・検証・テスト

この実験では、データを次のように使います。

| データ | 役割 |
|---|---|
| `train` | モデルの重みを更新する |
| `valid` | 学習中に性能を確認し、ベストモデルを選ぶ |
| `test` | 学習後に最終性能を確認する |

保存する主なメトリクス:

| メトリクス | 意味 |
|---|---|
| `train_loss` | 学習データでの損失 |
| `val_loss` | 検証データでの損失 |
| `val_acc` | 検証データでの分類精度 |
| `test_acc` | テストデータでの分類精度 |

ベストモデルは `val_acc` が最も高いものです。

## 実験管理

学習と実験管理には Lightning と MLflow を使います。

保存する情報:

| 種類 | 保存する内容 |
|---|---|
| ハイパーパラメータ | 学習率、バッチサイズ、エポック数、weight decay、入力サイズ |
| メトリクス | `train_loss`, `val_loss`, `val_acc`, `test_acc` |
| チェックポイント | ベストモデル、最後のモデル |
| メタデータ | 開始日時、実行コマンド、実行環境、データセット、クラス名、実験説明 |

## ファイル構成

デバッグしやすいように、役割ごとにファイルを分けています。

```text
cinic10_resnet/
├── train.py          # 実行入口、引数、Trainerの設定
├── datamodule.py     # データ読み込み、前処理、DataLoader
├── model.py          # ResNet50とLightningModule
├── experiment.py     # メタデータ保存、MLflow連携
├── requirements.txt  # 必要ライブラリ
└── README.md
```

各ファイルの役割:

| ファイル | 内容 |
|---|---|
| `train.py` | コマンドライン引数、MLflow logger、Trainer、学習実行 |
| `datamodule.py` | `ImageFolder` によるデータ読み込み、画像前処理、DataLoader |
| `model.py` | ResNet50 の作成、学習・検証・テストの処理 |
| `experiment.py` | `metadata.json` の保存、MLflow へのメタデータ登録 |

## 実行方法

`train.py` のデフォルト設定は、すでに `25cm_straight/0` を使うようになっています。

```bash
cd /home0/matsudairah/cinic10_resnet
python3 train.py \
  --pretrained \
  --epochs 20 \
  --batch-size 64 \
  --lr 0.001 \
  --description "25cm_straight/0 の初回実験"
```

よく変更する引数:

| 引数 | 意味 | 例 |
|---|---|---|
| `--lr` | 学習率 | `--lr 0.001` |
| `--batch-size` | バッチサイズ | `--batch-size 64` |
| `--epochs` | 学習エポック数 | `--epochs 20` |
| `--description` | 実験説明 | `--description "lrを変更"` |
| `--run-name` | 実験名 | `--run-name test_lr001` |

## 保存されるファイル

学習結果は `outputs/` に保存されます。

```text
outputs/
├── mlruns/
└── resnet50_YYYYmmdd_HHMMSS/
    ├── checkpoints/
    │   ├── resnet50_best.ckpt
    │   └── last.ckpt
    └── metadata.json
```

`resnet50_best.ckpt` が、検証精度 `val_acc` が最も高かったモデルです。

`metadata.json` には、開始日時、実行環境、実験説明などを保存します。

`mlruns/` には、MLflow の実験ログが保存されます。

`outputs/` や `.ckpt` ファイルは GitHub には保存しない設定にしています。

## MLflowで確認する

PyTorch と MLflow が入っている環境では、次のコマンドで実験ログを確認できます。

```bash
mlflow ui --backend-store-uri outputs/mlruns
```

表示されたURLをブラウザで開くと、実験ごとのハイパーパラメータやメトリクスを確認できます。

## 学習を再開する

途中から再開したい場合は `--resume` を使います。

```bash
python3 train.py \
  --pretrained \
  --epochs 20 \
  --batch-size 64 \
  --resume outputs/resnet50_YYYYmmdd_HHMMSS/checkpoints/last.ckpt
```

この例では、最後に保存されたチェックポイントから再開して、20 epoch まで学習します。

## 別のデータで試す

他のデータセットを使う場合は `--data-dir` を指定します。

```bash
python3 train.py --data-dir /data/dataset/cinic10/50cm_straight/0 --pretrained
python3 train.py --data-dir /data/dataset/cinic10/100cm_bend/0_R6_1roll --pretrained
python3 train.py --data-dir /data/dataset/cinic10/1000cm_bend/0_R6_23roll --pretrained
```

## 必要な環境

次のPythonライブラリが必要です。

```text
torch
torchvision
pytorch-lightning
mlflow
```

`requirements.txt` にも同じ内容を書いています。

もし次のエラーが出る場合は、PyTorch が入っていない環境で実行しています。

```text
ModuleNotFoundError: No module named 'torch'
```

その場合は、PyTorch が入った Python 環境またはコンテナ内で実行してください。

## GitHubへの保存

このディレクトリは Git 管理されています。

変更をGitHubへ保存する流れ:

```bash
git add .
git commit -m "メッセージ"
git push
```

GitHubリポジトリ:

```text
git@github.com:Matsudaira0501/cinic10_resnet.git
```

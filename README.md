# CINIC-10 ResNet Training

This directory is for ResNet-based image classification experiments using the CINIC-10 dataset.

Dataset location:
- `/data/dataset/cinic10`
- First target: `/data/dataset/cinic10/25cm_straight/0`

Implemented:
- `train.py` loads `train/valid/test` with `torchvision.datasets.ImageFolder`
- ResNet50 is created from `torchvision.models`
- The final classification layer is replaced for 10 classes
- Training, validation, checkpoint saving, resume, and test evaluation are included

## ResNet50 で試す手順

- まずは `ResNet50` を `torchvision.models.resnet50(weights=...)` で読み込み
- 最終層を `nn.Linear(model.fc.in_features, 10)` に書き換える
- 入力を `Resize(224)` し、`Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`
- `train/valid/test` サブディレクトリを用意したデータ構成で `ImageFolder` を使う
- 学習後は `resnet50_best.pth` を保存して、`test` で精度を確認する

### 使い方

```bash
cd /home0/matsudairah/cinic10_resnet
python3 train.py --data-dir /data/dataset/cinic10/25cm_straight/0 --pretrained --epochs 20 --batch-size 64
```

- `--data-dir` は `train/valid/test` を含むルートディレクトリ
- `--pretrained` を指定すると ImageNet 事前学習を利用します
- `outputs/resnet50_best.pth` に検証精度が最も高いモデルを保存します
- 各epochの再開用checkpointは `outputs/resnet50_epoch*.pth` に保存します

### 学習を再開する

```bash
python3 train.py \
  --data-dir /data/dataset/cinic10/25cm_straight/0 \
  --pretrained \
  --epochs 20 \
  --batch-size 64 \
  --resume outputs/resnet50_epoch10.pth
```

`--epochs` は最終epoch番号です。例えば `epoch10` から `--epochs 20` で再開すると、11から20epochまで学習します。

### 別のデータで試す

`/data/dataset/cinic10` 以下には同じ形式のデータがあります。例:

```bash
python3 train.py --data-dir /data/dataset/cinic10/1000cm_bend/0_R6_23roll --pretrained
python3 train.py --data-dir /data/dataset/cinic10/50cm_straight/0 --pretrained
python3 train.py --data-dir /data/dataset/cinic10/500cm_bend/0_R6_11roll --pretrained
python3 train.py --data-dir /data/dataset/cinic10/100cm_bend/0_R6_1roll --pretrained
```

### 必要な環境

`torch` と `torchvision` が必要です。手元のホスト環境で `ModuleNotFoundError: No module named 'torch'` が出る場合は、PyTorch入りのPython環境またはSingularity/Apptainerコンテナ内で実行してください。

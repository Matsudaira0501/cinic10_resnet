# CINIC-10 ResNet50 画像分類

CINIC-10 の画像を ResNet50 で分類するための実験用ディレクトリです。

まずは一番簡単だと思われる短いファイバのデータセット、
`25cm_straight/0` から学習します。

## 使用するデータ

最初に使うデータ:

```text
/data/dataset/cinic10/25cm_straight/0
```

このディレクトリの中は、次の構成になっています。

```text
25cm_straight/0/
├── train/
├── valid/
└── test/
```

各フォルダの中には、CINIC-10 の10クラスがあります。

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

## 実行方法

`train.py` のデフォルト設定は、すでに `25cm_straight/0` を使うようになっています。

```bash
cd /home0/matsudairah/cinic10_resnet
python3 train.py --pretrained --epochs 20 --batch-size 64
```

`--pretrained` を付けると、ImageNet で事前学習された ResNet50 を使います。

## 学習で行うこと

`train.py` は次の処理を行います。

- `train/valid/test` を `torchvision.datasets.ImageFolder` で読み込む
- ResNet50 を作成する
- 最後の全結合層を10クラス分類用に変更する
- `train` で学習する
- `valid` で性能を確認する
- 最もよいモデルを保存する
- 最後に `test` で精度を確認する

画像の前処理は次の設定です。

- `Resize(224)`
- ImageNet と同じ平均・標準偏差で `Normalize`

## 保存されるファイル

学習結果は `outputs/` に保存されます。

```text
outputs/
├── resnet50_best.pth
├── resnet50_epoch1.pth
├── resnet50_epoch2.pth
└── ...
```

`resnet50_best.pth` が、検証データで最も精度が高かったモデルです。

`outputs/` や `.pth` ファイルは GitHub には保存しない設定にしています。

## 学習を再開する

途中から再開したい場合は `--resume` を使います。

例:

```bash
python3 train.py \
  --pretrained \
  --epochs 20 \
  --batch-size 64 \
  --resume outputs/resnet50_epoch10.pth
```

この例では、10 epoch 目のチェックポイントから再開して、
20 epoch まで学習します。

## 別のデータで試す

他のデータセットを使う場合は `--data-dir` を指定します。

例:

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
```

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

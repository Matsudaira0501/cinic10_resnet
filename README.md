# POF画像のCINIC-10分類 | ResNet50 / PyTorch

**プラスチック光ファイバ（POF）を通して撮影した画像を、元画像に復元せずに10クラスへ分類する研究用コード**です。ImageNet事前学習済みResNet50の転移学習を含む、初期の分類実験を実装しています。

> **位置づけ：** このリポジトリは研究の**初期実験**を保存したものであり、研究全体の最新モデル・最終評価結果をまとめたものではありません。撮影画像、学習済みモデル、実験ログは同梱していません。

## 概要

- **入力：** POFを介して撮影したRGB画像
- **出力：** CINIC-10の10クラス（airplane / automobile / bird / cat / deer / dog / frog / horse / ship / truck）
- **モデル：** ResNet50の分類ヘッドを10クラス用に変更
- **学習：** CrossEntropyLoss、SGD（momentum 0.9）、StepLR
- **実験管理：** PyTorch Lightningで学習・評価し、MLflowでログを管理

```text
POF撮影画像
    ↓ Resize / Normalize
ResNet50（ImageNet事前学習重みを選択可）
    ↓
全結合層（10クラス）
    ↓
予測ラベル
```

学習時にはランダム水平反転・小角度回転を使い、検証・テスト時はリサイズとImageNetの平均・標準偏差による正規化を行います。これらは**現在のソースコードの実装**に対応します。

## 主なファイル

| ファイル | 内容 |
| --- | --- |
| [train.py](train.py) | CLI引数、学習・検証・テスト、チェックポイント・ログ設定 |
| [datamodule.py](datamodule.py) | 画像読み込み、前処理、DataLoader |
| [model.py](model.py) | ResNet50、損失、精度計算、最適化 |
| [experiment.py](experiment.py) | 実験条件・メタデータの保存 |
| [requirements.txt](requirements.txt) | Python依存ライブラリ |

## データ配置

このリポジトリにデータは含まれません。実行時には、以下のように各分割の下へ**同じ10種類のクラス名のフォルダ**を配置します。

```text
/path/to/dataset/
├── train/
│   ├── airplane/
│   ├── automobile/
│   └── ...（他のクラス）
├── valid/
│   ├── airplane/
│   └── ...
└── test/
    ├── airplane/
    └── ...
```

元の実験では、`/data/dataset/cinic10/25cm_straight/0` を初期のデータパスとして使用していました。手元の環境に合わせて `--data-dir` を変更してください。

## セットアップと実行例

```bash
git clone https://github.com/Matsudaira0501/cinic10_resnet.git
cd cinic10_resnet

python -m pip install -r requirements.txt
python train.py --data-dir /path/to/dataset --pretrained --epochs 20 --batch-size 64 --lr 0.001
```

`--pretrained` を指定するとImageNet事前学習重みを使用します。主な変更可能な引数は `--data-dir`、`--batch-size`、`--epochs`、`--lr`、`--weight-decay`、`--input-size`、`--output-dir`、`--no-cuda` などです。

学習処理の接続を短時間で確認するには `--fast-dev-run` を利用できます。ただしデータは別途必要です。

## 学習結果の記録

- 学習・検証・テストの損失および分類精度を記録
- 検証精度 `val_acc` に基づくベストモデルと最終チェックポイントを保存
- 実行日時、データパス、ハイパーパラメータなどのメタデータを保存
- MLflowのトラッキングログを `outputs/mlruns` に保存（初期設定）

ログの表示例：

```bash
mlflow ui --backend-store-uri outputs/mlruns
```

`outputs/` と学習済みチェックポイントは、GitHubへコミットしない設定にしています。

## 留意事項

**READMEに研究全体の分類精度を引用していない理由：** 本リポジトリのソースは初期実験に対応し、後続の前処理や学習設定の変更をすべて含んでいないためです。再現可能な条件と結果を混同しないようにしています。

関連する卒業研究：[画像謎解きの類題検索とLLMによる解答支援](https://github.com/Matsudaira0501/graduation-research-code)

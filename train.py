from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from datamodule import Cinic10DataModule
from experiment import ExperimentMetadataCallback, collect_metadata
from model import ResNet50Classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet50 on CINIC-10 with Lightning and MLflow")
    parser.add_argument("--data-dir", type=Path, default=Path("/data/dataset/cinic10/25cm_straight/0"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resume", type=Path, default=None, help="Lightning checkpoint path to resume training")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--experiment-name", type=str, default="cinic10_resnet50")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--description", type=str, default="ResNet50 training on CINIC-10 25cm_straight/0")
    parser.add_argument("--tracking-uri", type=str, default="file:outputs/mlruns")
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser.parse_args()


def build_datamodule(args: argparse.Namespace):
    datamodule = Cinic10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        input_size=args.input_size,
        num_workers=args.workers,
    )
    datamodule.setup()
    return datamodule


def build_logger(args: argparse.Namespace, run_name: str):
    return MLFlowLogger(
        experiment_name=args.experiment_name,
        run_name=run_name,
        tracking_uri=args.tracking_uri,
        tags={"description": args.description, "data_dir": str(args.data_dir)},
    )


def build_checkpoint_callback(run_output_dir: Path):
    return ModelCheckpoint(
        dirpath=run_output_dir / "checkpoints",
        filename="resnet50_best",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = build_datamodule(args)
    run_name = args.run_name or datetime.now().strftime("resnet50_%Y%m%d_%H%M%S")
    run_output_dir = args.output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_logger(args, run_name)
    model = ResNet50Classifier(
        num_classes=len(datamodule.classes),
        lr=args.lr,
        weight_decay=args.weight_decay,
        pretrained=args.pretrained,
    )

    metadata = collect_metadata(args, datamodule.classes, datamodule.dataset_sizes)
    metadata["mlflow"] = {
        "tracking_uri": args.tracking_uri,
        "experiment_name": args.experiment_name,
        "run_name": run_name,
    }
    metadata["output_dir"] = str(run_output_dir)

    checkpoint_callback = build_checkpoint_callback(run_output_dir)
    accelerator = "cpu" if args.no_cuda else "auto"
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            ExperimentMetadataCallback(metadata=metadata, output_dir=run_output_dir),
        ],
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
    )

    print(f"Classes ({len(datamodule.classes)}): {datamodule.classes}")
    print(
        "Dataset sizes: "
        f"train={datamodule.dataset_sizes['train']}, "
        f"valid={datamodule.dataset_sizes['valid']}, "
        f"test={datamodule.dataset_sizes['test']}"
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)
    test_ckpt_path = None if args.fast_dev_run else "best"
    trainer.test(model, datamodule=datamodule, ckpt_path=test_ckpt_path)

    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"MLflow tracking URI: {args.tracking_uri}")
    print(f"Run output dir: {run_output_dir}")


if __name__ == "__main__":
    main()

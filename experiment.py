from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger


class ExperimentMetadataCallback(Callback):
    def __init__(self, metadata: dict[str, Any], output_dir: Path):
        self.metadata = metadata
        self.output_dir = output_dir

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = self.output_dir / "metadata.json"
        self._write_metadata(metadata_path)

        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                self._set_start_tags(logger)
                logger.experiment.log_artifact(logger.run_id, str(metadata_path))

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._add_checkpoint_metadata(trainer)
        metadata_path = self.output_dir / "metadata.json"
        self._write_metadata(metadata_path)

        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                self._set_checkpoint_tags(logger)
                logger.experiment.log_artifact(logger.run_id, str(metadata_path))

    def _write_metadata(self, metadata_path: Path):
        metadata_path.write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _set_start_tags(self, logger: MLFlowLogger):
        logger.experiment.set_tag(logger.run_id, "started_at", self.metadata["started_at"])
        logger.experiment.set_tag(logger.run_id, "description", self.metadata["description"])
        logger.experiment.set_tag(logger.run_id, "host", self.metadata["environment"]["host"])
        logger.experiment.set_tag(logger.run_id, "python", self.metadata["environment"]["python"])
        logger.experiment.set_tag(logger.run_id, "torch", self.metadata["environment"]["torch"])
        logger.experiment.set_tag(logger.run_id, "torchvision", self.metadata["environment"]["torchvision"])

    def _add_checkpoint_metadata(self, trainer: pl.Trainer):
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                self.metadata["best_checkpoint"] = callback.best_model_path
                self.metadata["best_model_score"] = (
                    None if callback.best_model_score is None else float(callback.best_model_score)
                )

    def _set_checkpoint_tags(self, logger: MLFlowLogger):
        if self.metadata.get("best_checkpoint"):
            logger.experiment.set_tag(logger.run_id, "best_checkpoint", self.metadata["best_checkpoint"])
        if self.metadata.get("best_model_score") is not None:
            logger.experiment.set_tag(logger.run_id, "best_model_score", str(self.metadata["best_model_score"]))


def collect_metadata(args: argparse.Namespace, classes: list[str], dataset_sizes: dict[str, int]):
    return {
        "description": args.description,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "data": {
            "data_dir": str(args.data_dir),
            "classes": classes,
            "dataset_sizes": dataset_sizes,
        },
        "hyperparameters": {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "input_size": args.input_size,
            "pretrained": args.pretrained,
        },
        "environment": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.replace("\n", " "),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pytorch_lightning": pl.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }

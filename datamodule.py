from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(input_size: int):
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


class Cinic10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        input_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_workers = num_workers
        self.classes: list[str] = []
        self.dataset_sizes: dict[str, int] = {}

    def setup(self, stage: str | None = None):
        train_dir = self.data_dir / "train"
        valid_dir = self.data_dir / "valid"
        test_dir = self.data_dir / "test"

        if not train_dir.exists() or not valid_dir.exists() or not test_dir.exists():
            raise FileNotFoundError(
                f"Expected 'train', 'valid', 'test' subdirectories under {self.data_dir}."
            )

        train_transform, eval_transform = build_transforms(self.input_size)
        self.train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        self.valid_dataset = datasets.ImageFolder(valid_dir, transform=eval_transform)
        self.test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

        if self.valid_dataset.classes != self.train_dataset.classes or self.test_dataset.classes != self.train_dataset.classes:
            raise ValueError(
                "Class folders must match across train, valid, and test directories. "
                f"train={self.train_dataset.classes}, "
                f"valid={self.valid_dataset.classes}, "
                f"test={self.test_dataset.classes}"
            )

        self.classes = list(self.train_dataset.classes)
        self.dataset_sizes = {
            "train": len(self.train_dataset),
            "valid": len(self.valid_dataset),
            "test": len(self.test_dataset),
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

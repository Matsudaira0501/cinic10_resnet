from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def build_transforms(input_size: int):
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def build_dataloaders(data_dir: Path, batch_size: int, input_size: int, num_workers: int):
    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not valid_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected 'train', 'valid', 'test' subdirectories under {data_dir}."
        )

    train_transform, val_transform = build_transforms(input_size)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

    if valid_dataset.classes != train_dataset.classes or test_dataset.classes != train_dataset.classes:
        raise ValueError(
            "Class folders must match across train, valid, and test directories. "
            f"train={train_dataset.classes}, valid={valid_dataset.classes}, test={test_dataset.classes}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    dataset_sizes = {
        "train": len(train_dataset),
        "valid": len(valid_dataset),
        "test": len(test_dataset),
    }
    return train_loader, valid_loader, test_loader, train_dataset.classes, dataset_sizes


def build_model(num_classes: int, pretrained: bool):
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
    except AttributeError:
        model = models.resnet50(pretrained=pretrained)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_resume_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_acc = float(checkpoint.get("best_acc", 0.0))
    return start_epoch, best_acc


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train ResNet50 on CINIC-10 data")
    parser.add_argument("--data-dir", type=Path, default=Path("/data/dataset/cinic10/25cm_straight/0"),
                        help="Root directory containing train/valid/test subfolders")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume training")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader, test_loader, classes, dataset_sizes = build_dataloaders(
        args.data_dir, args.batch_size, args.input_size, args.workers
    )

    model = build_model(num_classes=len(classes), pretrained=args.pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print(f"Device: {device}")
    print(f"Classes ({len(classes)}): {classes}")
    print(
        "Dataset sizes: "
        f"train={dataset_sizes['train']}, valid={dataset_sizes['valid']}, test={dataset_sizes['test']}"
    )

    start_epoch = 1
    best_acc = 0.0
    if args.resume is not None:
        start_epoch, best_acc = load_resume_checkpoint(args.resume, model, optimizer, scheduler, device)
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best valid acc: {best_acc:.4f})")

    best_path = args.output_dir / "resnet50_best.pth"
    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{args.epochs} - {elapsed:.1f}s")
        print(f"  train loss: {train_loss:.4f}, train acc: {train_acc:.4f}")
        print(f"  valid loss: {valid_loss:.4f}, valid acc: {valid_acc:.4f}")

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "classes": classes,
                "best_acc": best_acc,
                "args": vars(args),
            }, best_path)
            print(f"  saved best model: {best_path}")

        checkpoint_path = args.output_dir / f"resnet50_epoch{epoch}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "classes": classes,
            "best_acc": best_acc,
            "args": vars(args),
        }, checkpoint_path)

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch']} for test evaluation")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()

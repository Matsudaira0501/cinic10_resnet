from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def build_resnet50(num_classes: int, pretrained: bool):
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
    except AttributeError:
        model = models.resnet50(pretrained=pretrained)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ResNet50Classifier(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        lr: float,
        weight_decay: float,
        pretrained: bool,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_resnet50(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        inputs, labels = batch
        logits = self(inputs)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, acc = self._shared_eval_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx: int):
        loss, acc = self._shared_eval_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"test_loss": loss, "test_acc": acc}

    def _shared_eval_step(self, batch):
        inputs, labels = batch
        logits = self(inputs)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

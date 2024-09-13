# PyTorch Lightning class used for I/Q models

import torch
import torchmetrics

import torch.nn.functional as F

from matplotlib import pyplot as plt
from pathlib import Path
from pytorch_lightning import LightningModule
from torch import optim

import torch
import torch.nn as nn
import torch.nn.functional as F


class mod_relu(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.b = torch.nn.parameter.Parameter(torch.rand(1) * 0.25)
        self.b.requiresGrad = True

    def forward(self, x):
        # This is probably not correct (specifically torch.abs(self.b)) but it works
        return F.relu(torch.abs(x) + torch.abs(self.b)) * torch.exp(
            1.0j * torch.angle(x)
        )


def calculate_output_length(length_in, kernel_size, stride=1, padding=0, dilation=1):
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class SimpleRealNet(nn.Module):
    def __init__(self, n_classes, n_input):
        super(SimpleRealNet, self).__init__()
        self.conv1 = nn.Conv1d(2, 8, 3, 1)
        self.conv2 = nn.Conv1d(8, 16, 3, 1)
        n_fc = 16 * calculate_output_length(calculate_output_length(n_input, 3), 3)
        self.fc1 = nn.Linear(n_fc, 8)
        self.fc2 = nn.Linear(8, n_classes)
        self.mod_relu = F.relu

    def forward(self, x):
        x = self.conv1(x)
        x = self.mod_relu(x)
        # x = F.tanh(x)
        x = self.conv2(x)
        x = self.mod_relu(x)
        # x = F.tanh(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.mod_relu(x)
        # x = F.tanh(x)
        x = self.fc2(x)
        x = x.abs()
        output = F.log_softmax(x, dim=1)
        return output


class ExampleNetwork(LightningModule):
    def __init__(
        self,
        model,
        data_loader=None,
        val_data_loader=None,
        num_classes=None,
        extra_metrics=True,
        logs_dir=None,
        learning_rate=None,
        class_list=None,
    ):
        super(ExampleNetwork, self).__init__()
        self.mdl = model
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader

        # Hyperparameters
        self.lr = learning_rate if learning_rate else 0.001
        self.batch_size = data_loader.batch_size

        self.num_classes = num_classes
        self.extra_metrics = extra_metrics

        if self.num_classes is None:
            self.extra_metrics = False

        self.logs_dir = logs_dir

        self.class_list = class_list

        # Metrics
        if self.extra_metrics:
            self.train_acc = torchmetrics.classification.Accuracy(
                task="multiclass", num_classes=num_classes
            )
            self.valid_acc = torchmetrics.classification.Accuracy(
                task="multiclass", num_classes=num_classes
            )
            self.confusion_mat = torchmetrics.classification.ConfusionMatrix(
                task="multiclass", normalize="true", num_classes=num_classes
            )

    def forward(self, x):
        return self.mdl(x)

    def predict(self, x):
        with torch.no_grad():
            out = self.forward(x)
        return out

    def configure_optimizers(self):
        # return optim.Adam(self.parameters(), lr=self.lr)
        return optim.AdamW(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return self.data_loader

    def training_step(self, batch, batch_nb):
        x, y = batch
        # print(x.shape)
        # print(y.shape)
        y = torch.squeeze(y.to(torch.int64))
        preds = self(x.float())

        if self.extra_metrics:
            self.train_acc(preds, y)
            self.log("train_acc", self.train_acc, on_step=True, on_epoch=False)

        loss = F.cross_entropy(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def val_dataloader(self):
        return self.val_data_loader

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y = torch.squeeze(y.to(torch.int64))
        preds = self(x.float())

        val_loss = F.cross_entropy(preds, y)

        if self.extra_metrics:
            self.valid_acc(preds, y)
            self.confusion_mat.update(preds, y)
            self.log("valid_acc", self.valid_acc, on_step=True, on_epoch=True)

        self.log("val_loss", val_loss, prog_bar=True)
        return {"val_loss": val_loss}

    def on_validation_end(self):
        if self.extra_metrics:
            self.confusion_mat.compute()
            fig, ax = self.confusion_mat.plot(labels=self.class_list)
            fig.savefig(
                Path(self.logs_dir, f"confusion_matrix_{self.current_epoch}.png")
            )  # save the figure to file
            plt.close(fig)
            self.confusion_mat.reset()

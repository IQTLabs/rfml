# PyTorch Lightning class used for I/Q models 

import torch
import torchmetrics

import torch.nn.functional as F

from matplotlib import pyplot as plt
from pathlib import Path
from pytorch_lightning import LightningModule
from torch import optim



class ExampleNetwork(LightningModule):
    def __init__(self, model, data_loader=None, val_data_loader=None, num_classes=None, extra_metrics=True, logs_dir=None):
        super(ExampleNetwork, self).__init__()
        self.mdl = model
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader

        # Hyperparameters
        self.lr = 0.001
        self.batch_size = data_loader.batch_size

        self.num_classes = num_classes
        self.extra_metrics = extra_metrics

        if self.num_classes is None:
            self.extra_metrics = False

        self.logs_dir = logs_dir
            
        # Metrics
        if self.extra_metrics:
            self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
            self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
            self.confusion_mat = torchmetrics.classification.ConfusionMatrix(task="multiclass", normalize='true', num_classes=num_classes)

    def forward(self, x):
        return self.mdl(x)

    def predict(self, x):
        with torch.no_grad():
            out = self.forward(x)
        return out

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

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
            self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        
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
            self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)
         
        
        self.log("val_loss", val_loss, prog_bar=True)
        return {"val_loss": val_loss}

    def on_validation_end(self):
        if self.extra_metrics:
            self.confusion_mat.compute()
            fig, ax = self.confusion_mat.plot()
            fig.savefig(Path(self.logs_dir, f"confusion_matrix_{self.current_epoch}.png"))  # save the figure to file
            plt.close(fig) 
            self.confusion_mat.reset()
            
# class CustomNetwork(LightningModule):
#     def __init__(self, model, data_loader=None, val_data_loader=None):
#         super(CustomNetwork, self).__init__()
#         self.mdl = model
#         self.data_loader = data_loader
#         self.val_data_loader = val_data_loader

#         # Hyperparameters
#         self.lr = 0.001
#         if data_loader:
#             self.batch_size = data_loader.batch_size

#     def forward(self, x):
#         return self.mdl(x)

#     def predict(self, x):
#         with torch.no_grad():
#             out = self.forward(x)
#         return out

#     def configure_optimizers(self):
#         return optim.Adam(self.parameters(), lr=self.lr)

#     def train_dataloader(self):
#         return self.data_loader

#     def training_step(self, batch, batch_nb):
#         x, y = batch
#         y = torch.squeeze(y.to(torch.int64))
#         loss = F.cross_entropy(self(x.float()), y)
#         return {"loss": loss}

#     def val_dataloader(self):
#         return self.val_data_loader

#     def validation_step(self, batch, batch_nb):
#         x, y = batch
#         y = torch.squeeze(y.to(torch.int64))
#         val_loss = F.cross_entropy(self(x.float()), y)
#         self.log("val_loss", val_loss, prog_bar=True)
#         return {"val_loss": val_loss}


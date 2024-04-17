#!/usr/bin/env python
# coding: utf-8

# In[33]:


from sigmf_pytorch_dataset import SigMFDataset
from torchsig.utils.visualize import IQVisualizer, SpectrogramVisualizer, two_channel_to_complex
from torchsig.utils.dataset import SignalDataset
from torchsig.datasets.sig53 import Sig53
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from typing import List
from tqdm import tqdm
from datetime import datetime
import numpy as np
import os
from pathlib import Path
import torchmetrics


from torchsig.models.iq_models.efficientnet.efficientnet import efficientnet_b4
#from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import DeviceStatsMonitor
from torchsig.utils.cm_plotter import plot_confusion_matrix

import lightning as L

#from pytorch_lightning import LightningModule, Trainer
from sklearn.metrics import classification_report
from torchsig.datasets.sig53 import Sig53
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
import torchsig.transforms as ST
import numpy as np
import torchsig
import torch
import os
from sigmf_db_dataset import SigMFDB
from sigmf_pytorch_dataset import SigMFDataset


# In[6]:


from torchsig.transforms import (
    Compose,
    IQImbalance,
    Normalize,
    RandomApply,
    RandomFrequencyShift,
    RandomPhaseShift,
    RandomResample,
    RandomTimeShift,
    RayleighFadingChannel,
    TargetSNR,
    ComplexTo2D,
)


# In[ ]:


dataset_path = "./dev_data/torchsig_test/"
num_iq_samples = 1024


# In[35]:


logs_dir = datetime.now().strftime('logs/%H_%M_%S_%m_%d_%Y')
logs_dir
logs_dir = Path(logs_dir)
logs_dir.mkdir(parents=True)


# In[3]:


def visualize_dataset(dataset_path, num_iq_samples):
    dataset = SigMFDataset( root=dataset_path, sample_count= num_iq_samples, allowed_filetypes=[".sigmf-data"])
    dataset_class_counts = {class_name:0 for class_name in dataset.class_list}
    for data,label in dataset:
        dataset_class_counts[dataset.class_list[label]] += 1
    print(f"{len(dataset)=}")
    print(dataset_class_counts)
    
    
    # In[4]:
    
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=100,
        shuffle=True,
    )
    
    visualizer = IQVisualizer(
        data_loader=data_loader
    )
    
    for figure in iter(visualizer):
        figure.set_size_inches(16, 16)
        plt.show()
        plt.savefig(Path(logs_dir, "dataset.png"))
        break

visualize_dataset(dataset_path, num_iq_samples)

# # SigMF based Model Training

# In[5]:



eb_no=False
level2 = Compose(
    [
        RandomApply(RandomPhaseShift((-1, 1)), 0.9),
        RandomApply(RandomTimeShift((-32, 32)), 0.9),
        RandomApply(RandomFrequencyShift((-0.16, 0.16)), 0.7),
        RandomApply(
            RayleighFadingChannel((0.05, 0.5), power_delay_profile=(1.0, 0.5, 0.1)),
            0.5,
        ),
        RandomApply(
            IQImbalance(
                (-3, 3),
                (-np.pi * 1.0 / 180.0, np.pi * 1.0 / 180.0),
                (-0.1, 0.1),
            ),
            0.9,
        ),
        RandomApply(
            RandomResample((0.75, 1.5), num_iq_samples=num_iq_samples),
            0.5,
        ),
        #TargetSNR((-2, 30), eb_no=eb_no),
        Normalize(norm=np.inf),
        ComplexTo2D(),
    ]
)


# ### Load the SigMF File dataset
# and generate the class list

# In[7]:


# transform = ST.Compose([
#     # ST.RandomPhaseShift(phase_offset=(-1, 1)),
#     ST.Normalize(norm=np.inf),
#     ST.ComplexTo2D(),
# ])
transform = level2
# class_list = ['mini2_video']

dataset = SigMFDataset( root=dataset_path,
                       sample_count=num_iq_samples,
                       transform=transform,
                       only_first_samples=False,
                       # class_list=class_list,
)





train_data, val_data = torch.utils.data.random_split(dataset, [0.8, 0.2])

sampler = dataset.get_weighted_sampler(indices=train_data.indices)

train_dataloader = DataLoader(
    dataset=train_data, #sig53_clean_train ,
    batch_size=180,
    num_workers=16,
    sampler=sampler,
    # shuffle=True,
    drop_last=True,
)
val_dataloader = DataLoader(
    dataset=val_data, #sig53_clean_train ,
    batch_size=180,
    num_workers=16,
    shuffle=False,
    drop_last=True,
)


train_class_counts = dataset.get_class_counts(indices=train_data.indices)
train_class_counts = {dataset.class_list[k]:v for k,v in train_class_counts.items()}
val_class_counts = dataset.get_class_counts(indices=val_data.indices)
val_class_counts = {dataset.class_list[k]:v for k,v in val_class_counts.items()}

print(f"{len(train_data)=}, {train_class_counts=}")
print(f"{len(val_data)=}, {val_class_counts=}")


# In[11]:


model = efficientnet_b4(
    pretrained=True,
    path="efficientnet_b4.pt",
    num_classes=len(dataset.class_list),
)
#model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features=len(dataset.class_list), bias=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# In[12]:


class ExampleNetwork(L.LightningModule):
    def __init__(self, model, data_loader, val_data_loader, num_classes):
        super(ExampleNetwork, self).__init__()
        self.mdl = model
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader

        # Hyperparameters
        self.lr = 0.001
        self.batch_size = data_loader.batch_size

        # Metrics
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
        self.valid_acc(preds, y)
        self.confusion_mat.update(preds, y)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)
         
        
        self.log("val_loss", val_loss, prog_bar=True)
        return {"val_loss": val_loss}

    def on_validation_end(self):
        self.confusion_mat.compute()
        fig, ax = self.confusion_mat.plot()
        fig.savefig(Path(logs_dir, f"confusion_matrix_{self.current_epoch}.png"))  # save the figure to file
        plt.close(fig) 
        self.confusion_mat.reset()



example_model = ExampleNetwork(model, train_dataloader, val_dataloader, num_classes=len(dataset.class_list))


# print(f"{dataset[0][0].shape=}")
# print(f"{type(dataset[0][0])=}")
# print(f"{model(torch.from_numpy(dataset[0][0]).to(device).float())=}")
# print(f"{example_model.predict(torch.from_numpy(dataset[0][0]).to(device).float())=}")
# In[ ]:


# Setup checkpoint callbacks
checkpoint_filename = "{}/checkpoints/checkpoint".format(os.getcwd())
checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    filename=checkpoint_filename,
    save_top_k=True,
    monitor="val_loss",
    mode="min",
)
print("Doing stuff")
# Create and fit trainer
epochs = 40
trainer = L.Trainer(
    max_epochs=epochs, callbacks=[DeviceStatsMonitor(),checkpoint_callback], accelerator="gpu", devices=1, profiler="advanced"
)
trainer.fit(example_model)


# In[21]:


checkpoint_callback.best_model_path


# ## Evaluate the Trained Model

# In[22]:


# Load best checkpoint
checkpoint = torch.load(checkpoint_callback.best_model_path, map_location=lambda storage, loc: storage)
example_model.load_state_dict(checkpoint["state_dict"], strict=False)
example_model = example_model.eval()
example_model = example_model.cuda() if torch.cuda.is_available() else example_model


# In[23]:


# Infer results over validation set
num_test_examples = len(val_data)
# num_classes = 5 #len(list(Sig53._idx_to_name_dict.values()))
# y_raw_preds = np.empty((num_test_examples,num_classes))
y_preds = np.zeros((num_test_examples,))
y_true = np.zeros((num_test_examples,))
y_true_list = []
y_preds_list = []
with torch.no_grad():
    example_model.eval()
    #for i in tqdm(range(0,num_test_examples)):
    for data, label in tqdm(val_dataloader):
        # Retrieve data
        # idx = i # Use index if evaluating over full dataset
        # data, label = val_data[idx]
        # Infer
        data = data.float()
        #data = torch.from_numpy(data).float()
        #data = torch.from_numpy(np.expand_dims(data,0)).float()
        data = data.cuda() if torch.cuda.is_available() else data
        pred_tmp = example_model.predict(data)
        pred_tmp = pred_tmp.cpu().numpy() if torch.cuda.is_available() else pred_tmp


        y_preds_list.extend(np.argmax(pred_tmp, axis=1).tolist())
        y_true_list.extend(label.tolist())
        # # Argmax
        # y_preds[i] = np.argmax(pred_tmp)
        # # Store label
        # y_true[i] = label


# In[24]:


y_preds = y_preds_list
y_true = y_true_list


# In[28]:


acc = np.sum(np.asarray(y_preds)==np.asarray(y_true))/len(y_true)
plot_confusion_matrix(
    y_true, 
    y_preds, 
    classes=dataset.class_list,
    normalize=True,
    title="Example Modulations Confusion Matrix\nTotal Accuracy: {:.2f}%".format(acc*100),
    text=True,
    rotate_x_text=90,
    figsize=(16,9),
)
#plt.show()
plt.savefig(Path(logs_dir, "confusion_matrix.png"))


# In[26]:


print(f"{len(train_data)=}, {train_class_counts=}")
print(f"{len(val_data)=}, {val_class_counts=}")


# In[27]:


dataset.class_list


# In[ ]:





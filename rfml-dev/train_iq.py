# I/Q model training script

from argparse import ArgumentParser, BooleanOptionalAction
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
# from lightning.pytorch.callbacks import DeviceStatsMonitor
from torchsig.utils.cm_plotter import plot_confusion_matrix
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning import Trainer

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
from models import ExampleNetwork

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

# # dataset_path = "./dev_data/torchsig_train/"
# dataset_path = "./data/gamutrf/gamutrf-sd-gr-ieee-wifi/v2_host/gain_40/"
# print(f"{dataset_path=}")
# num_iq_samples = 1024
# only_use_start_of_burst = True

# logs_dir = datetime.now().strftime('logs/%H_%M_%S_%m_%d_%Y')

# logs_dir = Path(logs_dir)
# logs_dir.mkdir(parents=True)

# epochs = 40
# batch_size = 180 
# class_list = ['anom_wifi','wifi']




def train_iq(
    train_dataset_path,
    val_dataset_path = None, 
    num_iq_samples = 1024, 
    only_use_start_of_burst = True,
    epochs = 40, 
    batch_size = 180, 
    class_list = None, 
    logs_dir = None, 
    output_dir = None,
):
    print(locals())
    if logs_dir is None:
        logs_dir = datetime.now().strftime('iq_logs/%m_%d_%Y_%H_%M_%S')
    if output_dir is None:
        output_dir = "./"
    output_dir = Path(output_dir)
    logs_dir = Path(output_dir,logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_dataset(train_dataset_path, num_iq_samples, logs_dir, class_list=class_list)

    # # SigMF based Model Training
    
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
    
    
    # transform = ST.Compose([
    #     # ST.RandomPhaseShift(phase_offset=(-1, 1)),
    #     ST.Normalize(norm=np.inf),
    #     ST.ComplexTo2D(),
    # ])

    val_transform = ST.Compose([
        ST.Normalize(norm=np.inf),
        ST.ComplexTo2D(),
    ])
    train_transform = level2
    
    
    
    ###
    if val_dataset_path:
        train_dataset = SigMFDataset( 
            root=train_dataset_path,
            sample_count=num_iq_samples,
            transform=train_transform,
            only_first_samples=only_use_start_of_burst,
            class_list=class_list,
        )
        val_dataset = SigMFDataset( 
            root=val_dataset_path,
            sample_count=num_iq_samples,
            transform=val_transform,
            only_first_samples=only_use_start_of_burst,
            class_list=class_list,
        )
        sampler = train_dataset.get_weighted_sampler()
        
        train_class_counts = train_dataset.get_class_counts()
        train_class_counts = {train_dataset.class_list[k]:v for k,v in train_class_counts.items()}
        val_class_counts = val_dataset.get_class_counts()
        val_class_counts = {val_dataset.class_list[k]:v for k,v in val_class_counts.items()}

        class_list = class_list if class_list else train_dataset.class_list
    ###
    else:
        dataset = SigMFDataset( 
            root=train_dataset_path,
            sample_count=num_iq_samples,
            transform=train_transform,
            only_first_samples=only_use_start_of_burst,
            class_list=class_list,    
        )
    
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
        sampler = dataset.get_weighted_sampler(indices=train_dataset.indices)

        train_class_counts = dataset.get_class_counts(indices=train_dataset.indices)
        train_class_counts = {dataset.class_list[k]:v for k,v in train_class_counts.items()}
        val_class_counts = dataset.get_class_counts(indices=val_dataset.indices)
        val_class_counts = {dataset.class_list[k]:v for k,v in val_class_counts.items()}

        class_list = class_list if class_list else dataset.class_list
        
    print(f"{len(train_dataset)=}, {train_class_counts=}")
    print(f"{len(val_dataset)=}, {val_class_counts=}")
    
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size,
        num_workers=16,
        sampler=sampler,
        # shuffle=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size,
        num_workers=16,
        shuffle=False,
        drop_last=True,
    )
    
    
    model = efficientnet_b4(
        pretrained=True,
        path="efficientnet_b4.pt",
        num_classes=len(class_list),
    )
    #model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features=len(class_list), bias=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    
    example_model = ExampleNetwork(model, train_dataloader, val_dataloader, num_classes=len(class_list), logs_dir=logs_dir)
    
    
    # Setup checkpoint callbacks
    checkpoint_filename = f"{str(output_dir)}/iq_checkpoints/checkpoint"
    checkpoint_callback = ModelCheckpoint(
        filename=checkpoint_filename,
        save_top_k=True,
        monitor="val_loss",
        mode="min",
    )
    # Create and fit trainer
    
    trainer = Trainer(
        max_epochs=epochs, callbacks=[DeviceStatsMonitor(),checkpoint_callback], accelerator="gpu", devices=1, profiler="advanced"
    )
    trainer.fit(example_model)
    
    # checkpoint_callback.best_model_path
    
    
    # ## Evaluate the Trained Model
    
    
    # Load best checkpoint
    checkpoint = torch.load(checkpoint_callback.best_model_path, map_location=lambda storage, loc: storage)
    example_model.load_state_dict(checkpoint["state_dict"], strict=False)
    example_model = example_model.eval()
    example_model = example_model.cuda() if torch.cuda.is_available() else example_model
    
    # Infer results over validation set
    num_test_examples = len(val_dataset)
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
            # data, label = val_dataset[idx]
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
    
    y_preds = y_preds_list
    y_true = y_true_list
    
    acc = np.sum(np.asarray(y_preds)==np.asarray(y_true))/len(y_true)
    plot_confusion_matrix(
        y_true, 
        y_preds, 
        classes=class_list,
        normalize=True,
        title="Example Modulations Confusion Matrix\nTotal Accuracy: {:.2f}%".format(acc*100),
        text=True,
        rotate_x_text=90,
        figsize=(16,9),
    )
    #plt.show()
    plt.savefig(Path(logs_dir, "confusion_matrix.png"))
    
    
    print(f"{len(train_dataset)=}, {train_class_counts=}")
    print(f"{len(val_dataset)=}, {val_class_counts=}")
    
    

def visualize_dataset(dataset_path, num_iq_samples, logs_dir, class_list):
    print("\nVisualizing Dataset\n")
    dataset = SigMFDataset( root=dataset_path, sample_count= num_iq_samples, allowed_filetypes=[".sigmf-data"], class_list=class_list)
    dataset_class_counts = {class_name:0 for class_name in dataset.class_list}
    for data,label in dataset:
        dataset_class_counts[dataset.class_list[label]] += 1
    print(f"{len(dataset)=}")
    print(dataset_class_counts)
    
    
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

def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "train_dataset_path",
        type=str,
        nargs='+',
        help="Path to training dataset",
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        nargs='+',
        help="Path to validation dataset",
    )
    parser.add_argument(
        "--num_iq_samples",
        type=int,
        default=1024,
        help="Number of I/Q samples per example",
    )
    parser.add_argument(
        "--only_use_start_of_burst",
        help="Only use start of burst for each example",
        default=True,
        action=BooleanOptionalAction,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=180,
        help="Training example batch size",
    )
    parser.add_argument(
        "--class_list",
        nargs='+',
        type=str,
        help="List of classes to use",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        help="Path to write logs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to write output",
    )

    return parser

if __name__ == "__main__":

    options = argument_parser().parse_args()
    train_iq(
        train_dataset_path = options.train_dataset_path,
        val_dataset_path = options.val_dataset_path,
        num_iq_samples = options.num_iq_samples, 
        only_use_start_of_burst = options.only_use_start_of_burst,
        epochs = options.epochs, 
        batch_size = options.batch_size, 
        class_list = options.class_list, 
        logs_dir = options.logs_dir, 
        output_dir = options.output_dir,
    )



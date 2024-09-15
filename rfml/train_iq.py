# I/Q model training script

from argparse import ArgumentParser, BooleanOptionalAction
from rfml.sigmf_pytorch_dataset import SigMFDataset
from torchsig.utils.visualize import (
    IQVisualizer,
    SpectrogramVisualizer,
    two_channel_to_complex,
)
from torchsig.utils.dataset import SignalDataset
from torchsig.datasets.sig53 import Sig53
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from typing import List
from tqdm import tqdm
from datetime import datetime
import numpy as np
import os
import json
from pathlib import Path

from torchsig.models.iq_models.efficientnet.efficientnet import (
    efficientnet_b0,
    efficientnet_b4,
)

# from lightning.pytorch.callbacks import DeviceStatsMonitor
from torchsig.utils.cm_plotter import plot_confusion_matrix
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from scipy import signal as sp

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
from rfml.sigmf_pytorch_dataset import SigMFDataset
from rfml.models import ExampleNetwork, SimpleRealNet
from rfml.export_model import *

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


def train_iq(
    train_dataset_path,
    val_dataset_path=None,
    num_iq_samples=1024,
    only_use_start_of_burst=True,
    epochs=40,
    batch_size=180,
    class_list=None,
    logs_dir=None,
    output_dir=None,
    learning_rate=None,
    experiment_name=None,
    early_stop=10,
    train_limit=1,
    val_limit=1,
):
    print(f"\nI/Q MODEL TRAINING")
    if logs_dir is None:
        logs_dir = datetime.now().strftime("iq_logs/%m_%d_%Y_%H_%M_%S")
    if output_dir is None:
        output_dir = "./"
    output_dir = Path(output_dir)
    logs_dir = Path(output_dir, logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # # SigMF based Model Training

    eb_no = False
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
            # TargetSNR((-2, 30), eb_no=eb_no),
            Normalize(norm=np.inf),
            ComplexTo2D(),
        ]
    )

    # TODO: add user parameters for
    # transforms
    # use pretrained weights

    basic_transform = ST.Compose(
        [
            # ST.RandomPhaseShift(phase_offset=(-1, 1)),
            # ST.AddNoise(),
            # ST.AutomaticGainControl(),
            # ST.Normalize(norm=2),
            ST.Normalize(norm=np.inf),
            ST.ComplexTo2D(),
        ]
    )

    val_transform = ST.Compose(
        [
            # ST.AutomaticGainControl(),
            # ST.Normalize(norm=2),
            ST.Normalize(norm=np.inf),
            ST.ComplexTo2D(),
        ]
    )

    visualize_transform = ST.Compose(
        [
            # ST.AddNoise(),
            # ST.AutomaticGainControl()
        ]
    )

    # train_transform = level2
    train_transform = basic_transform

    visualize_dataset(
        train_dataset_path,
        num_iq_samples,
        logs_dir,
        class_list=class_list,
        only_use_start_of_burst=only_use_start_of_burst,
        transform=visualize_transform,
    )

    if val_dataset_path:
        original_train_dataset = SigMFDataset(
            root=train_dataset_path,
            sample_count=num_iq_samples,
            transform=train_transform,
            only_first_samples=only_use_start_of_burst,
            class_list=class_list,
        )
        original_val_dataset = SigMFDataset(
            root=val_dataset_path,
            sample_count=num_iq_samples,
            transform=val_transform,
            only_first_samples=only_use_start_of_burst,
            class_list=class_list,
        )

        train_dataset, _ = torch.utils.data.random_split(
            original_train_dataset, [train_limit, 1 - train_limit]
        )
        val_dataset, _ = torch.utils.data.random_split(
            original_val_dataset, [val_limit, 1 - val_limit]
        )

        sampler = original_train_dataset.get_weighted_sampler(
            indices=train_dataset.indices
        )

        train_class_counts = original_train_dataset.get_class_counts(
            indices=train_dataset.indices
        )
        train_class_counts = {
            original_train_dataset.class_list[k]: v
            for k, v in train_class_counts.items()
        }
        val_class_counts = original_val_dataset.get_class_counts(
            indices=val_dataset.indices
        )
        val_class_counts = {
            original_val_dataset.class_list[k]: v for k, v in val_class_counts.items()
        }

        class_list = class_list if class_list else original_train_dataset.class_list
    else:
        dataset = SigMFDataset(
            root=train_dataset_path,
            sample_count=num_iq_samples,
            transform=train_transform,
            only_first_samples=only_use_start_of_burst,
            class_list=class_list,
        )
        train_dataset, val_dataset, _ = torch.utils.data.random_split(
            dataset, [train_limit * 0.8, train_limit * 0.2, 1 - train_limit]
        )
        sampler = dataset.get_weighted_sampler(indices=train_dataset.indices)

        train_class_counts = dataset.get_class_counts(indices=train_dataset.indices)
        train_class_counts = {
            dataset.class_list[k]: v for k, v in train_class_counts.items()
        }
        val_class_counts = dataset.get_class_counts(indices=val_dataset.indices)
        val_class_counts = {
            dataset.class_list[k]: v for k, v in val_class_counts.items()
        }

        class_list = class_list if class_list else dataset.class_list

    print(f"\nTraining dataset information:")
    print(f"{len(train_dataset)=}, {train_class_counts=}")
    print(f"\nValidation dataset information:")
    print(f"{len(val_dataset)=}, {val_class_counts=}")
    print("")

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=24,
        sampler=sampler,
        # shuffle=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=24,
        shuffle=False,
        drop_last=True,
    )

    # TODO: add feature to specify model

    # model = SimpleRealNet(
    #     n_classes=len(class_list),
    #     n_input=num_iq_samples,
    # )

    model = efficientnet_b0(
        pretrained=True,
        path="efficientnet_b0.pt",
        num_classes=len(class_list),
        drop_path_rate=0.4,
        drop_rate=0.4,
    )
    # model = efficientnet_b4(
    #     pretrained=True,
    #     path="efficientnet_b4.pt",
    #     num_classes=len(class_list),
    #     drop_path_rate=0.2,
    #     drop_rate=0.6,
    # )
    # model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features=len(class_list), bias=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    example_model = ExampleNetwork(
        model,
        train_dataloader,
        val_dataloader,
        num_classes=len(class_list),
        logs_dir=logs_dir,
        learning_rate=learning_rate,
        class_list=class_list,
    )

    # Setup checkpoint callbacks
    checkpoint_filename = f"checkpoints/checkpoint"
    checkpoint_callback = ModelCheckpoint(
        dirpath=logs_dir,
        filename=checkpoint_filename,
        save_top_k=True,
        monitor="val_loss",
        mode="min",
    )

    index_to_name_file = Path(
        logs_dir, "index_to_name.json"
    )  # f"lightning_logs/{experiment_name}/index_to_name.json"
    index_to_name = {i: class_list[i] for i in range(len(class_list))}
    index_to_name_object = json.dumps(index_to_name, indent=4)
    with open(index_to_name_file, "w") as outfile:
        outfile.write(index_to_name_object)

    # Create and fit trainer
    experiment_name = experiment_name if experiment_name else 1
    logger = TensorBoardLogger(
        save_dir="tensorboard_logs",
        # version=experiment_name,
        name=experiment_name,  # "lightning_logs"
    )
    trainer = Trainer(
        max_epochs=epochs,
        callbacks=[
            EarlyStopping(
                monitor="val_loss", mode="min", patience=early_stop, verbose=True
            ),
            checkpoint_callback,
        ],
        accelerator="gpu",
        devices=1,
        logger=logger,
        # profiler="simple",
        default_root_dir=logs_dir,
    )
    print(f"\nStarting training...")
    trainer.fit(example_model)

    # checkpoint_callback.best_model_path

    # ## Evaluate the Trained Model

    # Load best checkpoint
    checkpoint = torch.load(
        checkpoint_callback.best_model_path, map_location=lambda storage, loc: storage
    )
    example_model.load_state_dict(checkpoint["state_dict"], strict=False)
    example_model = example_model.eval()
    example_model = example_model.cuda() if torch.cuda.is_available() else example_model

    # Infer results over validation set
    num_test_examples = len(val_dataset)
    y_preds = np.zeros((num_test_examples,))
    y_true = np.zeros((num_test_examples,))
    y_true_list = []
    y_preds_list = []

    print(f"\nStarting final validation...")
    with torch.no_grad():
        example_model.eval()
        for data, label in tqdm(val_dataloader):
            # Retrieve data
            # idx = i # Use index if evaluating over full dataset
            # data, label = val_dataset[idx]
            # Infer
            data = data.float()
            # data = torch.from_numpy(data).float()
            # data = torch.from_numpy(np.expand_dims(data,0)).float()
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

    acc = np.sum(np.asarray(y_preds) == np.asarray(y_true)) / len(y_true)
    plot_confusion_matrix(
        y_true,
        y_preds,
        classes=class_list,
        normalize=True,
        title="Example Modulations Confusion Matrix\nTotal Accuracy: {:.2f}%".format(
            acc * 100
        ),
        text=True,
        rotate_x_text=90,
        figsize=(16, 9),
    )
    # plt.show()
    plt.savefig(Path(logs_dir, "confusion_matrix.png"))

    print(f"\n\nI/Q TRAINING COMPLETE\n\n")
    print(f"Find results in {str(Path(logs_dir))}\n")
    print(f"Total Accuracy: {acc*100:.2f}%")
    print(f"Best Model Checkpoint: {checkpoint_callback.best_model_path}")

    torchscript_file = convert_model(
        experiment_name, checkpoint_callback.best_model_path
    )
    export_model(
        experiment_name,
        torchscript_file,
        "custom_handlers/iq_custom_handler.py",
        index_to_name_file,
        "models/",
    )


def visualize_dataset(
    dataset_path,
    num_iq_samples,
    logs_dir,
    class_list,
    only_use_start_of_burst,
    transform=None,
):
    print("\nVisualizing Dataset")

    dataset = SigMFDataset(
        root=dataset_path,
        sample_count=num_iq_samples,
        class_list=class_list,
        transform=transform,
        only_first_samples=only_use_start_of_burst,
    )
    dataset_class_counts = {class_name: 0 for class_name in dataset.class_list}
    for data, label in dataset:
        dataset_class_counts[dataset.class_list[label]] += 1
    print(f"Visualize Dataset: {len(dataset)=}")
    print(f"Visualize Dataset: {dataset_class_counts=}")

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=36,
        shuffle=True,
    )

    visualizer = IQVisualizer(data_loader=data_loader)

    for figure in iter(visualizer):
        figure.set_size_inches(16, 16)
        # plt.show()
        iq_viz_path = Path(logs_dir, "iq_dataset.png")
        print(f"Saving IQ visualization at {iq_viz_path}")
        plt.savefig(iq_viz_path)
        break

    spec_visualizer = SpectrogramVisualizer(
        data_loader=data_loader,
        sample_rate=20e6,
        window=sp.windows.blackmanharris(32),
        nperseg=32,
        nfft=32,
    )
    for figure in iter(spec_visualizer):
        figure.set_size_inches(16, 16)
        # plt.show()
        spec_viz_path = Path(logs_dir, "spec_dataset.png")
        print(f"Saving spectrogram visualization at {spec_viz_path}")
        plt.savefig(spec_viz_path)
        break
    print("")


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "train_dataset_path",
        type=str,
        nargs="+",
        help="Path to training dataset",
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        nargs="+",
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
        nargs="+",
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
        train_dataset_path=options.train_dataset_path,
        val_dataset_path=options.val_dataset_path,
        num_iq_samples=options.num_iq_samples,
        only_use_start_of_burst=options.only_use_start_of_burst,
        epochs=options.epochs,
        batch_size=options.batch_size,
        class_list=options.class_list,
        logs_dir=options.logs_dir,
        output_dir=options.output_dir,
    )

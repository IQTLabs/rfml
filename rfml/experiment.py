import json

from datetime import datetime
from pathlib import Path

from rfml.train_iq import *
from rfml.train_spec import *


class Experiment:
    def __init__(
        self,
        experiment_name,
        class_list,
        train_dir,
        val_dir=None,
        test_dir=None,
        iq_num_samples=1024,
        iq_only_start_of_burst=False,
        iq_epochs=40,
        iq_batch_size=128,
        iq_learning_rate=0.0001,
        iq_early_stop=10,
        iq_train_limit=1,
        iq_val_limit=1,
        spec_n_fft=1024,
        spec_time_dim=512,
        spec_epochs=40,
        spec_batch_size=32,
        spec_yolo_augment=False,
        spec_skip_export=False,
        spec_force_yolo_label_larger=False,
        notes=None,
    ):
        self.experiment_name = experiment_name
        self.train_dir = train_dir
        self.class_list = class_list
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.iq_num_samples = iq_num_samples
        self.iq_only_start_of_burst = iq_only_start_of_burst
        self.iq_epochs = iq_epochs
        self.iq_batch_size = iq_batch_size
        self.iq_learning_rate = iq_learning_rate
        self.iq_early_stop = iq_early_stop
        self.iq_train_limit = iq_train_limit
        self.iq_val_limit = iq_val_limit
        self.spec_n_fft = spec_n_fft
        self.spec_time_dim = spec_time_dim
        self.spec_n_samples = spec_n_fft * spec_time_dim
        self.spec_epochs = spec_epochs
        self.spec_batch_size = spec_batch_size
        self.spec_yolo_augment = spec_yolo_augment
        self.spec_skip_export = spec_skip_export
        self.spec_force_yolo_label_larger = spec_force_yolo_label_larger
        self.notes = notes

        Path("experiment_logs", self.experiment_name).mkdir(parents=True, exist_ok=True)
        experiment_config_path = Path(
            "experiment_logs",
            self.experiment_name,
            f"{self.experiment_name}_info_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.json",
        )
        with open(experiment_config_path, "w") as f:
            f.write(json.dumps(vars(self), indent=4))
        print(f"\nFind experiment config file at {experiment_config_path}")

    def __repr__(self):
        return str(vars(self))


def train(experiment_configs):

    for experiment_name in experiment_configs:
        print(f"\nRunning {experiment_name}")
        try:
            exp = Experiment(
                experiment_name=experiment_name, **experiment_configs[experiment_name]
            )

            logs_timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

            if exp.iq_epochs > 0:
                train_iq(
                    train_dataset_path=exp.train_dir,
                    val_dataset_path=exp.val_dir,
                    num_iq_samples=exp.iq_num_samples,
                    only_use_start_of_burst=exp.iq_only_start_of_burst,
                    epochs=exp.iq_epochs,
                    batch_size=exp.iq_batch_size,
                    class_list=exp.class_list,
                    logs_dir=Path("iq_logs", logs_timestamp),
                    output_dir=Path("experiment_logs", exp.experiment_name),
                    learning_rate=exp.iq_learning_rate,
                    experiment_name=exp.experiment_name,
                    early_stop=exp.iq_early_stop,
                    train_limit=exp.iq_train_limit,
                    val_limit=exp.iq_val_limit,
                )
            else:
                print("Skipping IQ training")

            if exp.spec_epochs > 0:
                train_spec(
                    train_dataset_path=exp.train_dir,
                    val_dataset_path=exp.val_dir,
                    n_fft=exp.spec_n_fft,
                    time_dim=exp.spec_time_dim,
                    epochs=exp.spec_epochs,
                    batch_size=exp.spec_batch_size,
                    class_list=exp.class_list,
                    yolo_augment=exp.spec_yolo_augment,
                    skip_export=exp.spec_skip_export,
                    force_yolo_label_larger=exp.spec_force_yolo_label_larger,
                    logs_dir=Path("spec_logs", logs_timestamp),
                    output_dir=Path("experiment_logs", exp.experiment_name),
                )
            else:
                print("Skipping spectrogram training")

        except Exception as error:
            print(f"Error: {error}")

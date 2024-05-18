from pathlib import Path

from experiment import * 
from train_iq import *
from train_spec import *



experiments = {
    "experiment_1": {
        "experiment_name": "experiment_1",
        "class_list": ["wifi","anom_wifi"],
        "train_dir": ["data/gamutrf/gamutrf-nz-anon-wifi", "data/gamutrf/gamutrf-nz-nonanon-wifi"],
        "iq_epochs": 40,
        "spec_epochs": 100,
        "notes": "Wi-Fi vs anomalous Wi-Fi, Ettus B200Mini, anarkiwi collect"
    },
    "experiment_2": {
        "experiment_name": "experiment_2",
        "class_list": ["mini2_video","mini2_telem"],
        "train_dir": ["dev_data/torchsig_train/samples"],
        "iq_epochs": 40,
        "spec_epochs": 100,
        "notes": "DJI Mini2, Ettus B200Mini, copy of gamutrf/gamutrf-arl/01_30_23/mini2, collected in office"
    }
}


if __name__ == "__main__":


    exp = Experiment(**experiments["experiment_1"])

    logs_timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    
    train_iq(
        train_dataset_path = exp.train_dir,
        val_dataset_path = exp.val_dir,
        num_iq_samples = exp.iq_num_samples, 
        only_use_start_of_burst = exp.iq_only_start_of_burst,
        epochs = exp.iq_epochs, 
        batch_size = exp.iq_batch_size, 
        class_list = exp.class_list, 
        output_dir = Path("experiment_logs",exp.experiment_name),
        logs_dir = Path("iq_logs", logs_timestamp),
    )

    
    train_spec(
        train_dataset_path = exp.train_dir,
        val_dataset_path = exp.val_dir,
        n_fft = exp.spec_n_fft, 
        time_dim = exp.spec_time_dim,
        epochs = exp.spec_epochs, 
        batch_size = exp.spec_batch_size, 
        class_list = exp.class_list, 
        output_dir = Path("experiment_logs",exp.experiment_name),
        logs_dir = Path("spec_logs", logs_timestamp),
    )
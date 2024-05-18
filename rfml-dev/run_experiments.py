from experiment import * 
from train_iq import *
from train_spec import *



experiments = {
    "experiment_1": {
        "experiment_name": "experiment_1",
        "class_list": ["wifi","anom_wifi"],
        "train_dir": ["data/gamutrf/gamutrf-sd-gr-ieee-wifi/test_offline"],
    }
}


if __name__ == "__main__":


    exp = Experiment(**experiments["experiment_1"])


    train_iq(
        train_dataset_path = exp.train_dir,
        val_dataset_path = exp.val_dir,
        num_iq_samples = exp.iq_num_samples, 
        only_use_start_of_burst = exp.iq_only_start_of_burst,
        epochs = exp.iq_epochs, 
        batch_size = exp.iq_batch_size, 
        class_list = exp.class_list, 
        output_dir = exp.experiment_name,
    )

    
    # train_spec(
    #     train_dataset_path = exp.train_dir,
    #     val_dataset_path = exp.val_dir,
    #     n_fft = exp.spec_n_fft, 
    #     time_dim = exp.spec_time_dim,
    #     epochs = exp.spec_epochs, 
    #     batch_size = exp.spec_batch_size, 
    #     class_list = exp.class_list, 
    #     output_dir = exp.experiment_name,
    # )
import json

from datetime import datetime
from pathlib import Path

class Experiment():
    def __init__(
        self, 
        experiment_name,
        class_list,
        train_dir, 
        val_dir = None, 
        test_dir = None, 
        iq_num_samples = 1024, 
        iq_only_start_of_burst = True,
        iq_epochs = 40,
        iq_batch_size = 180,
        spec_n_fft = 1024,
        spec_time_dim = 512,
        spec_epochs = 40,
        spec_batch_size = 32,
        notes = None,
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
        self.spec_n_fft = spec_n_fft
        self.spec_time_dim = spec_time_dim
        self.spec_n_samples = spec_n_fft * spec_time_dim
        self.spec_epochs = spec_epochs
        self.spec_batch_size = spec_batch_size
        self.notes = notes

        Path(self.experiment_name).mkdir(parents=True, exist_ok=True)
        with open(Path(self.experiment_name,f"{self.experiment_name}_info_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.json"), "w") as f:
            f.write(json.dumps(vars(self), indent=4))

    def __repr__(self):
        return str(vars(self))
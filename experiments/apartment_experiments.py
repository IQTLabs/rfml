import torch

torch.set_float32_matmul_precision("medium")
from rfml.experiment import *

#
# python rfml/apartment_experiments.py
# cp models/apartment_experiment.mar ~/iqt/gamutrf-deploy/docker_rundir/model_store/
# sudo chmod -R 777 /home/ltindall/iqt/gamutrf-deploy/docker_rundir/
#


experiments = {
    "apartment_experiment": {
        "class_list": ["mini2_video", "mini2_telem", "environment"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-lucas-collect/train/environment/",
            "/data/s3_gamutrf/gamutrf-lucas-collect/train/mini2/",
        ],
        "iq_epochs": 10,
        "iq_train_limit": 0.5,
        "iq_only_start_of_burst": False,
        "iq_num_samples": 1024,
        "spec_epochs": 0,
    }
}


if __name__ == "__main__":

    train(experiments)

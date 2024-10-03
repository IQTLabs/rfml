import torch

torch.set_float32_matmul_precision("medium")
from rfml.experiment import *
import sys

root = sys.argv[1]

#
# python rfml/siggen_experiments.py
# python convert_model.py --model_name siggen_model --checkpoint /home/ltindall/iqt/rfml/lightning_logs/siggen_experiment/checkpoints/experiment_logs/siggen_experiment/iq_checkpoints/checkpoint-v3.ckpt
# torch-model-archiver --force --model-name siggen_model --version 1.0 --serialized-file rfml/weights/siggen_model_torchscript.pt --handler custom_handlers/iq_custom_handler.py --export-path models/ -r custom_handlers/requirements.txt
# cp models/siggen_model.mar ~/iqt/gamutrf-deploy/docker_rundir/model_store/
# sudo chmod -R 777 /home/ltindall/iqt/gamutrf-deploy/docker_rundir/
#


experiments = {
    "siggen_experiment": {
        "class_list": ["am", "fm"],
        "train_dir": [
            f"{root}/fm.sigmf-meta",
            f"{root}/am.sigmf-meta",
        ],
        "val_dir": [
            f"{root}/fm.sigmf-meta",
            f"{root}/am.sigmf-meta",
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

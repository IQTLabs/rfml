from pathlib import Path

from experiment import * 
from train_iq import *
from train_spec import *


# Ensure that data directories have sigmf-meta files with annotations 
# Annotations can be generated using Label_WiFi.ipynb and Label_DJI.ipynb 
experiments = {
    "experiment_0": {
        "experiment_name": "experiment_0",
        "class_list": ["wifi","anom_wifi"],
        "train_dir": ["data/gamutrf/gamutrf-sd-gr-ieee-wifi/test_offline"],
        "iq_epochs": 10,
        "spec_epochs": 10,
        "notes": "TESTING"
    },
    "experiment_1": {
        "experiment_name": "experiment_1",
        "class_list": ["wifi","anom_wifi"],
        "train_dir": ["data/gamutrf/gamutrf-nz-anon-wifi", "data/gamutrf/gamutrf-nz-nonanon-wifi"],
        "iq_epochs": 40,
        "spec_epochs": 40,
        "notes": "Wi-Fi vs anomalous Wi-Fi, Ettus B200Mini, anarkiwi collect"
    },
    "experiment_2": {
        "experiment_name": "experiment_2",
        "class_list": ["mini2_video","mini2_telem"],
        "train_dir": ["dev_data/torchsig_train/samples"],
        "iq_epochs": 25,
        "spec_epochs": 50,
        "notes": "DJI Mini2, Ettus B200Mini RX, copy of lab collection gamutrf/gamutrf-arl/01_30_23/mini2"
    },
    "experiment_3": {
        "experiment_name": "experiment_3",
        "class_list": ["wifi","anom_wifi"],
        "train_dir": ["data/gamutrf/gamutrf-nz-anon-wifi", "data/gamutrf/gamutrf-nz-nonanon-wifi"],
        "val_dir": ["data/gamutrf/gamutrf-wifi-and-anom-bladerf"],
        "iq_epochs": 40,
        "spec_epochs": 70,
        "spec_skip_export": True, # USE WITH CAUTION (but speeds up large directories significantly): skip after first run if using separate train/val directories 
        "notes": "Wi-Fi vs anomalous Wi-Fi, train on Ettus B200Mini RX/TX, validate on BladeRF TX & Ettus B200Mini RX, anarkiwi collect"
    },
    "experiment_4": {
        "experiment_name": "experiment_4",
        "class_list": ["wifi","anom_wifi"],
        "train_dir": ["data/gamutrf/gamutrf-nz-anon-wifi", "data/gamutrf/gamutrf-nz-nonanon-wifi"],
        "val_dir": ["data/gamutrf/gamutrf-nz-wifi"],
        "iq_epochs": 40,
        "spec_epochs": 100,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train on Ettus B200Mini RX/TX, validate on real Wi-Fi AP TX & Ettus B200Mini RX, anarkiwi collect"
    },
    "experiment_5": {
        "experiment_name": "experiment_5",
        "class_list": ["mini2_video","mini2_telem"],
        "train_dir": ["dev_data/torchsig_train/samples"],
        "val_dir": ["data/gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/test_iq_label"],
        "iq_epochs": 40,
        "spec_epochs": 100,
        "spec_force_yolo_label_larger": True,
        "notes": "DJI Mini2, Ettus B200Mini RX, train on copy of lab collection gamutrf/gamutrf-arl/01_30_23/mini2, validate on field collect gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/test_iq_label"
    },
    "experiment_6": {
        "experiment_name": "experiment_6",
        "class_list": ["wifi","anom_wifi"],
        "train_dir": [
            "data/gamutrf/wifi-data-03082024/20msps/normal/train",
            "data/gamutrf/wifi-data-03082024/20msps/normal/test",
            "data/gamutrf/wifi-data-03082024/20msps/normal/inference",
            "data/gamutrf/wifi-data-03082024/20msps/mod/train",
            "data/gamutrf/wifi-data-03082024/20msps/mod/test",
            "data/gamutrf/wifi-data-03082024/20msps/mod/inference",
        ],
        "val_dir": [
            "data/gamutrf/wifi-data-03082024/20msps/normal/validate",
            "data/gamutrf/wifi-data-03082024/20msps/mod/validate",
        ], 
        "iq_num_samples": 16*25, 
        "iq_epochs": 10,
        "iq_batch_size": 16,
        "spec_batch_size": 32,
        "spec_epochs": 40,
        "spec_n_fft": 16,
        "spec_time_dim": 25,
        "notes": "Ettus B200Mini RX, emair collect"
    },
    "experiment_7": {
        "experiment_name": "experiment_7",
        "class_list": ["wifi","anom_wifi"],
        "train_dir": ["data/gamutrf/gamutrf-wifi-and-anom-bladerf"],
        "iq_epochs": 40,
        "spec_epochs": 40,
        "notes": "Wi-Fi vs anomalous Wi-Fi, BladeRF, anarkiwi collect"
    },
    "experiment_8": {
        "experiment_name": "experiment_8",
        "class_list": ["wifi","anom_wifi"],
        "train_dir": ["data/gamutrf/gamutrf-wifi-and-anom-bladerf"],
        "val_dir": ["data/gamutrf/gamutrf-nz-anon-wifi", "data/gamutrf/gamutrf-nz-nonanon-wifi"],
        "iq_epochs": 40,
        "spec_epochs": 70,
        "spec_skip_export": True, # USE WITH CAUTION (but speeds up large directories significantly): skip after first run if using separate train/val directories 
        "notes": "Wi-Fi vs anomalous Wi-Fi, validate on BladeRF TX & Ettus B200Mini RX, train on Ettus B200Mini RX/TX, anarkiwi collect"
    },
    "experiment_9": {
        "experiment_name": "experiment_9",
        "class_list": ["mini2_video","mini2_telem"],
        "train_dir": ["dev_data/torchsig_train/samples"],
        "val_dir": ["data/gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/test_iq_label"],
        "iq_epochs": 40,
        "spec_epochs": 200,
        "spec_yolo_augment": True,
        "notes": "DJI Mini2, Ettus B200Mini RX, train on copy of lab collection gamutrf/gamutrf-arl/01_30_23/mini2, validate on field collect gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/test_iq_label"
    },
    "experiment_10": {
        "experiment_name": "experiment_10",
        "class_list": ["wifi","anom_wifi"],
        "train_dir": ["data/gamutrf/anom_wifi/train_gamutrf-nz-anon-wifi", "data/gamutrf/anom_wifi/train_gamutrf-nz-nonanon-wifi", "data/gamutrf/anom_wifi/train_gamutrf-wifi-and-anom-bladerf"],
        "val_dir": ["data/gamutrf/anom_wifi/val_gamutrf-nz-anon-wifi", "data/gamutrf/anom_wifi/val_gamutrf-nz-nonanon-wifi"],
        "iq_epochs": 40,
        "spec_epochs": 70,
        "spec_skip_export": True, # USE WITH CAUTION (but speeds up large directories significantly): skip after first run if using separate train/val directories 
        "notes": "Wi-Fi vs anomalous Wi-Fi, train on both BladeRF TX & Ettus B200Mini RX and on Ettus B200Mini RX/TX, validate on Ettus B200Mini RX/TX, anarkiwi collect"
    },
    "experiment_11": {
        "experiment_name": "experiment_11",
        "class_list": ["wifi","anom_wifi"],
        "train_dir": ["data/gamutrf/anom_wifi/train_gamutrf-nz-anon-wifi", "data/gamutrf/anom_wifi/train_gamutrf-nz-nonanon-wifi", "data/gamutrf/anom_wifi/train_gamutrf-wifi-and-anom-bladerf"],
        "val_dir": ["data/gamutrf/anom_wifi/val_gamutrf-wifi-and-anom-bladerf"],
        "iq_epochs": 40,
        "spec_epochs": 70,
        "spec_skip_export": True, # USE WITH CAUTION (but speeds up large directories significantly): skip after first run if using separate train/val directories 
        "notes": "Wi-Fi vs anomalous Wi-Fi, train on both BladeRF TX & Ettus B200Mini RX and on Ettus B200Mini RX/TX, validate on BladeRF TX, anarkiwi collect"
    },
}


if __name__ == "__main__":

    for i in [9]: #range(7):
        print(f"Experiment {i}")    
        try:
            exp = Experiment(**experiments[f"experiment_{i}"])

            logs_timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            
            # train_iq(
            #     train_dataset_path = exp.train_dir,
            #     val_dataset_path = exp.val_dir,
            #     num_iq_samples = exp.iq_num_samples, 
            #     only_use_start_of_burst = exp.iq_only_start_of_burst,
            #     epochs = exp.iq_epochs, 
            #     batch_size = exp.iq_batch_size, 
            #     class_list = exp.class_list, 
            #     output_dir = Path("experiment_logs",exp.experiment_name),
            #     logs_dir = Path("iq_logs", logs_timestamp),
            # )

            
            train_spec(
                train_dataset_path = exp.train_dir,
                val_dataset_path = exp.val_dir,
                n_fft = exp.spec_n_fft, 
                time_dim = exp.spec_time_dim,
                epochs = exp.spec_epochs, 
                batch_size = exp.spec_batch_size, 
                class_list = exp.class_list, 
                yolo_augment = exp.spec_yolo_augment,
                skip_export = exp.spec_skip_export,
                force_yolo_label_larger = exp.spec_force_yolo_label_larger,
                output_dir = Path("experiment_logs",exp.experiment_name),
                logs_dir = Path("spec_logs", logs_timestamp),
            )
        except Exception as error: 
            print(f"Error: {error}")

import torch

torch.set_float32_matmul_precision("high")


from rfml.experiment import *

# Ensure that data directories have sigmf-meta files with annotations
# Annotations can be generated using scripts in label_scripts directory or notebooks/Label_WiFi.ipynb and notebooks/Label_DJI.ipynb

experiments = {
    "experiment_test": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": ["data/gamutrf/gamutrf-sd-gr-ieee-wifi/test_offline"],
        "iq_epochs": 10,
        "spec_epochs": 10,
        "notes": "TESTING",
    },
    "experiment_nz_wifi_ettus": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "data/gamutrf/gamutrf-nz-anon-wifi",
            "data/gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "iq_epochs": 40,
        "spec_epochs": 40,
        "notes": "Wi-Fi vs anomalous Wi-Fi, Ettus B200Mini, anarkiwi collect",
    },
    "experiment_mini2_lab": {
        "class_list": ["mini2_video", "mini2_telem"],
        "train_dir": ["dev_data/torchsig_train/samples"],
        "iq_epochs": 25,
        "spec_epochs": 50,
        "notes": "DJI Mini2, Ettus B200Mini RX, copy of lab collection gamutrf/gamutrf-arl/01_30_23/mini2",
    },
    "experiment_nz_wifi_ettus_blade": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "data/gamutrf/gamutrf-nz-anon-wifi",
            "data/gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "val_dir": ["data/gamutrf/gamutrf-wifi-and-anom-bladerf"],
        "iq_epochs": 40,
        "spec_epochs": 70,
        "spec_skip_export": True,  # USE WITH CAUTION (but speeds up large directories significantly): skip after first run if using separate train/val directories
        "notes": "Wi-Fi vs anomalous Wi-Fi, train on Ettus B200Mini RX/TX, validate on BladeRF TX & Ettus B200Mini RX, anarkiwi collect",
    },
    "experiment_nz_wifi_ettus_ap": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "data/gamutrf/gamutrf-nz-anon-wifi",
            "data/gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "val_dir": ["data/gamutrf/gamutrf-nz-wifi"],
        "iq_epochs": 40,
        "spec_epochs": 100,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train on Ettus B200Mini RX/TX, validate on real Wi-Fi AP TX & Ettus B200Mini RX, anarkiwi collect",
    },
    "experiment_mini2_lab_field_large_label": {
        "class_list": ["mini2_video", "mini2_telem"],
        "train_dir": ["dev_data/torchsig_train/samples"],
        "val_dir": [
            "data/gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/test_iq_label"
        ],
        "iq_epochs": 40,
        "spec_epochs": 100,
        "spec_force_yolo_label_larger": True,
        "notes": "DJI Mini2, Ettus B200Mini RX, train on copy of lab collection gamutrf/gamutrf-arl/01_30_23/mini2, validate on field collect gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/test_iq_label",
    },
    "experiment_emair": {
        "class_list": ["wifi", "anom_wifi"],
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
        "iq_num_samples": 16 * 25,
        "iq_epochs": 10,
        "iq_batch_size": 16,
        "spec_batch_size": 32,
        "spec_epochs": 40,
        "spec_n_fft": 16,
        "spec_time_dim": 25,
        "notes": "Ettus B200Mini RX, emair collect",
    },
    "experiment_nz_wifi_blade": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": ["data/gamutrf/gamutrf-wifi-and-anom-bladerf"],
        "iq_epochs": 40,
        "spec_epochs": 40,
        "notes": "Wi-Fi vs anomalous Wi-Fi, BladeRF, anarkiwi collect",
    },
    "experiment_nz_wifi_blade_ettus": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": ["data/gamutrf/gamutrf-wifi-and-anom-bladerf"],
        "val_dir": [
            "data/gamutrf/gamutrf-nz-anon-wifi",
            "data/gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "iq_epochs": 40,
        "spec_epochs": 70,
        "spec_skip_export": True,  # USE WITH CAUTION (but speeds up large directories significantly): skip after first run if using separate train/val directories
        "notes": "Wi-Fi vs anomalous Wi-Fi, validate on BladeRF TX & Ettus B200Mini RX, train on Ettus B200Mini RX/TX, anarkiwi collect",
    },
    "experiment_mini2_lab_field": {
        "class_list": ["mini2_video", "mini2_telem"],
        "train_dir": ["dev_data/torchsig_train/samples"],
        "val_dir": [
            "data/gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/test_iq_label"
        ],
        "iq_epochs": 40,
        "spec_epochs": 100,
        "notes": "DJI Mini2, Ettus B200Mini RX, train on copy of lab collection gamutrf/gamutrf-arl/01_30_23/mini2, validate on field collect gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/test_iq_label",
    },
    "experiment_train_blade_2": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/ettus/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/ettus/",
        ],
        "iq_epochs": 40,
        "spec_epochs": 70,
        "spec_batch_size": -1,
        "spec_skip_export": True,  # USE WITH CAUTION (but speeds up large directories significantly): skip after first run if using separate train/val directories
        "notes": "Wi-Fi vs anomalous Wi-Fi, train on BladeRF TX & Ettus B200Mini RX, validate on Ettus B200Mini RX/TX, anarkiwi collect 2",
    },
    "experiment_train_ettus_2": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/ettus/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/ettus/",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/",
        ],
        "iq_epochs": 40,
        "spec_epochs": 70,
        "spec_batch_size": -1,
        "spec_skip_export": True,  # USE WITH CAUTION (but speeds up large directories significantly): skip after first run if using separate train/val directories
        "notes": "Wi-Fi vs anomalous Wi-Fi, validate on BladeRF TX & Ettus B200Mini RX, train on Ettus B200Mini RX/TX, anarkiwi collect 2",
    },
    "experiment_train_blade_1": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": ["/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf"],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-nz-anon-wifi",
            "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "iq_epochs": 40,
        "spec_epochs": 70,
        "spec_batch_size": -1,
        "spec_skip_export": True,  # USE WITH CAUTION (but speeds up large directories significantly): skip after first run if using separate train/val directories
        "notes": "Wi-Fi vs anomalous Wi-Fi, train on BladeRF TX & Ettus B200Mini RX, validate on Ettus B200Mini RX/TX, anarkiwi collect 1",
    },
    "experiment_train_ettus_1": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-nz-anon-wifi",
            "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "val_dir": ["/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf"],
        "iq_epochs": 40,
        "spec_epochs": 70,
        "spec_batch_size": -1,
        "spec_skip_export": True,  # USE WITH CAUTION (but speeds up large directories significantly): skip after first run if using separate train/val directories
        "notes": "Wi-Fi vs anomalous Wi-Fi, validate on BladeRF TX & Ettus B200Mini RX, train on Ettus B200Mini RX/TX, anarkiwi collect 1",
    },
    "experiment_ettus_1": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-nz-anon-wifi",
            "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "iq_epochs": 40,
        "spec_epochs": 70,
        "spec_batch_size": -1,
        "spec_skip_export": True,  # USE WITH CAUTION (but speeds up large directories significantly): skip after first run if using separate train/val directories
        "notes": "Wi-Fi vs anomalous Wi-Fi, train and validate on Ettus B200Mini RX/TX, anarkiwi collect 1",
    },
    "experiment_blade_1": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": ["/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf"],
        "iq_epochs": 40,
        "spec_epochs": 70,
        "spec_batch_size": -1,
        "spec_skip_export": True,  # USE WITH CAUTION (but speeds up large directories significantly): skip after first run if using separate train/val directories
        "notes": "Wi-Fi vs anomalous Wi-Fi, train and validate on BladeRF TX & Ettus B200Mini RX, anarkiwi collect 1",
    },
    "experiment_blade_2": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/",
        ],
        "iq_epochs": 40,
        "spec_epochs": 70,
        "spec_batch_size": -1,
        "spec_skip_export": True,  # USE WITH CAUTION (but speeds up large directories significantly): skip after first run if using separate train/val directories
        "notes": "Wi-Fi vs anomalous Wi-Fi, train and validate on BladeRF TX & Ettus B200Mini RX, anarkiwi collect 2",
    },
    "experiment_ettus_2": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/ettus/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/ettus/",
        ],
        "iq_epochs": 40,
        "spec_epochs": 70,
        "spec_batch_size": -1,
        "spec_skip_export": True,  # USE WITH CAUTION (but speeds up large directories significantly): skip after first run if using separate train/val directories
        "notes": "Wi-Fi vs anomalous Wi-Fi, train and validate on Ettus B200Mini RX/TX, anarkiwi collect 2",
    },
    "experiment_mavic3": {
        "class_list": ["mavic3_video", "environment"],
        "train_dir": [
            "/home/iqt/lberndt/gamutrf-depoly/data/samples/mavic-30db",
            "/home/iqt/lberndt/gamutrf-depoly/data/samples/mavic-0db",
            "/home/iqt/lberndt/gamutrf-depoly/data/samples/mavic-30gain",
            "/home/iqt/lberndt/gamutrf-depoly/data/samples/mavic-40gain",
            "/home/iqt/lberndt/gamutrf-depoly/data/samples/environment",
        ],
        "iq_epochs": 10,
        "spec_epochs": 0,
        "notes": "DJI Mavic3 Detection",
    },
    "experiment_siggen": {
        "class_list": ["am", "fm"],
        "train_dir": ["/scratch/tmp/rfmltest"],
        "iq_epochs": 10,
        "spec_epochs": 0,
        "notes": "narrowband test",
        "iq_only_start_of_burst": False,
    },
}


if __name__ == "__main__":

    experiments_to_run = [
        # "experiment_test",
        # "experiment_nz_wifi_ettus",
        # "experiment_mini2_lab",
        # "experiment_nz_wifi_ettus_blade",
        # "experiment_nz_wifi_ettus_ap",
        # "experiment_mini2_lab_field_large_label",
        # "experiment_emair",
        # "experiment_nz_wifi_blade",
        # "experiment_nz_wifi_blade_ettus",
        # "experiment_mini2_lab_field",
        # "experiment_train_blade_1",
        # "experiment_train_ettus_1",
        # "experiment_train_blade_2",
        # "experiment_train_ettus_2",
        # "experiment_ettus_1",
        # "experiment_blade_1",
        # "experiment_ettus_2",
        # "experiment_blade_2",
        # "experiment_mavic3",
        "experiment_siggen",
    ]

    train({name: experiments[name] for name in experiments_to_run})

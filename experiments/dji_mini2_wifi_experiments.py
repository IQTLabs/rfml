from rfml.experiment import *

# Ensure that data directories have sigmf-meta files with annotations
# Annotations can be generated using scripts in label_scripts directory or notebooks/Label_WiFi.ipynb and notebooks/Label_DJI.ipynb

spec_epochs = 0
iq_epochs = 10
iq_only_start_of_burst = False
iq_num_samples = 4000
iq_early_stop = 3
iq_train_limit = 0.01
iq_val_limit = 0.1

experiments = {
    "experiment_nz_wifi_arl_mini2_pdx_mini2_to_leesburg_mini2": {
        "class_list": ["mini2_video", "mini2_telem", "wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-nz-wifi",
            "/data/s3_gamutrf/gamutrf-arl/01_30_23/mini2",
            "/data/s3_gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/iq_recordings",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-nz-wifi",
            "/data/s3_gamutrf/gamutrf-birdseye-field-days/leesburg_field_day_2022_06_15/iq_recordings",
        ],
        "iq_epochs": iq_epochs,
        "spec_epochs": spec_epochs,
        "iq_only_start_of_burst": iq_only_start_of_burst,
        "iq_early_stop": iq_early_stop,
        "iq_train_limit": iq_train_limit,
        "iq_val_limit": iq_val_limit,
        "notes": "",
    },
    "experiment_nz_wifi_arl_mini2_to_leesburg_mini2": {
        "class_list": ["mini2_video", "mini2_telem", "wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-nz-wifi",
            "/data/s3_gamutrf/gamutrf-arl/01_30_23/mini2",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-nz-wifi",
            "/data/s3_gamutrf/gamutrf-birdseye-field-days/leesburg_field_day_2022_06_15/iq_recordings",
        ],
        "iq_epochs": iq_epochs,
        "spec_epochs": spec_epochs,
        "iq_only_start_of_burst": iq_only_start_of_burst,
        "iq_early_stop": iq_early_stop,
        "iq_train_limit": iq_train_limit,
        "iq_val_limit": iq_val_limit,
        "notes": "",
    },
    "experiment_nz_wifi_pdx_mini2_to_leesburg_mini2": {
        "class_list": ["mini2_video", "mini2_telem", "wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-nz-wifi",
            "/data/s3_gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/iq_recordings",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-nz-wifi",
            "/data/s3_gamutrf/gamutrf-birdseye-field-days/leesburg_field_day_2022_06_15/iq_recordings",
        ],
        "iq_epochs": iq_epochs,
        "spec_epochs": spec_epochs,
        "iq_only_start_of_burst": iq_only_start_of_burst,
        "iq_early_stop": iq_early_stop,
        "iq_train_limit": iq_train_limit,
        "iq_val_limit": iq_val_limit,
        "notes": "",
    },
    "experiment_nz_wifi_arl_mini2_to_pdx_mini2": {
        "class_list": ["mini2_video", "mini2_telem", "wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-nz-wifi",
            "/data/s3_gamutrf/gamutrf-arl/01_30_23/mini2",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-nz-wifi",
            "/data/s3_gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/iq_recordings",
        ],
        "iq_epochs": iq_epochs,
        "spec_epochs": spec_epochs,
        "iq_only_start_of_burst": iq_only_start_of_burst,
        "iq_early_stop": iq_early_stop,
        "iq_train_limit": iq_train_limit,
        "iq_val_limit": iq_val_limit,
        "notes": "",
    },
    "experiment_nz_wifi_leesburg_mini2_to_pdx_mini2": {
        "class_list": ["mini2_video", "mini2_telem", "wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-nz-wifi",
            "/data/s3_gamutrf/gamutrf-birdseye-field-days/leesburg_field_day_2022_06_15/iq_recordings",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-nz-wifi",
            "/data/s3_gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/iq_recordings",
        ],
        "iq_epochs": iq_epochs,
        "spec_epochs": spec_epochs,
        "iq_only_start_of_burst": iq_only_start_of_burst,
        "iq_early_stop": iq_early_stop,
        "iq_train_limit": iq_train_limit,
        "iq_val_limit": iq_val_limit,
        "notes": "",
    },
    "experiment_nz_wifi_leesburg_mini2_pdx_mini2_to_arl_mini2": {
        "class_list": ["mini2_video", "mini2_telem", "wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-nz-wifi",
            "/data/s3_gamutrf/gamutrf-birdseye-field-days/leesburg_field_day_2022_06_15/iq_recordings",
            "/data/s3_gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/iq_recordings",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-nz-wifi",
            "/data/s3_gamutrf/gamutrf-arl/01_30_23/mini2",
        ],
        "iq_epochs": iq_epochs,
        "spec_epochs": spec_epochs,
        "iq_only_start_of_burst": iq_only_start_of_burst,
        "iq_early_stop": iq_early_stop,
        "iq_train_limit": iq_train_limit,
        "iq_val_limit": iq_val_limit,
        "notes": "",
    },
}


if __name__ == "__main__":

    experiments_to_run = [
        # "experiment_nz_wifi_arl_mini2_pdx_mini2_to_leesburg_mini2",
        # "experiment_nz_wifi_arl_mini2_to_leesburg_mini2",
        # "experiment_nz_wifi_pdx_mini2_to_leesburg_mini2",
        # "experiment_nz_wifi_arl_mini2_to_pdx_mini2",
        # "experiment_nz_wifi_leesburg_mini2_to_pdx_mini2",
        "experiment_nz_wifi_leesburg_mini2_pdx_mini2_to_arl_mini2"
    ]

    train({name: experiments[name] for name in experiments_to_run})

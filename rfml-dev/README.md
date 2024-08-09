# RFML

This repo provides the pipeline for working with RF datasets, labeling them and training both IQ and spectragram based models. The SigMF standard is used for managing RF data and the labels/anotations on the data. It also make use ot the Torchsig framework for performing RF related augmentation of the data to help make the trained models more robust and functional in the real world.
 
## Preqs

### Poetry

Follow the instructions here to install Poetry: https://python-poetry.org/docs/#installation

### Torchsig

Download from [Github](https://github.com/TorchDSP/torchsig) and then use Poetry to install it.

```
git clone https://github.com/TorchDSP/torchsig.git
cd torchsig
git checkout 8049b43
```

Make you have the Poetry environment activated, using `poetry shell`, then run:

```
poetry add ./torchsig
```

(update it with the correct path the directoy where Torchsig is)

### Torch Model Archiver

Install the Torch Model Archiver:
```
sudo pip install torch-model-archiver
```

More information about this tool is available here: 
https://github.com/pytorch/serve/blob/master/model-archiver/README.md

### Inspectrum (optional)

This utility is useful for inspecting sigmf files and the annotations that the auto label scripts make.
https://github.com/miek/inspectrum



# Building a Model


## Approach

Our current approach is to capture samples of the background RF environment and then also isolate signals of interest and capture samples of each of the signals. The same label will be applied to all of the signals present in the background environment samples. We use this to essentially teach the model to ignore those signals. For this to work, it is important that none of the signals of interest are present. Since it is really tough these days to find an RF free environment, we have build a mini-faraday cage enclosure by lining the inside of a pelican case with foil. There are lots of instructions, like [this one](https://mosequipment.com/blogs/blog/build-your-own-faraday-cage), available online if you want to build your own. With this, the signal will be very strong, so make sure you adjust the SDR's gain appropriately.

## Labeling IQ Data

The scripts in the [label_scripts](./label_scripts/) use signal processing to automatically label IQ data. The scripts looks at the signal power to detect when there is a signal present in the IQ data. When a signal is detected, the script will look at the frequencies for that set of samples and find the upper and lower bounds.


### Tunning Autolabeling

In the Labeling Scripts, the settings for autolabeling need to be tuned for the type of signals that were collected.

```python
annotation_utils.annotate(
                f, 
                label="mavic3_video",               # This is the label that is applied to all of the matching annotations
                avg_window_len=256,                 # The number of samples over which to average signal power
                avg_duration=0.25,                  # The number of seconds, from the start of the recording to use to automatically calculate the SNR threshold, if it is None then all of the samples will be used
                debug=False,    
                estimate_frequency=True,            # Whether the frequency bounds for an annotation should be calculated. estimate_frequency needs to be enabled if you use min/max_bandwidth
                spectral_energy_threshold=0.95,     # Percentage used to determine the upper and lower frequency bounds for an annotation
                force_threshold_db=-58,             # Used to manually set the threshold used for detecting a signal and creating an annotation. If None, then the automatic threshold calcuation will be used instead.
                overwrite=False,                    # If True, any existing annotations in the .sigmf-meta file will be removed
                min_bandwidth=16e6,                 # The minimum bandwidth (in Hz) of a signal to annotate
                max_bandwidth=None,                 # The maximum bandwidth (in Hz) of a signal to annotate
                min_annotation_length=10000,        # The minimum numbers of samples in length a signal needs to be in order for it to be annotated. This is directly related to the sample rate a signal was captured at and does not take into account bandwidth. So 10000 samples at 20,000,000 samples per second, would mean a minimum transmission length of 0.0005 seconds
                # max_annotations=500,              # The maximum number of annotations to automatically add  
                dc_block=True                       # De-emphasize the DC spike when trying to calculate the frequencies for a signal
            )
```

### Tips for Tuning Autolabeling

#### Force Threshold dB
![low threshold](./images/low_threshold.png)

If you see annotations where harmonics or lower power, unintentional signals are getting selected, try setting the `force_threshold_db`. The automatic threshold calculation maybe selecting a value that is too low. Find a value for `force_threshold_db` where it is selecting the intended signals and ignoring the low power ones.

#### Spectral Energy Threshold
![spectral energy](./images/spectral_energy.png)

If the frequency bounds are not lining up with the top or bottom part of a signal, make the `spectral_energy_threshold` higher. Sometime a setting as high as 0.99 is required 

#### Skipping "small" Signals
![small signals](./images/min_annotation.png)

Some tuning is needed for signals that have a short transmission duration and/or limited bandwidth. Here are a couple things to try if they are getting skipped:
- `min_annotation_length` is the minimum number of samples for an annotation. If the signal is has less samples than this, it will not be annotated. Try lowering this.
- The `average_duration` setting maybe too long and the signal is getting averaged into the noise. Try lowering this.
- `min_bandwidth` is the minimum bandwidth (in Hz) for a signal to be detected. If this value is too high, signals that have less bandiwdth will be ignored. Try lowering this.

## Training a Model

After you have finished labeling your data, the next step is to train a model on it. This repo makes it easy to train both IQ and Spectragram based models from sigmf data. 

### Configure

This repo provides an automated script for training and evaluating models. To do this, configure the [run_experiments.py](./run_experiments.py) file to point to the data you want to use and set the training parameters:

```python
    "experiment_0": {                                   # This is the Key to use, it needs to be `experiment_` followed by an increasing number
        "experiment_name": "experiment_1",              # A name to refer to the experiment
        "class_list": ["mavic3_video","mavic3_remoteid","environment"],     # The labels that are present in the sigmf-meta files
        "train_dir": ["/home/iqt/lberndt/gamutrf-depoly/data/samples/mavic-30db", "/home/iqt/lberndt/gamutrf-depoly/data/samples/mavic-0db", "/home/iqt/lberndt/gamutrf-depoly/data/samples/environment"],  # The sigmf files to use, including the path to the file
        "iq_epochs": 10,                # Number of epochs for IQ training, if it is 0 or None, it will be skipped
        "spec_epochs": 10,              # Number of epochs for spctragram training, if it is 0 or None, it will be skipped
        "notes": "DJI Mavic3 Detection" # Notes to your future self
    }
```

Once you have the **run_experiments.py** file configured, run it:

```bash
python3 run_experiments.py
```

Once the training has completed, it will print out the logs location, model accuracy, and the location of the best checkpoint: 

```bash
I/Q TRAINING COMPLETE


Find results in experiment_logs/experiment_1/iq_logs/08_08_2024_09_17_32

Total Accuracy: 98.10%
Best Model Checkpoint: /home/iqt/lberndt/rfml-dev-1/rfml-dev/lightning_logs/version_5/checkpoints/experiment_logs/experiment_1/iq_checkpoints/checkpoint.ckpt
```

### Convert Model

Once you have a trained model, you need to convert it into a portable format that can easily be served by TorchServe. To do this, use **convert_model.py**:

```bash
python3 convert_model.py --model_name=drone_detect --checkpoint=/home/iqt/lberndt/rfml-dev-1/rfml-dev/lightning_logs/version_5/checkpoints/experiment_logs/experiment_1/iq_checkpoints/checkpoint.ckpt
```
This will export a **_torchscript.pt** file.

```bash
torch-model-archiver --force --model-name drone_detect --version 1.0 --serialized-file weights/drone_detect_torchscript.pt --handler custom_handlers/iq_custom_handler.py  --export-path models/ -r custom_handler/requirements.txt
```

## Files

[annotation_utils.py](annotation_utils.py) - DSP based automated labelling tools

[auto_label.py](auto_label.py) - CV based automated labelling tools

[data.py](data.py) - RF data operations tool

[experiment.py](experiment.py) - Class to manage experiments 

[models.py](models.py) - Class for I/Q models (based on TorchSig) 

[run_experiments.py](run_experiments.py) - Experiment configurations and run script

[sigmf_pytorch_dataset.py](sigmf_pytorch_dataset.py) - PyTorch style dataset class for SigMF data (based on TorchSig) 

[spectrogram.py](spectrogram.py) - Spectrogram tools 

[test_data.py](test_data.py) - Test for data.py (might be outdated)

[train_iq.py](train_iq.py) - Training script for I/Q models

[train_spec.py](train_spec.py) - Training script for spectrogram models

[zst_parse.py](zst_parse.py) - ZST file parsing tool, for GamutRF-style filenames  


## Should be removed 

[wifi_label_utils.py](wifi_label_utils.py) - Old version of annotation_utils.py

[sigmf_db_dataset.py](sigmf_db_dataset.py) - TorchSig class for pickle based dataset (NOT USED) 



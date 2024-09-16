# RFML

This repo provides the pipeline for working with RF datasets, labeling them and training both IQ and spectrogram based models. The SigMF standard is used for managing RF data and the labels/annotations on the data. It also uses the Torchsig framework for performing RF related augmentation of the data to help make the trained models more robust and functional in the real world.
 
[Prerequisites](#prerequisites)

[Virtual environment](#activate-virtual-environment)

[Install](#install)

[Verify install with GPU support](#verify-install-with-gpu-support-optional)

[Building a model](#building-a-model)

[Labelling I/Q data](#labeling-iq-data)

[Training a model](#training-a-model)

[Files](#files)

## Prerequisites

### Poetry

Follow the instructions here to install Poetry: https://python-poetry.org/docs/#installation


### Inspectrum (optional)

https://github.com/miek/inspectrum

This utility is useful for inspecting sigmf files and the annotations that the auto label scripts make.



### Anylabelling (optional)
https://github.com/vietanhdev/anylabeling

This program is used for image annotation and offers AI-assisted labelling. 


## Activate virtual environment 

This project uses Poetry for dependency management and packaging. Poetry can be used with external virtual environments. 
If using a non-Poetry virtual environment, start by activating the environment before running Poetry commands. See note in [Poetry docs](https://python-poetry.org/docs/basic-usage/#using-your-virtual-environment) for more info. 


### Using Poetry

To activate the Poetry virtual environment with all of the Python modules configured, run the following:

```bash
poetry shell
```
See [Poetry docs](https://python-poetry.org/docs/basic-usage/#activating-the-virtual-environment) for more information. 

## Install

```bash
git clone https://github.com/IQTLabs/rfml.git
cd rfml
git submodule update --init --recursive
poetry install
```

## Verify install with GPU support (optional)

```bash
$ python -c 'import torch; print(torch.cuda.is_available())'
True
```

If the output does not match or errors occur, try installing Pytorch manually ([current version](https://pytorch.org/get-started/locally/) or [previous versions](https://pytorch.org/get-started/previous-versions/)).
#### Example

```bash
pip install torch==2.0.1 torchvision==0.15.2
```


# Building a model


## Approach

Our current approach is to capture examples of signals of interest to create labeled datasets. There are many methods for doing this and many challenges to consider. One practical method for accomplishing this is to isolate signals of interest and compare those to a specific background RF environment. For simplicity we apply the same label to all the signals present in the background environment samples. We use this to essentially teach the model to ignore those signals. For this to work, it is important that the signals of interest are isolated from the background RF environment. Since it is really tough these days to find an RF free environment, we have build a mini-faraday cage enclosure by lining the inside of a pelican case with foil. There are lots of instructions, like [this one](https://mosequipment.com/blogs/blog/build-your-own-faraday-cage), available online if you want to build your own. With this, the signal will be very strong, so make sure you adjust the SDR's gain appropriately.

## Labeling IQ Data

The scripts in [label_scripts](./label_scripts/) use signal processing to automatically label IQ data. The scripts looks at the signal power to detect when there is a signal present in the IQ data and estimate the occupied bandwidth of the signal. 

### Tuning Autolabeling

In the labeling scripts, the settings for autolabeling need to be tuned for the type of signals that were collected.

```python
annotation_utils.annotate(
    rfml.data.Data(filename),
    avg_window_len=256,                             # The window size to use when averaging signal power
    power_estimate_duration=0.1,                    # Process the file in chunks of power_estimate_duration seconds
    debug_duration=0.25,                            # If debug==True, then plot debug_duration seconds of data in debug plots
    debug=False,                                    # Set True to enable debugging plots                       
    verbose=False,                                  # Set True to eanble verbose messages 
    dry_run=False,                                  # Set True to disable annotations being written to SigMF-Meta file. 
    bandwidth_estimation=True,                      # If set to True, will estimate signal bandwidth using Gaussian Mixture Models. If set to a float will estimate signal bandwidth using spectral thresholding. 
    force_threshold_db=None,                        # Used to manually set the threshold used for detecting a signal and creating an annotation. If None, then the automatic threshold calculation will be used instead.
    overwrite=True,                                 # If True, any existing annotations in the .sigmf-meta file will be removed
    max_annotations=None,                           # If set, limits the number of annotations to add. 
    dc_block=None,                                  # De-emphasize the DC spike when trying to calculate the frequencies for a signal
    time_start_stop=None,                           # Sets the start/stop time for annotating the recording (must be tuple or list of length 2).
    n_components = None,                            # Sets the number of mixture components to use when calculating signal detection threshold. If not set, then automatically calculated from labels. 
    n_init=1,                                       # Number of initializations to use in Gaussian Mixture Method. Increasing this number can significantly increase run time. 
    fft_len=256,                                    # FFT length used in calculating bandwidth
    labels = {                                      # The labels dictionary defines the annotations that the script will attempt to find. 
        "mavic3_video": {                           # The dictionary keys define the annotation labels. Only a key is necessary. 
            "bandwidth_limits": (8e6, None),        # Optional. Set min/max bandwidth limit for a signal. If None, no min/max limit. 
            "annotation_length": (10000, None),     # Optional. Set min/max annoation length in number of samples. If None, no min/max limit.
            "annotation_seconds": (0.0001, 0.0025), # Optional. Set min/max annotation length in seconds. If None, no min/max limit. 
            "set_bandwidth": (-8.5e6, 9.5e6)        # Optional. Ignore bandwidth estimation, set bandwidth manually. Limits are in relation to center frequency. 
        }
    }
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
- `annotation_length` sets the minimum and maximum number of samples for an annotation. If the signal does not satisfy these constraints, it will not be annotated. Try modifying this.
- `annotation_seconds` follows the same concept as `annotation_length` except with units being changed to seconds instead of samples.
- `bandwidth_limits` sets the minimum and maximum bandwidth (in Hz) for a signal to be annotated. These limits will be compared against the estimated bandwidth. Try modifying these limits or the bandwidth estimation method. 

## Training a Model

After you have finished labeling your data, the next step is to train a model on it. This repo makes it easy to train both IQ and Spectrogram based models from sigmf data. 

### Configure

This repo provides an automated script for training and evaluating models. To do this, configure the [mixed_experiments.py](experiments/mixed_experiments.py) file or create your own experiment to point to the data you want to use and set the training parameters:

```python
    "experiment_0": { # A name to refer to the experiment
        "class_list": ["mavic3_video","mavic3_remoteid","environment"], # The labels that are present in the sigmf-meta files
        "train_dir": ["data/samples/mavic-30db", "data/samples/mavic-0db", "data/samples/environment"], # Directory with SigMF files
        "iq_epochs": 10, # Number of epochs for IQ training, if it is 0 or None, it will be skipped
        "spec_epochs": 10, # Number of epochs for spectrogram training, if it is 0 or None, it will be skipped
        "notes": "DJI Mavic3 Detection" # Notes to your future self
    }
```

Once you have an experiments file configured, run it:

```bash
python3 mixed_experiments.py
```

Once the training has completed, it will print out the logs location, model accuracy, and the location of the best checkpoint: 

```bash
I/Q TRAINING COMPLETE


Find results in experiment_logs/experiment_1/iq_logs/08_08_2024_09_17_32

Total Accuracy: 98.10%
Best Model Checkpoint: experiment_logs/experiment_1/iq_logs/08_08_2024_09_17_32/checkpoints/checkpoint.ckpt
```

### Convert & Export IQ Models

Once you have a trained model, you need to convert it into a portable format that can easily be served by TorchServe. To do this, use **export_model.py**:

```bash
python3 rfml/export_model.py --model_name=drone_detect --checkpoint=experiment_logs/experiment_1/iq_logs/08_08_2024_09_17_32/checkpoints/checkpoint.ckpt --index_to_name=experiment_logs/experiment_1/iq_logs/08_08_2024_09_17_32/index_to_name.json
```
This will create a **_torchscript.pt** and **_torchserve.pt** file in the weights folder.

A **.mar** file will also be created in the [models/](./models/) folder. [GamutRF](https://github.com/IQTLabs/gamutRF) can run this model and use it to classify signals.


## Files


[annotation_utils.py](rfml/annotation_utils.py) - DSP based automated labelling tools

[auto_label.py](rfml/auto_label.py) - CV based automated labelling tools

[data.py](rfml/data.py) - RF data operations tool

[experiment.py](rfml/experiment.py) - Class to manage experiments 

[export_model.py](rfml/export_model.py) - Convert and export model checkpoints to Torchscript/Torchserve/MAR format. 

[models.py](rfml/models.py) - Class for I/Q models (based on TorchSig) 

[experiments/](experiments/) - Experiment configurations and run script

[sigmf_pytorch_dataset.py](rfml/sigmf_pytorch_dataset.py) - PyTorch style dataset class for SigMF data (based on TorchSig) 

[spectrogram.py](rfml/spectrogram.py) - Spectrogram tools 

[test_data.py](rfml/test_data.py) - Test for data.py (might be outdated)

[train_iq.py](rfml/train_iq.py) - Training script for I/Q models

[train_spec.py](rfml/train_spec.py) - Training script for spectrogram models

[zst_parse.py](rfml/zst_parse.py) - ZST file parsing tool, for GamutRF-style filenames  

The [notebooks/](./notebooks/) directory contains various experiments we have conducted during development.


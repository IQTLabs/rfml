## Building a Model

### Label Data
 use the [label_scripts](./label_scripts/) to automatically label the data
 





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



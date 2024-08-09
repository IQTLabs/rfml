import glob

from pathlib import Path
from tqdm import tqdm
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import annotation_utils
import data as data_class



data_globs = {
    "dji_samples": [
        "/home/iqt/lberndt/gamutrf-depoly/data/samples/mavic-30gain/samples_1723144831.684000_2408703998Hz_20480000sps.raw.sigmf-meta",
        "/home/iqt/lberndt/gamutrf-depoly/data/samples/mavic-40gain/samples_1723144790.020000_2408703998Hz_20480000sps.raw.sigmf-meta",
        "/home/iqt/lberndt/gamutrf-depoly/data/samples/mavic-30db/samples_1722867361.666000_2408703998Hz_20480000sps.raw.sigmf-meta",
        "/home/iqt/lberndt/gamutrf-depoly/data/samples/mavic-0db/samples_1722883251.180000_2408703998Hz_20480000sps.raw.sigmf-meta"
    ]
}



for label in data_globs:
    for data_glob in data_globs[label]:
        for f in tqdm(glob.glob(str(Path(data_glob)))):
            # annotation_utils.annotate(f, label=label, avg_window_len=256, avg_duration=0.25, debug=True, estimate_frequency=True, spectral_energy_threshold=0.99, force_threshold_db=-40)

            print(f"\nAnnotating {f}\n-----------------------------------\n")

            data_obj = data_class.Data(f)
            annotation_utils.reset_annotations(data_obj)
            # annotation_utils.annotate(
            #     f, 
            #     label="mavic3_remoteid", 
            #     avg_window_len=256, 
            #     avg_duration=0.10, 
            #     debug=False, 
            #     estimate_frequency=True, 
            #     spectral_energy_threshold=0.90, 
            #     #force_threshold_db=-48, 
            #     overwrite=True, 
            #     min_bandwidth=1e5, 
            #     max_bandwidth=2e6,
            #     min_annotation_length=500, 
            #     # max_annotations=500, 
            #     dc_block=True
            # )
            annotation_utils.annotate(
                f, 
                label="mavic3_video",  
                avg_window_len=256, 
                avg_duration=0.25, 
                debug=False, 
                estimate_frequency=True, 
                spectral_energy_threshold=0.99, 
                force_threshold_db=-58, 
                overwrite=True, 
                min_bandwidth=8e6, 
                min_annotation_length=10000, 
                # max_annotations=500, 
                dc_block=True
            )
            
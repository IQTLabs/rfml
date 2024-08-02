import glob

from pathlib import Path
from tqdm import tqdm

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import annotation_utils
import data as data_class



data_globs = {
    # "mini2_video": [
    #     "data/gamutrf/gamutrf-arl/01_30_23/mini2_iq_label/*.sigmf-meta",
    # ],
    "mini2_video": [
        # "data/gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/test_iq_label/dji-mini2-200m-0deg-5735mhz-lp-50-gain_20p5Msps_craft_flying-1.raw.sigmf-meta",
        "/home/iqt/lberndt/gamutrf-depoly/data/samples/drone-30db/samples_1722545037.483000_2408703998Hz_20480000sps.raw.sigmf-meta",
    ]
}



for label in data_globs:
    for data_glob in data_globs[label]:
        for f in tqdm(glob.glob(str(Path(data_glob)))):
            # annotation_utils.annotate(f, label=label, avg_window_len=256, avg_duration=0.25, debug=True, estimate_frequency=True, spectral_energy_threshold=0.99, force_threshold_db=-40)

            data_obj = data_class.Data(f)
            annotation_utils.reset_annotations(data_obj)
            annotation_utils.annotate(
                f, 
                label="mavic3_video", 
                avg_window_len=256, 
                avg_duration=0.25, 
                debug=False, 
                estimate_frequency=True, 
                spectral_energy_threshold=0.95, 
                force_threshold_db=-58, 
                overwrite=False, 
                min_bandwidth=16e6, 
                min_annotation_length=10000, 
                # max_annotations=500, 
                dc_block=True
            )
            annotation_utils.annotate(
                f, 
                label="mavic3_telem",  
                avg_window_len=256, 
                avg_duration=0.25, 
                debug=False, 
                estimate_frequency=True, 
                spectral_energy_threshold=0.95, 
                force_threshold_db=-58, 
                overwrite=False, 
                max_bandwidth=16e6, 
                min_annotation_length=10000, 
                # max_annotations=500, 
                dc_block=True
            )
            
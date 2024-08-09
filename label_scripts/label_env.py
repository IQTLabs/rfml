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
        "/home/iqt/lberndt/gamutrf-depoly/data/samples/environment/samples_1722872733.648000_2408703998Hz_20480000sps.raw.sigmf-meta"
        #"/home/iqt/lberndt/gamutrf-depoly/data/samples/mavic-0db/samples_1722883251.180000_2408703998Hz_20480000sps.raw.sigmf-meta"
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
                label="environment", 
                avg_window_len=256, 
                avg_duration=0.10, 
                debug=False, 
                estimate_frequency=True, 
                spectral_energy_threshold=0.90, 
                #force_threshold_db=-48, 
                overwrite=True, 
                min_bandwidth=None, 
                max_bandwidth=None, 
                min_annotation_length=1000, 
                # max_annotations=500, 
                dc_block=True
            )

            
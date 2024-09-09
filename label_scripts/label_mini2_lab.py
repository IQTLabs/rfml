import glob

from pathlib import Path
from tqdm import tqdm

import rfml.annotation_utils as annotation_utils
import rfml.data as data_class


data_globs = [
    "/data/s3_gamutrf/gamutrf-arl/01_30_23/mini2/*.zst",
]


for file_glob in data_globs:
    for f in tqdm(glob.glob(str(Path(file_glob)))):

        data_obj = data_class.Data(f)
        annotation_utils.reset_annotations(data_obj)
        annotation_utils.annotate(
            data_obj,
            # label="mini2_video",
            avg_window_len=256,
            avg_duration=0.25,
            debug=False,
            spectral_energy_threshold=True,
            # force_threshold_db=-60,
            overwrite=False,
            # min_bandwidth=16e6,
            # min_annotation_length=10000,
            # max_annotations=500,
            # dc_block=True,
            # time_start_stop=(1,3.5), 
            # necessary={
            #     "annotation_seconds": (0.001, -1)
            # },
            labels = {
                "mini2_video": {
                    "bandwidth_limits": (16e6, None),
                    "annotation_length": (10000, None),
                    "annotation_seconds": (0.001, None),
                    "set_bandwidth": (-9e6, 9e6)
                },
                "mini2_telem": {
                    "bandwidth_limits": (None, 16e6),
                    "annotation_length": (10000, None),
                    "annotation_seconds": (None, 0.001),
                }
            }
        )
        # annotation_utils.annotate(
        #     data_obj,
        #     label="mini2_telem",
        #     avg_window_len=256,
        #     avg_duration=0.25,
        #     debug=False,
        #     spectral_energy_threshold=True,
        #     # force_threshold_db=-58,
        #     overwrite=False,
        #     max_bandwidth=16e6,
        #     min_annotation_length=10000,
        #     # max_annotations=500,
        #     # dc_block=True,
        #     necessary={
        #         "annotation_seconds": (0, 0.001)
        #     },
        # )

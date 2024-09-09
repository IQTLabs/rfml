import glob

from pathlib import Path
from tqdm import tqdm

import rfml.annotation_utils as annotation_utils
import rfml.data as data_class


data_globs = [
    # "/data/s3_gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/iq_recordings/dji-mini2-0to100m-0deg-5735mhz-lp-50-gain_20p5Msps_craft_flying-1.sigmf-meta"
    # "/data/s3_gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/iq_recordings/dji-mini2-0to100m-0deg-5735mhz-lp-60-gain_20Msps_craft_flying-1.sigmf-meta"
    "/data/s3_gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/iq_recordings/*.sigmf-meta",
    # "/data/s3_gamutrf/gamutrf-birdseye-field-days/leesburg_field_day_2022_06_15/iq_recordings/*.sigmf-meta",
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
            bandwidth_estimation=True,
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
                    "set_bandwidth": (-8.5e6, 9.5e6)
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

import glob

from pathlib import Path
from tqdm import tqdm

import rfml.annotation_utils as annotation_utils
import rfml.data as data_class


data_globs = [
    # "/data/s3_gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/iq_recordings/dji-mini2-0to100m-0deg-5735mhz-lp-50-gain_20p5Msps_craft_flying-1.sigmf-meta"
    # "/data/s3_gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/iq_recordings/dji-mini2-0to100m-0deg-5735mhz-lp-60-gain_20Msps_craft_flying-1.sigmf-meta"
    # "/data/s3_gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/iq_recordings/*.sigmf-meta",
    # "/data/s3_gamutrf/gamutrf-birdseye-field-days/leesburg_field_day_2022_06_15/iq_recordings/*.sigmf-meta",
    "/data/s3_gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/iq_recordings/dji-mini2-0to100m-0deg-5735mhz-lp-50-gain_20p5Msps_craft_flying-1.sigmf-meta"
]


for file_glob in data_globs:
    for f in tqdm(glob.glob(str(Path(file_glob)))):

        data_obj = data_class.Data(f)
        annotation_utils.reset_annotations(data_obj)
        annotation_utils.annotate(
            data_obj,
            avg_window_len=256,
            debug=False,
            bandwidth_estimation=True,
            overwrite=False,
            power_estimate_duration=0.1,
            n_components=4,
            n_init=2,
            labels={
                "mini2_video": {
                    "bandwidth_limits": (16e6, None),
                    "annotation_length": (10000, None),
                    "annotation_seconds": (0.001, None),
                    "set_bandwidth": (-8.5e6, 9.5e6),
                },
                "mini2_telem": {
                    "bandwidth_limits": (None, 16e6),
                    "annotation_length": (10000, None),
                    "annotation_seconds": (None, 0.001),
                },
            },
        )

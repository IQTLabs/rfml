import glob

from pathlib import Path
from tqdm import tqdm

import rfml.annotation_utils as annotation_utils
import rfml.data as data_class

data_globs = [
    # "/data/s3_gamutrf/gamutrf-nz-wifi/gamutrf_ax_gain10_2430000000Hz_20480000sps.raw.zst",
    "/data/s3_gamutrf/gamutrf-nz-wifi/*.zst"
]


for file_glob in data_globs:
    for f in tqdm(glob.glob(str(Path(file_glob)))):

        data_obj = data_class.Data(f)
        annotation_utils.reset_annotations(data_obj)
        annotation_utils.annotate(
            data_obj,
            avg_window_len=256,
            avg_duration=0.25,
            debug=False,
            verbose=False,
            bandwidth_estimation=0.99,
            overwrite=False,
            labels={
                "wifi": {
                    "bandwidth_limits": (5e6, None),
                    # "annotation_length": (10000, None),
                    "annotation_seconds": (0.0005, None),
                }
            },
        )

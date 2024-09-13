import glob

from pathlib import Path
from tqdm import tqdm

import rfml.annotation_utils as annotation_utils
import rfml.data as data_class


mavic_globs = [
    "/data/s3_gamutrf/gamutrf-lab-collect/mavic-30db/*.sigmf-meta",
    # "/data/s3_gamutrf/gamutrf-lab-collect/mavic-0db/*.sigmf-meta",
    # "/data/s3_gamutrf/gamutrf-drone-detection/drone.sigmf-meta",
]

for file_glob in mavic_globs:
    for f in tqdm(glob.glob(str(Path(file_glob)))):
        data_obj = data_class.Data(f)
        annotation_utils.reset_annotations(data_obj)
        annotation_utils.annotate(
            data_obj,
            avg_window_len=256,
            power_estimate_duration=0.1,
            # debug_duration=0.25,
            # debug=True,
            # verbose=True,
            bandwidth_estimation=True,
            overwrite=False,
            labels={
                "mavic3_video": {
                    "bandwidth_limits": (8e6, None),
                    "annotation_length": (10000, None),
                    "annotation_seconds": (0.0001, 0.0025),
                    # "set_bandwidth": (-8.5e6, 9.5e6)
                },
                "mavic3_telem": {
                    "bandwidth_limits": (None, 5e6),
                    # "annotation_length": (10000, None),
                    "annotation_seconds": (0.0003, 0.001),
                },
            },
        )

import glob

from pathlib import Path
from tqdm import tqdm

import rfml.annotation_utils as annotation_utils
import rfml.data as data_class

data_globs = ["/data/s3_gamutrf/gamutrf-lucas-collect/mini2/*.zst"]

for file_glob in data_globs:
    for f in tqdm(glob.glob(str(Path(file_glob)))):
        data_obj = data_class.Data(f)
        annotation_utils.reset_annotations(data_obj)
        annotation_utils.annotate(
            data_obj,
            avg_window_len=256,
            debug=False,
            bandwidth_estimation=0.99,  # True,
            overwrite=False,
            # power_estimate_duration = 0.1,
            # n_components=3,
            # n_init=2,
            # dry_run=True,
            # time_start_stop=(1,None),
            labels={
                "mini2_video": {
                    "bandwidth_limits": (16e6, None),
                    "annotation_length": (10000, None),
                    "annotation_seconds": (0.001, None),
                    # "set_bandwidth": (-8.5e6, 9.5e6)
                },
                # "mini2_telem": {
                #     "bandwidth_limits": (None, 16e6),
                #     "annotation_length": (10000, None),
                #     "annotation_seconds": (None, 0.001),
                # }
            },
        )


# data_globs = [
#    "/data/s3_gamutrf/gamutrf-lucas-collect/environment/*.zst"
# ]


# for file_glob in data_globs:
#     for f in tqdm(glob.glob(str(Path(file_glob)))):
#         data_obj = data_class.Data(f)
#         annotation_utils.reset_annotations(data_obj)
#         annotation_utils.annotate(
#             data_obj,
#             avg_window_len=1024,
#             debug=False,
#             bandwidth_estimation=0.99,#True,
#             overwrite=False,
#             # power_estimate_duration = 0.1,
#             # n_components=3,
#             # n_init=2,
#             # dry_run=True,
#             labels = {
#                 "environment": {
#                     "annotation_length": (2048, None),
#                 },
#             }
#         )

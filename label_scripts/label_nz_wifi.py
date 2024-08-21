import glob

from pathlib import Path
from tqdm import tqdm

import rfml.annotation_utils as annotation_utils

s3_data = {
    "anom_wifi": [
        # "data/gamutrf/gamutrf-nz-anon-wifi/*.sigmf-meta",
        # "data/gamutrf/gamutrf-wifi-and-anom-bladerf/anom*.sigmf-meta"
    ],
    "wifi": [
        # "data/gamutrf/gamutrf-nz-nonanon-wifi/*.sigmf-meta",
        # "data/gamutrf/gamutrf-wifi-and-anom-bladerf/wifi*.sigmf-meta",
        "data/gamutrf/gamutrf-nz-wifi/*.sigmf-meta",
    ],
}

for label in s3_data:
    for data_glob in s3_data[label]:
        for f in tqdm(glob.glob(str(Path(data_glob)))):
            annotation_utils.annotate(f, label=label, avg_window_len=256, avg_duration=3, debug=False)
import glob

from pathlib import Path
from tqdm import tqdm

import rfml.annotation_utils as annotation_utils
import rfml.data as data_class
import sys

root = sys.argv[1]

# generated with
# $ for i in am fm ; do /scratch/iqtlabs/rfml/utils/siggen.py --samp_rate 1000000 --siggen $i --int_count 1000 ; done
data_globs = {
    "am": [f"{root}/am.sigmf-meta"],
    "fm": [f"{root}/fm.sigmf-meta"],
}


for data_glob in data_globs["am"]:
    for f in tqdm(glob.glob(str(Path(data_glob)))):
        data_obj = data_class.Data(f)
        annotation_utils.reset_annotations(data_obj)
        annotation_utils.annotate(
            data_obj,
            avg_window_len=1024,
            debug=False,
            bandwidth_estimation=False,
            force_threshold_db=-200,
            overwrite=True,
            dc_block=True,
            labels={"am": {"annotation_length": (1024, None)}},
        )

for data_glob in data_globs["fm"]:
    for f in tqdm(glob.glob(str(Path(data_glob)))):
        data_obj = data_class.Data(f)
        annotation_utils.reset_annotations(data_obj)
        annotation_utils.annotate(
            data_obj,
            avg_window_len=1024,
            debug=False,
            bandwidth_estimation=False,
            force_threshold_db=-200,
            overwrite=True,
            dc_block=True,
            labels={
                "fm": {
                    "annotation_length": (1024, None),
                }
            },
        )

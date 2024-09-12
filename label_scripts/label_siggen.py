import glob

from pathlib import Path
from tqdm import tqdm

import rfml.annotation_utils as annotation_utils
import rfml.data as data_class

# generated with
# $ for i in am fm ; do /scratch/iqtlabs/rfml/utils/siggen.py --samp_rate 1000000 --siggen $i --int_count 1000 ; done
data_globs = {
    "am": ["/scratch/tmp/rfmltest/am.sigmf-meta"],
    "fm": ["/scratch/tmp/rfmltest/fm.sigmf-meta"],
}


for data_glob in data_globs["am"]:
    for f in tqdm(glob.glob(str(Path(data_glob)))):
        data_obj = data_class.Data(f)
        annotation_utils.reset_annotations(data_obj)
        annotation_utils.annotate(
            f,
            label="am",
            avg_window_len=256,
            avg_duration=0.10,
            debug=False,
            spectral_energy_threshold=0.95,
            # force_threshold_db=-1,
            overwrite=True,
            min_bandwidth=1e2,
            min_annotation_length=256,
            dc_block=True,
        )

for data_glob in data_globs["fm"]:
    for f in tqdm(glob.glob(str(Path(data_glob)))):
        data_obj = data_class.Data(f)
        annotation_utils.reset_annotations(data_obj)
        annotation_utils.annotate(
            f,
            label="fm",
            avg_window_len=256,
            avg_duration=0.10,
            debug=False,
            # spectral_energy_threshold=0.95,
            force_threshold_db=-0.1,
            overwrite=True,
            min_bandwidth=1e2,
            min_annotation_length=256,
            dc_block=True,
        )

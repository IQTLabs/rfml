import copy
import glob
import json

from datetime import datetime
from pathlib import Path

import rfml.data


def manual_to_sigmf(file, datatype, sample_rate, frequency, iso_date_string):
    # change to .sigmf-data
    if file.suffix in [".raw"]:
        file = file.rename(file.with_suffix(".sigmf-data"))
    else:
        raise NotImplementedError

    sigmf_meta = copy.deepcopy(rfml.data.SIGMF_META_DEFAULT)
    sigmf_meta["global"]["core:dataset"] = str(file)
    sigmf_meta["global"]["core:datatype"] = datatype
    sigmf_meta["global"]["core:sample_rate"] = sample_rate
    sigmf_meta["captures"][0]["core:frequency"] = frequency
    sigmf_meta["captures"][0]["core:datetime"] = (
        datetime.fromisoformat(iso_date_string)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )

    with open(file.with_suffix(".sigmf-meta"), "w") as outfile:
        print(f"Saving {file.with_suffix('.sigmf-meta')}\n")
        outfile.write(json.dumps(sigmf_meta, indent=4))


if __name__ == "__main__":

    data_globs = [
        (
            "/data/s3_gamutrf/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/iq_recordings/*.raw",
            {
                "datatype": "cf32_le",
                "sample_rate": 20500000,
                "frequency": 5735000000,
                "iso_date_string": "2022-05-26",
            },
        ),
        (
            "/data/s3_gamutrf/gamutrf-birdseye-field-days/leesburg_field_day_2022_06_15/iq_recordings/*.raw",
            {
                "datatype": "cf32_le",
                "sample_rate": 20500000,
                "frequency": 5735000000,
                "iso_date_string": "2022-06-15",
            }
        )
    ]
    for file_glob, metadata in data_globs:
        files = glob.glob(str(Path(file_glob)))
        for f in files:
            f = Path(f)
            manual_to_sigmf(
                f,
                metadata["datatype"],
                metadata["sample_rate"],
                metadata["frequency"],
                metadata["iso_date_string"],
            )

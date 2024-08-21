import glob
import psutil
import resource


from itertools import repeat
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm

import rfml.annotation_utils as annotation_utils

def worker_wrapper(fn, kwargs):
    def try_fn():
        try:
            return fn(**kwargs)
        except Exception as e: 
            print(e)
            print(kwargs)

    return try_fn()
    #return fn(**kwargs)

s3_data = {
    ("anom_wifi", 5000000): [
        "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/ettus/*5000000*.sigmf-meta",
        "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/*5000000*.sigmf-meta",
        "/data/s3_gamutrf/gamutrf-nz-anon-wifi/*5000000*.sigmf-meta",
        "/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf/anom*5000000*.sigmf-meta",
    ],
    ("anom_wifi", 10000000): [
        "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/ettus/*10000000*.sigmf-meta",
        "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/*10000000*.sigmf-meta",
        "/data/s3_gamutrf/gamutrf-nz-anon-wifi/*10000000*.sigmf-meta",
        "/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf/anom*10000000*.sigmf-meta",
    ],
    ("anom_wifi", 20000000): [
        "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/ettus/*20000000*.sigmf-meta",
        "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/*20000000*.sigmf-meta",
        "/data/s3_gamutrf/gamutrf-nz-anon-wifi/*20000000*.sigmf-meta",
        "/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf/anom*20000000*.sigmf-meta",
    ],
    ("wifi", 5000000): [
        "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/ettus/*5000000*.sigmf-meta",
        "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/*5000000*.sigmf-meta",
        "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi/*5000000*.sigmf-meta",
        "/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf/wifi*5000000*.sigmf-meta",
    ],
    ("wifi", 10000000): [
        "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/ettus/*10000000*.sigmf-meta",
        "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/*10000000*.sigmf-meta",
        "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi/*10000000*.sigmf-meta",
        "/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf/wifi*10000000*.sigmf-meta",
    ],
    ("wifi", 20000000): [
        "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/ettus/*20000000*.sigmf-meta",
        "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/*20000000*.sigmf-meta",
        "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi/*20000000*.sigmf-meta",
        "/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf/wifi*20000000*.sigmf-meta",
    ],
}

job_args = []
for label in s3_data:
    for data_glob in s3_data[label]:
        for f in tqdm(glob.glob(str(Path(data_glob)))):
            # job_args.append([f, label])
            job_args.append({
                'filename': f, 
                'label': label[0],
                'avg_duration':3, 
                'estimate_frequency': label[1],
                'time_start_stop': 5,
            })

# annotation_utils.annotate(f, label=label, avg_window_len=256, avg_duration=3, debug=False)

# Calculate the maximum memory limit (80% of available memory)
virtual_memory = psutil.virtual_memory()
available_memory = virtual_memory.available
memory_limit = int(available_memory * 0.8)

# Set the memory limit
resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

pool = Pool(16)
# common_args = {
#     'avg_duration':3, 
#     'estimate_frequency':  
#     'time_start_stop': 5,
# }
# pool.starmap(partial(annotation_utils.annotate, **common_args), job_args)


args_for_starmap = zip(repeat(annotation_utils.annotate), job_args)
pool.starmap(worker_wrapper, args_for_starmap)
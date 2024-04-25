import glob

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

import data
import wifi_label_utils


def annotate(filename, label, avg_window_len, avg_duration=-1, debug=False):
    
    data_obj = data.Data(filename)

    # use a seconds worth of data to calculate threshold
    if avg_duration > -1:
        iq_samples = data_obj.get_samples(n_samples=int(data_obj.metadata["global"]["core:sample_rate"]*avg_duration))
    else:
        iq_samples = data_obj.get_samples()
        
    avg_pwr = wifi_label_utils.moving_average(iq_samples, avg_window_len)
    avg_pwr_db = 10*np.log10(avg_pwr)

    
    # guess_threshold = (np.max(avg_pwr_db) + np.mean(avg_pwr_db))/2
    guess_threshold = 1.05 * np.max(avg_pwr_db)
    
    if debug:
        print(f"{np.max(avg_pwr_db)=}")
        print(f"{np.mean(avg_pwr_db)=}")
        print(f"{guess_threshold=}")
        print(f"{len(avg_pwr_db)=}")
        
        plt.figure()
        plt.plot(avg_pwr_db[int(0*20480000e-2):int(5*20.48e6)])
        plt.axhline(y = guess_threshold, color = 'r', linestyle = '-') 
        plt.show()
        
    wifi_label_utils.annotate_power_squelch(data_obj, label, guess_threshold, avg_window_len, skip_validate=True)


for f in tqdm(glob.glob("data/gamutrf/gamutrf-wifi-and-anom-bladerf/wifi*.sigmf-meta")):
    annotate(f, label="wifi", avg_window_len=256, avg_duration=4, debug=False)


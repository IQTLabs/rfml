import data as data_class
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

def moving_average(complex_iq, window_len):
    return (
        np.convolve(np.abs(complex_iq) ** 2, np.ones(window_len), "valid") / window_len
    )

def power_squelch(iq_samples, threshold, window):
    avg_pwr = moving_average(iq_samples, window)
    avg_pwr_db = 10 * np.log10(avg_pwr)

    good_samples = np.zeros(len(iq_samples))
    good_samples[np.where(avg_pwr_db > threshold)] = 1

    idx = (
        np.ediff1d(np.r_[0, good_samples == 1, 0]).nonzero()[0].reshape(-1, 2)
    )  # gets indices where signal power above threshold

    return idx


def annotate_power_squelch(data_obj, threshold, avg_window_len, label=None, skip_validate=False):
    iq_samples = data_obj.get_samples()
    idx = power_squelch(iq_samples, threshold=threshold, window=avg_window_len)

    data_obj.sigmf_obj._metadata[data_obj.sigmf_obj.ANNOTATION_KEY] = []
    for start, stop in idx:
        start, stop = int(start), int(stop)
        metadata = {
            "core:freq_lower_edge": data_obj.metadata["captures"][0]["core:frequency"]
            - data_obj.metadata["global"]["core:sample_rate"] / 2,
            "core:freq_upper_edge": data_obj.metadata["captures"][0]["core:frequency"]
            + data_obj.metadata["global"]["core:sample_rate"] / 2,
        }
        if label:
            metadata["core:label"] = label
        data_obj.sigmf_obj.add_annotation(start, length=stop - start, metadata=metadata)

    data_obj.sigmf_obj.tofile(data_obj.sigmf_meta_filename, skip_validate=skip_validate)
    print(f"Writing {len(data_obj.sigmf_obj._metadata[data_obj.sigmf_obj.ANNOTATION_KEY])} annotations to {data_obj.sigmf_meta_filename}")


def annotate(filename, label, avg_window_len, avg_duration=-1, debug=False, dry_run=False):
    
    data_obj = data_class.Data(filename)

    # use a seconds worth of data to calculate threshold
    if avg_duration > -1:
        iq_samples = data_obj.get_samples(n_samples=int(data_obj.metadata["global"]["core:sample_rate"]*avg_duration))
    else:
        iq_samples = data_obj.get_samples()
        
    avg_pwr = moving_average(iq_samples, avg_window_len)
    avg_pwr_db = 10*np.log10(avg_pwr)


    # current threshold in custom_handler 
    guess_threshold_old = (np.max(avg_pwr_db) + np.mean(avg_pwr_db))/2

    # MAD estimator
    def median_absolute_deviation(series):
        mad = 1.4826 * np.median(np.abs(series - np.median(series)))
        # sci_mad = scipy.stats.median_abs_deviation(series, scale="normal")
        return np.median(series) + 6*mad

    mad = median_absolute_deviation(avg_pwr_db)
    guess_threshold = mad
    
    if debug:
        print(f"{np.max(avg_pwr_db)=}")
        print(f"{np.mean(avg_pwr_db)=}")
        print(f"median absolute deviation threshold = {mad}")
        print(f"using threshold = {guess_threshold}")
        # print(f"{len(avg_pwr_db)=}")
        
        plt.figure()
        db_plot = avg_pwr_db[int(0*20.48e6):int(avg_duration*20.48e6)]
        plt.plot(np.arange(len(db_plot))/data_obj.metadata["global"]["core:sample_rate"], db_plot)
        plt.axhline(y = guess_threshold_old, color = 'g', linestyle = '-', label="old threshold") 
        plt.axhline(y = np.mean(avg_pwr_db), color = 'r', linestyle = '-', label="average") 
        plt.axhline(y = mad, color = 'b', linestyle = '-', label="median absolute deviation threshold") 
        plt.legend(loc="upper left")
        plt.ylabel("dB")
        plt.xlabel("time (seconds)")
        plt.title("Signal Power")
        plt.show()

    if not dry_run:
        annotate_power_squelch(data_obj, guess_threshold, avg_window_len, label=label, skip_validate=True)


def reset_predictions_sigmf(dataset):
    data_files = set([dataset.index[i][1].absolute_path for i in range(len(dataset))])
    for f in data_files: 
        data_obj = data_class.Data(f)
        prediction_meta_path = Path(Path(data_obj.sigmf_meta_filename).parent, f"prediction_{Path(data_obj.sigmf_meta_filename).name}")
        data_obj.sigmf_obj._metadata[data_obj.sigmf_obj.ANNOTATION_KEY] = []
        data_obj.sigmf_obj.tofile(prediction_meta_path, skip_validate=True)
        print(f"Reset annotations in {prediction_meta_path}")

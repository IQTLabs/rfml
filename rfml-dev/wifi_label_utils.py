import numpy as np


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


def annotate_power_squelch(data_obj, label, threshold, avg_window_len, skip_validate=False):
    iq_samples = data_obj.get_samples()
    idx = power_squelch(iq_samples, threshold=threshold, window=avg_window_len)

    data_obj.sigmf_obj._metadata[data_obj.sigmf_obj.ANNOTATION_KEY] = []
    for start, stop in idx:
        start, stop = int(start), int(stop)
        metadata = {
            "core:label": label,
            "core:freq_lower_edge": data_obj.metadata["captures"][0]["core:frequency"]
            - data_obj.metadata["global"]["core:sample_rate"] / 2,
            "core:freq_upper_edge": data_obj.metadata["captures"][0]["core:frequency"]
            + data_obj.metadata["global"]["core:sample_rate"] / 2,
        }
        data_obj.sigmf_obj.add_annotation(start, length=stop - start, metadata=metadata)

    data_obj.sigmf_obj.tofile(data_obj.sigmf_meta_filename, skip_validate=skip_validate)
    print(f"Writing {len(data_obj.sigmf_obj._metadata[data_obj.sigmf_obj.ANNOTATION_KEY])} annotations to {data_obj.sigmf_meta_filename}")

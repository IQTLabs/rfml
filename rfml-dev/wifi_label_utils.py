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

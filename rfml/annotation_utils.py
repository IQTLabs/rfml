# Tools for annotating RF data
import seaborn as sns
import time
from collections.abc import Iterable
import cupy
from cupyx.scipy.signal import spectrogram as cupyx_spectrogram
from cupyx.scipy.ndimage import gaussian_filter as cupyx_gaussian_filter
import cupyx.scipy.signal
import scipy.signal

from rfml.spectrogram import *

import rfml.data as data_class
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from tqdm import tqdm
from sklearn import mixture
import warnings


def annotate(
    data_obj,
    avg_window_len=256,
    debug_duration=0.25,
    debug=False,
    dry_run=False,
    bandwidth_estimation=True,
    force_threshold_db=None,
    overwrite=True,
    max_annotations=None,
    dc_block=None,
    verbose=False,
    time_start_stop=None,
    labels=None,
    power_estimate_duration=1,  # only process n seconds of I/Q samples at a time
    n_components=None,
    n_init=1,
    fft_len=256,
):

    sample_rate = data_obj.metadata["global"]["core:sample_rate"]

    # set n_seek_samples (skip n samples at start) and n_samples (process n samples)
    # if isinstance(time_start_stop, int) and time_start_stop > 0:
    #     n_seek_samples = int(sample_rate * time_start_stop)
    #     n_samples = -1
    if isinstance(time_start_stop, Iterable):
        if len(time_start_stop) != 2:  # or time_start_stop[1] < time_start_stop[0]:
            raise ValueError

        if time_start_stop[0] is None:
            time_start_stop = (0, time_start_stop[1])

        if time_start_stop[1] is None:
            n_samples = -1
        elif time_start_stop[1] < time_start_stop[0]:
            raise ValueError
        else:
            n_samples = int(sample_rate * (time_start_stop[1] - time_start_stop[0]))

        n_seek_samples = int(sample_rate * time_start_stop[0])

    else:
        n_seek_samples = 0
        n_samples = -1

    if n_samples > -1:
        sample_idxs = np.arange(
            n_seek_samples,
            n_seek_samples + n_samples,
            sample_rate * power_estimate_duration,
        )
    else:
        sample_idxs = np.arange(
            n_seek_samples,
            data_obj.sigmf_obj.sample_count,
            sample_rate * power_estimate_duration,
        )

    # if overwrite, delete existing annotations
    if overwrite:
        data_obj.sigmf_obj._metadata[data_obj.sigmf_obj.ANNOTATION_KEY] = []

    if n_components is None:
        n_components = len(labels) + 1 if labels else 2

    n_annotations = 0
    for sample_idx in tqdm(sample_idxs):

        if n_samples > -1:
            get_n_samples = min(
                sample_rate * power_estimate_duration,
                n_samples - (sample_idx - n_seek_samples),
            )
        else:
            get_n_samples = min(
                data_obj.sigmf_obj.sample_count - sample_idx,
                sample_rate * power_estimate_duration,
            )

        iq_samples = data_obj.get_samples(
            n_seek_samples=int(sample_idx), n_samples=int(get_n_samples)
        )

        if iq_samples is None:
            break

        iq_samples = scipy.signal.detrend(
            iq_samples, type="linear", bp=np.arange(0, len(iq_samples), 1024)  # 1024)
        )

        if force_threshold_db:
            threshold_db = force_threshold_db
        else:
            avg_pwr = moving_average(iq_samples, avg_window_len)
            avg_pwr_db = 10 * np.log10(avg_pwr)
            del avg_pwr

            # current threshold in custom_handler
            guess_threshold_old = (np.max(avg_pwr_db) + np.mean(avg_pwr_db)) / 2
            mad = median_absolute_deviation(avg_pwr_db)

            tqdm.write(
                f"Estimating noise floor for signal detection (may take a while)..."
            )

            clf = mixture.GaussianMixture(n_components=n_components, n_init=n_init)
            clf.fit(avg_pwr_db.reshape(-1, 1))
            # TODO: add standard deviation parameter (was 2 *)
            gaussian_mixture_model_estimate = np.min(clf.means_) + 2 * np.sqrt(
                clf.covariances_[np.argmin(clf.means_)].squeeze()
            )

            if verbose:
                print(f"\n{gaussian_mixture_model_estimate=}")
                print(f"{clf.weights_=}")
                print(f"{clf.means_=}")
                print(f"{clf.covariances_=}")
                print(f"{clf.converged_=}\n")

            threshold_db = gaussian_mixture_model_estimate

            if debug:
                print(f"Debug")
                debug_plot(
                    avg_pwr_db,
                    mad,
                    threshold_db,
                    debug_duration,
                    data_obj,
                    guess_threshold_old,
                    force_threshold_db,
                    n_components=n_components,
                )

        tqdm.write(
            f"Using dB threshold = {threshold_db} for detecting signals to annotate"
        )

        # apply power squelch to I/Q samples using dB threshold
        idx = power_squelch(
            iq_samples, threshold=threshold_db, avg_window_len=avg_window_len
        )

        for start, stop in tqdm(idx[:max_annotations]):

            candidate_labels = list(labels.keys())

            start, stop = int(start), int(stop)

            annotation_n_samples = stop - start
            annotation_seconds = annotation_n_samples / sample_rate

            if verbose:
                print(
                    f"\nAnnotation start={(int(sample_idx) + start)/sample_rate}, stop={(int(sample_idx) + stop)/sample_rate}"
                )

            for label in candidate_labels[:]:
                if "annotation_seconds" in labels[label]:
                    min_annotation_seconds, max_annotation_seconds = labels[label][
                        "annotation_seconds"
                    ]
                    if min_annotation_seconds and (
                        annotation_seconds < min_annotation_seconds
                    ):
                        candidate_labels.remove(label)
                        if verbose:
                            print(
                                f"min_annotation_seconds not satisfied for {label}: {annotation_seconds} < {min_annotation_seconds}"
                            )
                        continue
                    if max_annotation_seconds and (
                        annotation_seconds > max_annotation_seconds
                    ):
                        candidate_labels.remove(label)
                        if verbose:
                            print(
                                f"max_annotation_seconds not satisfied for {label}: {annotation_seconds} > {max_annotation_seconds}"
                            )
                        continue

                if "annotation_length" in labels[label]:
                    min_annotation_length, max_annotation_length = labels[label][
                        "annotation_length"
                    ]
                    # skip if proposed annotation length is less than min_annotation_length
                    if min_annotation_length and (
                        annotation_n_samples < min_annotation_length
                    ):
                        candidate_labels.remove(label)
                        if verbose:
                            print(
                                f"min_annotation_length not satisfied for {label}: {annotation_n_samples} < {min_annotation_length}"
                            )
                        continue
                    if max_annotation_length and (
                        annotation_n_samples > max_annotation_length
                    ):
                        candidate_labels.remove(label)
                        if verbose:
                            print(
                                f"max_annotation_length not satisfied for {label}: {annotation_n_samples} > {max_annotation_length}"
                            )
                        continue

            if len(candidate_labels) == 0:
                continue

            freq_edges = None

            # if any candidate labels manually set bandwidth, then skip get_bandwidth
            for label in candidate_labels[:]:
                if "set_bandwidth" in labels[label]:
                    freq_edges = [
                        data_obj.metadata["captures"][0]["core:frequency"]
                        + labels[label]["set_bandwidth"][0],
                        data_obj.metadata["captures"][0]["core:frequency"]
                        + labels[label]["set_bandwidth"][1],
                    ]
                    candidate_labels = [label]
                    break

            if freq_edges is None:
                if bandwidth_estimation and annotation_n_samples < fft_len:
                    if verbose:
                        print(
                            f"annotation length smaller than FFT size {annotation_n_samples} < {fft_len}"
                        )
                    continue

                freq_edges = get_bandwidth(
                    data_obj,
                    iq_samples,
                    start,
                    stop,
                    bandwidth_estimation,
                    dc_block,
                    debug,
                    fft_len=fft_len,
                )

            freq_lower_edge, freq_upper_edge = freq_edges

            bandwidth = freq_upper_edge - freq_lower_edge

            for label in candidate_labels[:]:
                if "bandwidth_limits" in labels[label]:
                    min_bandwidth, max_bandwidth = labels[label]["bandwidth_limits"]
                    if min_bandwidth and bandwidth < min_bandwidth:
                        candidate_labels.remove(label)
                        if verbose:
                            print(
                                f"min_bandwidth not satisfied for {label}, {bandwidth} < {min_bandwidth}, ({freq_lower_edge=}, {freq_upper_edge=})"
                            )
                        continue
                    if max_bandwidth and bandwidth > max_bandwidth:
                        candidate_labels.remove(label)
                        if verbose:
                            print(
                                f"max_bandwidth not satisfied for {label}, {bandwidth} > {max_bandwidth}, ({freq_lower_edge=}, {freq_upper_edge=})"
                            )
                        continue

            if len(candidate_labels) == 0:
                continue
            elif len(candidate_labels) > 1:
                warnings.warn(
                    f"Multiple labels are possible {candidate_labels}. Using first label {candidate_labels[0]}."
                )

            metadata = {
                "core:freq_lower_edge": freq_lower_edge,
                "core:freq_upper_edge": freq_upper_edge,
            }

            metadata["core:label"] = candidate_labels[0]

            data_obj.sigmf_obj.add_annotation(
                int(sample_idx) + start, length=stop - start, metadata=metadata
            )

            if verbose:
                print(f"Adding annotation {metadata}\n")

            n_annotations += 1

    if not dry_run and n_annotations:
        data_obj.sigmf_obj.tofile(data_obj.sigmf_meta_filename, skip_validate=True)
        print(
            f"Writing {len(data_obj.sigmf_obj._metadata[data_obj.sigmf_obj.ANNOTATION_KEY])} annotations to {data_obj.sigmf_meta_filename}"
        )


def get_bandwidth(
    data_obj,
    iq_samples,
    start,
    stop,
    bandwidth_estimation,
    dc_block,
    debug,
    fft_len=256,
):

    if isinstance(bandwidth_estimation, bool) and bandwidth_estimation:
        freq_lower_edge, freq_upper_edge = get_occupied_bandwidth_gmm(
            iq_samples[start:stop],
            data_obj.metadata["global"]["core:sample_rate"],
            data_obj.metadata["captures"][0]["core:frequency"],
            dc_block=dc_block,
            debug=debug,
            fft_len=fft_len,
        )

    elif isinstance(bandwidth_estimation, float):
        freq_lower_edge, freq_upper_edge = get_occupied_bandwidth_spectral_threshold(
            iq_samples[start:stop],
            data_obj.metadata["global"]["core:sample_rate"],
            data_obj.metadata["captures"][0]["core:frequency"],
            spectral_energy_threshold=bandwidth_estimation,
            debug=debug,
            fft_len=fft_len,
        )
    # set bandwidth as full capture bandwidth
    else:
        freq_lower_edge = (
            data_obj.metadata["captures"][0]["core:frequency"]
            - data_obj.metadata["global"]["core:sample_rate"] / 2
        )
        freq_upper_edge = (
            data_obj.metadata["captures"][0]["core:frequency"]
            + data_obj.metadata["global"]["core:sample_rate"] / 2
        )

    return [freq_lower_edge, freq_upper_edge]


def get_occupied_bandwidth_spectral_threshold(
    samples,
    sample_rate,
    center_frequency,
    spectral_energy_threshold,
    debug,
    fft_len=256,
):
    f, t, Sxx = cupyx_spectrogram(
        samples,
        fs=sample_rate,
        return_onesided=False,
        scaling="spectrum",
        # mode="complex",
        detrend=False,
        window=cupyx.scipy.signal.windows.boxcar(fft_len),
    )

    freq_power = cupy.median(cupy.fft.fftshift(Sxx, axes=0), axis=1)

    freq_power_normalized = freq_power / freq_power.sum(axis=0)

    lower_idx = 0
    upper_idx = freq_power_normalized.shape[0]

    while True:
        if (
            freq_power_normalized[lower_idx:upper_idx].sum()
            <= spectral_energy_threshold
        ):
            break

        if freq_power_normalized[lower_idx] < freq_power_normalized[upper_idx - 1]:
            lower_idx += 1
        else:
            upper_idx -= 1

    freq_upper_edge = (
        center_frequency
        - (freq_power.shape[0] / 2 - upper_idx) / freq_power.shape[0] * sample_rate
    )
    freq_lower_edge = (
        center_frequency
        - (freq_power.shape[0] / 2 - lower_idx) / freq_power.shape[0] * sample_rate
    )

    if debug:
        max_power_idx = int(cupy.asnumpy(freq_power_normalized.argmax(axis=0)))

        print(
            f"\nEstimated frequency edges {freq_lower_edge=}, {freq_upper_edge=}, {lower_idx=}, {upper_idx=}\n"
        )
        ###
        # Figure 1
        ###
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(
            cupy.asnumpy(10 * cupy.log10(cupy.fft.fftshift(Sxx, axes=0))),
            origin="lower",
        )
        axs[0].axhline(y=upper_idx, color="r", linestyle="-")
        axs[0].axhline(y=lower_idx, color="r", linestyle="-")
        # axs[0].pcolormesh(cupy.asnumpy(t), cupy.asnumpy(cupy.fft.fftshift(f)), cupy.asnumpy(cupy.fft.fftshift(Sxx, axes=0)))
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        axs[1].imshow(
            np.tile(
                np.expand_dims(
                    cupy.asnumpy(cupy.median(cupy.fft.fftshift(Sxx, axes=0), axis=1)), 1
                ),
                25,
            ),
            origin="lower",
        )
        # axs[1].axhline(y = upper_idx, color = 'r', linestyle = '-')
        # axs[1].axhline(y = lower_idx, color = 'g', linestyle = '-')

        axs[2].imshow(
            np.tile(np.expand_dims(cupy.asnumpy(freq_power_normalized), 1), 25),
            origin="lower",
        )
        axs[2].axhline(y=max_power_idx, color="pink", linestyle="-")
        axs[2].axhline(y=upper_idx, color="r", linestyle="-")
        axs[2].axhline(y=lower_idx, color="r", linestyle="-")
        plt.show()

    return freq_lower_edge, freq_upper_edge


def get_occupied_bandwidth_gmm(
    samples,
    sample_rate,
    center_frequency,
    dc_block=False,
    debug=False,
    fft_len=256,
):

    f, t, Sxx = cupyx_spectrogram(
        samples,
        fs=sample_rate,
        return_onesided=False,
        scaling="spectrum",
        # mode="complex",
        detrend=False,
        window=cupyx.scipy.signal.windows.boxcar(fft_len),
    )

    # Sxx = np.abs(Sxx)**2

    # freq_power = cupy.asnumpy(cupy.fft.fftshift(Sxx, axes=0))
    freq_power = cupy.median(cupy.fft.fftshift(Sxx, axes=0), axis=1)
    # freq_power = cupy.median(cupyx_gaussian_filter(cupy.fft.fftshift(Sxx, axes=0), sigma=2, mode="reflect"), axis=1)

    # lessen DC
    if dc_block:
        dc_start = int(len(freq_power) / 2) - 1
        dc_stop = int(len(freq_power) / 2) + 2
        freq_power[dc_start:dc_stop] /= 2

    freq_power_normalized = freq_power / freq_power.sum(axis=0)

    #####
    start_time = time.time()
    clf = mixture.GaussianMixture(n_components=2)
    predictions = clf.fit_predict(
        cupy.asnumpy(10 * cupy.log10(freq_power_normalized)).reshape(-1, 1)
    )
    signal_predictions = np.zeros(len(predictions))
    signal_predictions[np.where(predictions == np.argmax(clf.means_))] = 1

    signal_predictions_idx = (
        np.ediff1d(np.r_[0, signal_predictions == 1, 0]).nonzero()[0].reshape(-1, 2)
    )  # gets indices where signal power above threshold

    freq_bounds = signal_predictions_idx[
        np.argmax(np.abs(signal_predictions_idx[:, 0] - signal_predictions_idx[:, 1]))
    ]
    lower_idx = freq_bounds[0]
    upper_idx = freq_bounds[1]

    freq_upper_edge = (
        center_frequency
        - (freq_power.shape[0] / 2 - upper_idx) / freq_power.shape[0] * sample_rate
    )
    freq_lower_edge = (
        center_frequency
        - (freq_power.shape[0] / 2 - lower_idx) / freq_power.shape[0] * sample_rate
    )

    if debug:
        max_power_idx = int(cupy.asnumpy(freq_power_normalized.argmax(axis=0)))

        print(
            f"\nEstimated frequency edges {freq_lower_edge=}, {freq_upper_edge=}, {lower_idx=}, {upper_idx=}\n"
        )
        ###
        # Figure 1
        ###

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(
            cupy.asnumpy(10 * cupy.log10(cupy.fft.fftshift(Sxx, axes=0))),
            origin="lower",
        )
        axs[0].axhline(y=upper_idx, color="r", linestyle="-")
        axs[0].axhline(y=lower_idx, color="r", linestyle="-")
        # axs[0].pcolormesh(cupy.asnumpy(t), cupy.asnumpy(cupy.fft.fftshift(f)), cupy.asnumpy(cupy.fft.fftshift(Sxx, axes=0)))
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        axs[1].imshow(
            np.tile(
                np.expand_dims(
                    cupy.asnumpy(cupy.median(cupy.fft.fftshift(Sxx, axes=0), axis=1)), 1
                ),
                25,
            ),
            origin="lower",
        )
        # axs[1].axhline(y = upper_idx, color = 'r', linestyle = '-')
        # axs[1].axhline(y = lower_idx, color = 'g', linestyle = '-')

        axs[2].imshow(
            np.tile(np.expand_dims(cupy.asnumpy(freq_power_normalized), 1), 25),
            origin="lower",
        )
        axs[2].axhline(y=max_power_idx, color="pink", linestyle="-")
        axs[2].axhline(y=upper_idx, color="r", linestyle="-")
        axs[2].axhline(y=lower_idx, color="r", linestyle="-")
        plt.show()

        # ###
        # # Figure 2
        # ###
        # start_time = time.time()
        # plt.figure()
        # sns.histplot(cupy.asnumpy(freq_power), kde=True)
        # plt.xlabel("power")
        # plt.title(f"Occupied Bandwidth Signal Power Histogram & Density")
        # plt.show()
        # print(f"Plot time = {time.time()-start_time}")

        # ###
        # # Figure 3
        # ###
        # start_time = time.time()
        # plt.figure()
        # sns.histplot(cupy.asnumpy(freq_power_normalized), kde=True)
        # plt.xlabel("power")
        # plt.title(f"Normalized Occupied Bandwidth Signal Power Histogram & Density")
        # plt.show()
        # print(f"Plot time = {time.time()-start_time}")

        # ###
        # # Figure 4
        # ###
        # start_time = time.time()
        # plt.figure()
        # sns.histplot(cupy.asnumpy(10 * cupy.log10(freq_power)), kde=True)
        # plt.xlabel("dB")
        # plt.title(f"10*cupy.log10(freq_power)")
        # plt.show()
        # print(f"Plot time = {time.time()-start_time}")

        # ###
        # # Figure 5
        # ###
        # start_time = time.time()
        # plt.figure()
        # sns.histplot(cupy.asnumpy(10 * cupy.log10(freq_power_normalized)), kde=True)
        # plt.xlabel("dB")
        # plt.title(f"10*cupy.log10(freq_power_normalized)")
        # plt.show()
        # print(f"Plot time = {time.time()-start_time}")

        # ###
        # # Figure 6
        # ###
        # start_time = time.time()
        # plt.figure()
        # sns.histplot(
        #     cupy.asnumpy(10 * cupy.log10(cupy.fft.fftshift(Sxx, axes=0))).flatten(),
        #     kde=True,
        # )
        # plt.xlabel("dB")
        # plt.title(f"10*cupy.log10(cupy.fft.fftshift(Sxx, axes=0))")
        # plt.show()
        # print(f"Plot time = {time.time()-start_time}")

        # ###
        # # Figure 7
        # ###
        # start_time = time.time()
        # plt.figure()
        # plt.plot(cupy.asnumpy(10 * cupy.log10(freq_power)))
        # plt.xlabel("frequency")
        # plt.ylabel("power")
        # plt.title(f"10*cupy.log10(freq_power)")
        # plt.show()
        # print(f"Plot time = {time.time()-start_time}")

        ###
        # Figure 8
        ###
        # start_time = time.time()
        # plt.figure()
        # plt.plot(cupy.asnumpy(10 * cupy.log10(freq_power_normalized)))
        # plt.xlabel("frequency")
        # plt.ylabel("power")
        # plt.title(f"10*cupy.log10(freq_power_normalized)")
        # plt.show()
        # print(f"Plot time = {time.time()-start_time}")

        # fit a Gaussian Mixture Model with two components
        start_time = time.time()
        clf = mixture.GaussianMixture(n_components=2)
        predictions = clf.fit_predict(
            cupy.asnumpy(10 * cupy.log10(freq_power_normalized)).reshape(-1, 1)
        )
        # predictions = clf.fit_predict(cupy.asnumpy(freq_power_normalized).reshape(-1, 1))
        print(f"Gaussian mixture model time = {time.time()-start_time}")
        print(f"{clf.weights_=}")
        print(f"{clf.means_=}")
        print(f"{clf.covariances_=}")
        print(f"{clf.converged_=}")

        ###
        # Figure 9
        ###
        start_time = time.time()
        plt.figure()
        plt.plot(predictions)
        plt.xlabel("")
        plt.ylabel("gaussian mixture labels")
        plt.title(f"")
        plt.show()
        print(f"Plot time = {time.time()-start_time}")

        ####
        ####
        # signal_predictions = np.zeros(len(predictions))
        # signal_predictions[np.where(predictions == np.argmax(clf.means_))] = 1

        # signal_predictions_idx = (
        #     np.ediff1d(np.r_[0, signal_predictions == 1, 0]).nonzero()[0].reshape(-1, 2)
        # )  # gets indices where signal power above threshold

        # freq_bounds = signal_predictions_idx[
        #     np.argmax(
        #         np.abs(signal_predictions_idx[:, 0] - signal_predictions_idx[:, 1])
        #     )
        # ]
        # print(f"{signal_predictions_idx.shape=}")
        # print(f"{signal_predictions_idx=}")
        # plt.figure()
        # plt.imshow(cupy.asnumpy(10 * cupy.log10(cupy.fft.fftshift(Sxx, axes=0))), origin='lower')
        # plt.axhline(y=freq_bounds[0], color="r", linestyle="-")
        # plt.axhline(y=freq_bounds[1], color="r", linestyle="-")
        # plt.show()

    return freq_lower_edge, freq_upper_edge


def moving_average(complex_iq, avg_window_len):
    return (
        np.convolve(np.abs(complex_iq) ** 2, np.ones(avg_window_len), "valid")
        / avg_window_len
    )


def power_squelch(iq_samples, threshold, avg_window_len):
    avg_pwr = moving_average(iq_samples, avg_window_len)
    avg_pwr_db = 10 * np.log10(avg_pwr)

    good_samples = np.zeros(len(iq_samples))
    good_samples[np.where(avg_pwr_db > threshold)] = 1

    idx = (
        np.ediff1d(np.r_[0, good_samples == 1, 0]).nonzero()[0].reshape(-1, 2)
    )  # gets indices where signal power above threshold

    return idx


def reset_annotations(data_obj):
    data_obj.sigmf_obj._metadata[data_obj.sigmf_obj.ANNOTATION_KEY] = []
    data_obj.sigmf_obj.tofile(data_obj.sigmf_meta_filename, skip_validate=True)
    print(f"Resetting annotations in {data_obj.sigmf_meta_filename}")


# MAD estimator
def median_absolute_deviation(series):
    mad = 1.4826 * np.median(np.abs(series - np.median(series)))
    # sci_mad = scipy.stats.median_abs_deviation(series, scale="normal")
    return np.median(series) + 6 * mad


def debug_plot(
    avg_pwr_db,
    mad,
    threshold_db,
    debug_duration,
    data_obj,
    guess_threshold_old,
    force_threshold_db,
    n_components=None,
):
    n_components = n_components if n_components else 3

    print(f"Using threshold = {threshold_db} dB")

    ####
    # Figure 1
    ###
    plt.figure()
    db_plot = avg_pwr_db[
        int(0 * data_obj.metadata["global"]["core:sample_rate"]) : int(
            debug_duration * data_obj.metadata["global"]["core:sample_rate"]
        )
    ]
    plt.plot(
        np.arange(len(db_plot)) / data_obj.metadata["global"]["core:sample_rate"],
        db_plot,
    )
    plt.axhline(y=guess_threshold_old, color="g", linestyle="-", label="old threshold")
    plt.axhline(y=np.mean(avg_pwr_db), color="r", linestyle="-", label="average")
    # plt.axhline(
    #     y=mad,
    #     color="b",
    #     linestyle="-",
    #     label="median absolute deviation threshold",
    # )
    if force_threshold_db:
        plt.axhline(
            y=force_threshold_db,
            color="yellow",
            linestyle="-",
            label="force threshold db",
        )
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.ylabel("dB")
    plt.xlabel("time (seconds)")
    plt.title("Signal Power")
    plt.show()

    ###
    # Figure 2
    ###
    db_plot = avg_pwr_db[
        int(0 * data_obj.metadata["global"]["core:sample_rate"]) : int(
            debug_duration * data_obj.metadata["global"]["core:sample_rate"]
        )
    ]
    start_time = time.time()
    plt.figure()
    sns.histplot(db_plot, kde=True)
    plt.xlabel("dB")
    plt.title(f"Signal Power Histogram & Density ({debug_duration} seconds)")
    plt.show()
    print(f"Plot time = {time.time()-start_time}")

    # fit a Gaussian Mixture Model with two components
    start_time = time.time()
    clf = mixture.GaussianMixture(n_components=n_components)
    clf.fit(db_plot.reshape(-1, 1))
    print(f"Gaussian mixture model time = {time.time()-start_time}")
    print(f"{clf.weights_=}")
    print(f"{clf.means_=}")
    print(f"{clf.covariances_=}")
    print(f"{clf.converged_=}")

    ###
    # Figure 3
    ###
    db_plot = avg_pwr_db
    start_time = time.time()
    plt.figure()
    sns.histplot(db_plot, kde=True)
    plt.xlabel("dB")
    plt.title(f"Signal Power Histogram & Density")
    plt.show()
    print(f"Plot time = {time.time()-start_time}")

    # fit a Gaussian Mixture Model with two components
    start_time = time.time()
    clf = mixture.GaussianMixture(n_components=n_components)
    clf.fit(db_plot.reshape(-1, 1))
    print(f"Gaussian mixture model time = {time.time()-start_time}")
    print(f"{clf.weights_=}")
    print(f"{clf.means_=}")
    print(f"{clf.covariances_=}")
    print(f"{clf.converged_=}")

    ###
    # Figure 4
    ###
    plt.figure()
    db_plot = avg_pwr_db[
        int(0 * data_obj.metadata["global"]["core:sample_rate"]) : int(
            debug_duration * data_obj.metadata["global"]["core:sample_rate"]
        )
    ]
    plt.plot(
        np.arange(len(db_plot)) / data_obj.metadata["global"]["core:sample_rate"],
        db_plot,
    )
    plt.axhline(y=guess_threshold_old, color="g", linestyle="-", label="old threshold")
    plt.axhline(y=np.mean(avg_pwr_db), color="r", linestyle="-", label="average")
    # plt.axhline(
    #     y=mad,
    #     color="b",
    #     linestyle="-",
    #     label="median absolute deviation threshold",
    # )
    plt.axhline(
        y=np.min(clf.means_)
        + 3 * np.sqrt(clf.covariances_[np.argmin(clf.means_)].squeeze()),
        color="yellow",
        linestyle="-",
        label="gaussian mixture model estimate",
    )
    if force_threshold_db:
        plt.axhline(
            y=force_threshold_db,
            color="yellow",
            linestyle="-",
            label="force threshold db",
        )
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.ylabel("dB")
    plt.xlabel("time (seconds)")
    plt.title("Signal Power")
    plt.show()

    ###
    # Figure 5
    ###
    plt.figure()

    db_plot = avg_pwr_db
    plt.plot(
        np.arange(len(db_plot)) / data_obj.metadata["global"]["core:sample_rate"],
        db_plot,
    )
    plt.axhline(y=guess_threshold_old, color="g", linestyle="-", label="old threshold")
    plt.axhline(y=np.mean(avg_pwr_db), color="r", linestyle="-", label="average")
    # plt.axhline(
    #     y=mad,
    #     color="b",
    #     linestyle="-",
    #     label="median absolute deviation threshold",
    # )
    plt.axhline(
        y=np.min(clf.means_)
        + 3 * np.sqrt(clf.covariances_[np.argmin(clf.means_)].squeeze()),
        color="yellow",
        linestyle="-",
        label="gaussian mixture model estimate",
    )
    if force_threshold_db:
        plt.axhline(
            y=force_threshold_db,
            color="yellow",
            linestyle="-",
            label="force threshold db",
        )
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.ylabel("dB")
    plt.xlabel("time (seconds)")
    plt.title("Signal Power")
    plt.show()


def reset_predictions_sigmf(dataset):
    data_files = set([dataset.index[i][1].absolute_path for i in range(len(dataset))])
    for f in data_files:
        data_obj = data_class.Data(f)
        prediction_meta_path = Path(
            Path(data_obj.sigmf_meta_filename).parent,
            f"prediction_{Path(data_obj.sigmf_meta_filename).name}",
        )
        data_obj.sigmf_obj._metadata[data_obj.sigmf_obj.ANNOTATION_KEY] = []
        data_obj.sigmf_obj.tofile(prediction_meta_path, skip_validate=True)
        print(f"Reset annotations in {prediction_meta_path}")

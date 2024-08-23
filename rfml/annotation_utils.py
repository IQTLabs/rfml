# Tools for annotating RF data

from collections.abc import Iterable
import cupy
from cupyx.scipy.signal import spectrogram as cupyx_spectrogram
from cupyx.scipy.ndimage import gaussian_filter as cupyx_gaussian_filter

from rfml.spectrogram import *

import rfml.data as data_class
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from tqdm import tqdm


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


def annotate_power_squelch(
    data_obj,
    threshold,
    avg_window_len,
    label=None,
    skip_validate=False,
    spectral_energy_threshold=False,
    dry_run=False,
    min_annotation_length=400,
    min_bandwidth=None,
    max_bandwidth=None,
    overwrite=True,
    max_annotations=None,
    dc_block=False,
    verbose=False,
    n_seek_samples=None,
    n_samples=None,
    set_bandwidth=None,
):
    iq_samples = data_obj.get_samples(
        n_seek_samples=n_seek_samples, n_samples=n_samples
    )
    idx = power_squelch(iq_samples, threshold=threshold, avg_window_len=avg_window_len)

    if overwrite:
        data_obj.sigmf_obj._metadata[data_obj.sigmf_obj.ANNOTATION_KEY] = []
    for start, stop in tqdm(idx[:max_annotations]):
        # print(f"{start=}, {stop=}  {max_annotations=}")
        start, stop = int(start), int(stop)
        if min_annotation_length and (stop - start < min_annotation_length):
            continue

        if isinstance(spectral_energy_threshold, bool) and spectral_energy_threshold:
            spectral_energy_threshold = 0.94

        if set_bandwidth:
            freq_lower_edge = (
                data_obj.metadata["captures"][0]["core:frequency"] - set_bandwidth / 2
            )
            freq_upper_edge = (
                data_obj.metadata["captures"][0]["core:frequency"] + set_bandwidth / 2
            )

        elif isinstance(spectral_energy_threshold, float):
            freq_lower_edge, freq_upper_edge = get_occupied_bandwidth(
                iq_samples[start:stop],
                data_obj.metadata["global"]["core:sample_rate"],
                data_obj.metadata["captures"][0]["core:frequency"],
                spectral_energy_threshold=spectral_energy_threshold,
                dc_block=dc_block,
                verbose=verbose,
            )
            bandwidth = freq_upper_edge - freq_lower_edge
            if min_bandwidth and bandwidth < min_bandwidth:
                if verbose:
                    print(
                        f"min_bandwidth - Skipping, {label}, {start=}, {stop=}, {bandwidth=}, {freq_upper_edge=}, {freq_lower_edge=}"
                    )
                # print(f"Skipping, {label}, {start=}, {stop=}, {bandwidth=}, {freq_upper_edge=}, {freq_lower_edge=}")
                continue
            if max_bandwidth and bandwidth > max_bandwidth:
                if verbose:
                    print(
                        f"max_bandwidth - Skipping, {label}, {start=}, {stop=}, {bandwidth=}, {freq_upper_edge=}, {freq_lower_edge=}"
                    )
                continue

        else:
            freq_lower_edge = (
                data_obj.metadata["captures"][0]["core:frequency"]
                - data_obj.metadata["global"]["core:sample_rate"] / 2
            )
            freq_upper_edge = (
                data_obj.metadata["captures"][0]["core:frequency"]
                + data_obj.metadata["global"]["core:sample_rate"] / 2
            )
        metadata = {
            "core:freq_lower_edge": freq_lower_edge,
            "core:freq_upper_edge": freq_upper_edge,
        }
        if label:
            metadata["core:label"] = label

        data_obj.sigmf_obj.add_annotation(
            n_seek_samples + start, length=stop - start, metadata=metadata
        )

    # print(f"{data_obj.sigmf_obj=}")

    if not dry_run:
        data_obj.sigmf_obj.tofile(
            data_obj.sigmf_meta_filename, skip_validate=skip_validate
        )
        print(
            f"Writing {len(data_obj.sigmf_obj._metadata[data_obj.sigmf_obj.ANNOTATION_KEY])} annotations to {data_obj.sigmf_meta_filename}"
        )


def annotate(
    filename,
    label,
    avg_window_len,
    avg_duration=-1,
    debug=False,
    dry_run=False,
    min_annotation_length=400,
    spectral_energy_threshold=True,
    force_threshold_db=None,
    overwrite=True,
    min_bandwidth=None,
    max_bandwidth=None,
    max_annotations=None,
    dc_block=None,
    verbose=False,
    time_start_stop=None,
    set_bandwidth=None,
):

    data_obj = data_class.Data(filename)

    sample_rate = data_obj.metadata["global"]["core:sample_rate"]

    if isinstance(time_start_stop, int) and time_start_stop > 0:
        n_seek_samples = int(sample_rate * time_start_stop)
        n_samples = -1
    elif isinstance(time_start_stop, Iterable):
        if len(time_start_stop) != 2 or time_start_stop[1] < time_start_stop[0]:
            raise ValueError

        n_seek_samples = int(sample_rate * time_start_stop[0])
        n_samples = int(sample_rate * (time_start_stop[1] - time_start_stop[0]))
    else:
        n_seek_samples = 0
        n_samples = -1

    if force_threshold_db:
        threshold_db = force_threshold_db
    else:
        # use a seconds worth of data to calculate threshold
        if avg_duration > -1:
            iq_samples = data_obj.get_samples(
                n_seek_samples=n_seek_samples, n_samples=int(sample_rate * avg_duration)
            )
            if iq_samples is None:
                iq_samples = data_obj.get_samples(
                    n_seek_samples=n_seek_samples, n_samples=n_samples
                )
        else:
            iq_samples = data_obj.get_samples(
                n_seek_samples=n_seek_samples, n_samples=n_samples
            )

        avg_pwr = moving_average(iq_samples, avg_window_len)
        avg_pwr_db = 10 * np.log10(avg_pwr)
        del avg_pwr
        del iq_samples

        # current threshold in custom_handler
        guess_threshold_old = (np.max(avg_pwr_db) + np.mean(avg_pwr_db)) / 2

        # MAD estimator
        def median_absolute_deviation(series):
            mad = 1.4826 * np.median(np.abs(series - np.median(series)))
            # sci_mad = scipy.stats.median_abs_deviation(series, scale="normal")
            return np.median(series) + 6 * mad

        mad = median_absolute_deviation(avg_pwr_db)

        threshold_db = mad

        if debug:
            print(f"{np.max(avg_pwr_db)=}")
            print(f"{np.mean(avg_pwr_db)=}")
            print(f"median absolute deviation threshold = {mad}")
            print(f"using threshold = {threshold_db}")
            # print(f"{len(avg_pwr_db)=}")

            plt.figure()
            db_plot = avg_pwr_db[int(0 * 20.48e6) : int(avg_duration * 20.48e6)]
            plt.plot(
                np.arange(len(db_plot))
                / data_obj.metadata["global"]["core:sample_rate"],
                db_plot,
            )
            plt.axhline(
                y=guess_threshold_old, color="g", linestyle="-", label="old threshold"
            )
            plt.axhline(
                y=np.mean(avg_pwr_db), color="r", linestyle="-", label="average"
            )
            plt.axhline(
                y=mad,
                color="b",
                linestyle="-",
                label="median absolute deviation threshold",
            )
            if force_threshold_db:
                plt.axhline(
                    y=force_threshold_db,
                    color="yellow",
                    linestyle="-",
                    label="force threshold db",
                )
            plt.legend(loc="upper left")
            plt.ylabel("dB")
            plt.xlabel("time (seconds)")
            plt.title("Signal Power")
            plt.show()
    print(f"Using dB threshold = {threshold_db} for detecting signals to annotate")
    annotate_power_squelch(
        data_obj,
        threshold_db,
        avg_window_len,
        label=label,
        skip_validate=True,
        spectral_energy_threshold=spectral_energy_threshold,
        min_bandwidth=min_bandwidth,
        max_bandwidth=max_bandwidth,
        dry_run=dry_run,
        min_annotation_length=min_annotation_length,
        overwrite=overwrite,
        max_annotations=max_annotations,
        dc_block=dc_block,
        verbose=verbose,
        n_seek_samples=n_seek_samples,
        n_samples=n_samples,
        set_bandwidth=set_bandwidth,
    )


def get_occupied_bandwidth(
    samples,
    sample_rate,
    center_frequency,
    spectral_energy_threshold=None,
    dc_block=False,
    verbose=False,
):

    if not spectral_energy_threshold:
        spectral_energy_threshold = 0.94

    f, t, Sxx = cupyx_spectrogram(
        samples, fs=sample_rate, return_onesided=False, scaling="spectrum"
    )

    # freq_power = cupy.asnumpy(cupy.fft.fftshift(Sxx, axes=0))

    freq_power = cupy.median(cupy.fft.fftshift(Sxx, axes=0), axis=1)

    # lessen DC
    if dc_block:
        dc_start = int(len(freq_power) / 2) - 1
        dc_stop = int(len(freq_power) / 2) + 2
        freq_power[dc_start:dc_stop] /= 2

    freq_power_normalized = freq_power / freq_power.sum(axis=0)

    max_power_idx = int(cupy.asnumpy(freq_power_normalized.argmax(axis=0)))
    lower_idx = max_power_idx
    upper_idx = max_power_idx
    # print(f"{max_power_idx=}")
    while True:
        # print(f"{lower_idx=}, {upper_idx=}, {freq_power_normalized[lower_idx]=}, {freq_power_normalized[upper_idx]=}, {spectral_energy_threshold=}")
        if upper_idx == freq_power_normalized.shape[0] - 1:
            lower_idx -= 1
        elif lower_idx == 0:
            upper_idx += 1
        elif (
            freq_power_normalized[lower_idx - 1] > freq_power_normalized[upper_idx + 1]
        ):
            lower_idx -= 1
        else:
            upper_idx += 1

        if (
            freq_power_normalized[lower_idx : upper_idx + 1].sum()
            >= spectral_energy_threshold
        ):
            break

        if lower_idx == 0 and upper_idx == freq_power_normalized.shape[0] - 1:
            print(
                f"Could not find spectral energy threshold - max was: {freq_power_normalized[lower_idx:upper_idx].sum()}"
            )
            break

    freq_upper_edge = (
        center_frequency
        - (freq_power.shape[0] / 2 - upper_idx) / freq_power.shape[0] * sample_rate
    )
    freq_lower_edge = (
        center_frequency
        - (freq_power.shape[0] / 2 - lower_idx) / freq_power.shape[0] * sample_rate
    )

    if verbose:
        # print(f"{freq_power_normalized[lower_idx]=}")
        # print(f"{freq_power_normalized[upper_idx]=}")
        # print(f"{freq_power_normalized=}")
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(cupy.asnumpy(cupy.fft.fftshift(Sxx, axes=0)))
        axs[0].axhline(y=upper_idx, color="r", linestyle="-")
        axs[0].axhline(y=lower_idx, color="g", linestyle="-")
        # axs[0].pcolormesh(cupy.asnumpy(t), cupy.asnumpy(cupy.fft.fftshift(f)), cupy.asnumpy(cupy.fft.fftshift(Sxx, axes=0)))
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        axs[1].imshow(
            np.tile(
                np.expand_dims(
                    cupy.asnumpy(cupy.median(cupy.fft.fftshift(Sxx, axes=0), axis=1)), 1
                ),
                25,
            )
        )
        # axs[1].axhline(y = upper_idx, color = 'r', linestyle = '-')
        # axs[1].axhline(y = lower_idx, color = 'g', linestyle = '-')

        axs[2].imshow(
            np.tile(np.expand_dims(cupy.asnumpy(freq_power_normalized), 1), 25)
        )
        axs[2].axhline(y=max_power_idx, color="orange", linestyle="-")
        axs[2].axhline(y=upper_idx, color="r", linestyle="-")
        axs[2].axhline(y=lower_idx, color="g", linestyle="-")
        plt.show()
    # exit()
    return freq_lower_edge, freq_upper_edge


def get_occupied_bandwidth_backup(samples, sample_rate, center_frequency):

    # spectrogram_data, spectrogram_raw = spectrogram(
    #     samples,
    #     sample_rate,
    #     256,
    #     0,
    # )
    # spectrogram_color = spectrogram_cmap(spectrogram_data, plt.get_cmap("viridis"))

    # plt.figure()
    # plt.imshow(spectrogram_color)
    # plt.show()

    # print(f"{samples.shape=}")
    # print(f"{samples=}")

    f, t, Sxx = cupyx_spectrogram(
        samples, fs=sample_rate, return_onesided=False, scaling="spectrum"
    )

    freq_power = cupy.asnumpy(cupy.fft.fftshift(Sxx, axes=0))
    # print(f"{freq_power.shape=}")

    # print(f"{freq_power.argmax(axis=0).shape=}")
    # print(f"{freq_power.argmax(axis=0)=}")

    # freq_power = cupy.asnumpy(cupyx_gaussian_filter(cupy.fft.fftshift(Sxx, axes=0), sigma=1))
    freq_power = cupyx_gaussian_filter(cupy.fft.fftshift(Sxx, axes=0), sigma=1)

    # plt.figure()
    # plt.pcolormesh(cupy.asnumpy(t), cupy.asnumpy(cupy.fft.fftshift(f)), cupy.asnumpy(cupy.fft.fftshift(Sxx, axes=0)))
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    # plt.figure()
    # plt.pcolormesh(cupy.asnumpy(t), cupy.asnumpy(cupy.fft.fftshift(f)), cupy.asnumpy(freq_power))
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()

    freq_power_normalized = freq_power / freq_power.sum(axis=0)

    # print(f"{freq_power_normalized.shape=}")
    # print(f"{freq_power_normalized.argmax(axis=0).shape=}")
    # print(f"{freq_power_normalized.argmax(axis=0)=}")
    bounds = []
    for i, max_power_idx in enumerate(freq_power_normalized.argmax(axis=0)):
        max_power_idx = int(cupy.asnumpy(max_power_idx))
        # print(f"{i=}, {max_power_idx=}")
        lower_idx = max_power_idx
        upper_idx = max_power_idx
        while True:

            if upper_idx == freq_power_normalized.shape[0] - 1:
                lower_idx -= 1
            elif lower_idx == 0:
                upper_idx += 1
            elif (
                freq_power_normalized[lower_idx, i]
                > freq_power_normalized[upper_idx, i]
            ):
                lower_idx -= 1
            else:
                upper_idx += 1

            # print(f"{lower_idx=}, {upper_idx=}")
            # print(f"{freq_power_normalized[lower_idx:upper_idx, i].sum()=}")
            if freq_power_normalized[lower_idx:upper_idx, i].sum() >= 0.94:
                break

        bounds.append([lower_idx, upper_idx])
    bounds = np.array(bounds)

    plt.figure()
    plt.imshow(cupy.asnumpy(cupy.fft.fftshift(Sxx, axes=0)))
    plt.plot(cupy.asnumpy(freq_power.argmax(axis=0)))
    plt.plot(bounds[:, 0])
    plt.plot(bounds[:, 1])
    plt.axhline(y=np.median(bounds[:, 0]), color="r", linestyle="-")
    plt.axhline(y=np.median(bounds[:, 1]), color="b", linestyle="-")
    plt.show()

    freq_lower_edge = (
        center_frequency
        + (freq_power.shape[0] / 2 - np.median(bounds[:, 1]))
        / freq_power.shape[0]
        * sample_rate
    )
    freq_upper_edge = (
        center_frequency
        + (freq_power.shape[0] / 2 - np.median(bounds[:, 0]))
        / freq_power.shape[0]
        * sample_rate
    )

    # print(f"{freq_lower_edge=}")
    # print(f"{freq_upper_edge=}")
    print(f"estimated bandwidth = {freq_upper_edge-freq_lower_edge}")
    return freq_lower_edge, freq_upper_edge


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

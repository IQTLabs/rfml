# Spectrogram helper scripts

import numpy as np
from scipy import signal


def spectrogram(
    samples, sample_rate, nfft, noverlap, min_freq=None, max_freq=None, freq_center=None
):
    # Convert samples into spectrogram
    freq_bins, t_bins, spectrogram = signal.spectrogram(
        samples,
        sample_rate,
        window=signal.windows.hann(int(nfft), sym=True),
        nperseg=nfft,
        noverlap=noverlap,
        detrend="constant",
        return_onesided=False,
    )
    # FFT shift
    freq_bins = np.fft.fftshift(freq_bins)
    spectrogram = np.fft.fftshift(spectrogram, axes=0)
    # Transpose spectrogram
    spectrogram = spectrogram.T
    spectrogram_raw = spectrogram.copy()
    # dB scale spectrogram
    spectrogram = 10 * np.log10(spectrogram)

    # Normalize spectrogram
    spectrogram_normalized = (spectrogram - np.min(spectrogram)) / (
        np.max(spectrogram) - np.min(spectrogram)
    )  # (spectrogram - db_min) / (db_max - db_min)

    spectrogram_data = spectrogram_normalized

    if min_freq is not None and max_freq is not None and freq_center is not None:
        if fft_count is None:
            fft_count = len(t_bins)
        spectrogram_data, max_idx, freq_resolution = prepare_custom_spectrogram(
            min_freq, max_freq, sample_rate, nfft, fft_count, noverlap
        )
        idx = np.array(
            [
                round((item - min_freq) / freq_resolution)
                for item in freq_bins + freq_center
            ]
        ).astype(int)
        spectrogram_data[
            : spectrogram_normalized.shape[0],
            idx[np.flatnonzero((idx >= 0) & (idx <= max_idx))],
        ] = spectrogram_normalized[:, np.flatnonzero((idx >= 0) & (idx <= max_idx))]

    return spectrogram_data, spectrogram_raw


def prepare_custom_spectrogram(
    min_freq, max_freq, sample_rate, nfft, fft_count, noverlap
):
    freq_resolution = sample_rate / nfft
    max_idx = round((max_freq - min_freq) / freq_resolution)
    total_time = (nfft * fft_count) / sample_rate
    expected_time_bins = int((nfft * fft_count) / (nfft - noverlap))
    X, Y = np.meshgrid(
        np.linspace(
            min_freq,
            max_freq,
            int((max_freq - min_freq) / freq_resolution + 1),
        ),
        np.linspace(0, total_time, expected_time_bins),
    )
    spectrogram_array = np.empty(X.shape)
    spectrogram_array.fill(np.nan)

    return spectrogram_array, max_idx, freq_resolution


def spectrogram_cmap(spectrogram_data, cmap):
    # Spectrogram color transforms
    # spectrogram_color = cv2.resize(cmap(spectrogram_data)[:,:,:3], dsize=(1640, 640), interpolation=cv2.INTER_CUBIC)[:,:,::-1]
    spectrogram_color = cmap(spectrogram_data)[:, :, :3]  # remove alpha dimension
    spectrogram_color = spectrogram_color[::-1, :, :]  # flip vertically
    spectrogram_color *= 255
    spectrogram_color = spectrogram_color.astype(int)
    spectrogram_color = np.ascontiguousarray(spectrogram_color, dtype=np.uint8)
    return spectrogram_color

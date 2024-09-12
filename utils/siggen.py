#!/usr/bin/env python3

import os
import random
import sys
import tempfile
from argparse import ArgumentParser
from subprocess import Popen, PIPE
import sigmf
from gnuradio import analog
from gnuradio import blocks
from gnuradio import filter as grfilter
from gnuradio.fft import window
from gnuradio import gr
from scipy.io import wavfile


CENTER_FREQ = 100e6


class fmsiggen(gr.top_block):

    def __init__(
        self,
        wav_file,
        sample_file,
        samp_rate,
        audio_samp_rate,
        audio_gain,
        audio_interp=4,
    ):
        gr.top_block.__init__(self, "fmsiggen", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate
        self.audio_samp_rate = audio_samp_rate
        self.audio_interp = audio_interp
        self.audio_gain = audio_gain

        ##################################################
        # Blocks
        ##################################################

        self.rational_resampler_xxx_0 = grfilter.rational_resampler_ccc(
            interpolation=samp_rate,
            decimation=(audio_samp_rate * audio_interp),
            taps=[],
            fractional_bw=0,
        )
        self.blocks_wavfile_source_0 = blocks.wavfile_source(wav_file, False)
        self.blocks_sigmf_sink_minimal_0 = blocks.sigmf_sink_minimal(
            item_size=gr.sizeof_gr_complex,
            filename=sample_file,
            sample_rate=samp_rate,
            center_freq=CENTER_FREQ,
            author="",
            description="",
            hw_info="",
            is_complex=True,
        )
        self.blocks_multiply_const_xx_0 = blocks.multiply_const_ff(audio_gain, 1)
        self.analog_nbfm_tx_0 = analog.nbfm_tx(
            audio_rate=audio_samp_rate,
            quad_rate=(audio_samp_rate * audio_interp),
            tau=(75e-6),
            max_dev=5e3,
            fh=(-1.0),
        )

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_nbfm_tx_0, 0), (self.rational_resampler_xxx_0, 0))
        self.connect((self.blocks_multiply_const_xx_0, 0), (self.analog_nbfm_tx_0, 0))
        self.connect(
            (self.blocks_wavfile_source_0, 0), (self.blocks_multiply_const_xx_0, 0)
        )
        self.connect(
            (self.rational_resampler_xxx_0, 0), (self.blocks_sigmf_sink_minimal_0, 0)
        )

    def bw(self):
        return self.audio_samp_rate


class amsiggen(gr.top_block):

    def __init__(
        self,
        wav_file,
        sample_file,
        samp_rate,
        audio_samp_rate,
        audio_gain,
        audio_interp=4,
    ):
        gr.top_block.__init__(self, "amsiggen", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate
        self.audio_samp_rate = audio_samp_rate
        self.audio_interp = audio_interp
        self.audio_gain = audio_gain

        ##################################################
        # Blocks
        ##################################################

        self.rational_resampler_xxx_2 = grfilter.rational_resampler_ccc(
            interpolation=samp_rate,
            decimation=audio_samp_rate,
            taps=[],
            fractional_bw=0,
        )
        self.low_pass_filter_1 = grfilter.interp_fir_filter_ccf(
            1,
            grfilter.firdes.low_pass(
                0.5, audio_samp_rate, 5000, 400, window.WIN_HAMMING, 6.76
            ),
        )
        self.blocks_wavfile_source_0 = blocks.wavfile_source(wav_file, False)
        self.blocks_multiply_xx_2 = blocks.multiply_vcc(1)
        self.blocks_float_to_complex_0 = blocks.float_to_complex(1)
        self.blocks_add_const_vxx_0 = blocks.add_const_cc(0.5)
        self.analog_sig_source_x_2 = analog.sig_source_c(
            audio_samp_rate, analog.GR_COS_WAVE, 0, 1, 0, 0
        )
        self.analog_const_source_x_0 = analog.sig_source_f(
            0, analog.GR_CONST_WAVE, 0, 0, 0
        )
        self.blocks_sigmf_sink_minimal_0 = blocks.sigmf_sink_minimal(
            item_size=gr.sizeof_gr_complex,
            filename=sample_file,
            sample_rate=samp_rate,
            center_freq=CENTER_FREQ,
            author="",
            description="",
            hw_info="",
            is_complex=True,
        )

        ##################################################
        # Connections
        ##################################################
        self.connect(
            (self.analog_const_source_x_0, 0), (self.blocks_float_to_complex_0, 1)
        )
        self.connect((self.analog_sig_source_x_2, 0), (self.blocks_multiply_xx_2, 1))
        self.connect((self.blocks_add_const_vxx_0, 0), (self.blocks_multiply_xx_2, 0))
        self.connect((self.blocks_float_to_complex_0, 0), (self.low_pass_filter_1, 0))
        self.connect((self.blocks_multiply_xx_2, 0), (self.rational_resampler_xxx_2, 0))
        self.connect(
            (self.blocks_wavfile_source_0, 0), (self.blocks_float_to_complex_0, 0)
        )
        self.connect((self.low_pass_filter_1, 0), (self.blocks_add_const_vxx_0, 0))
        self.connect(
            (self.rational_resampler_xxx_2, 0), (self.blocks_sigmf_sink_minimal_0, 0)
        )

    def bw(self):
        return self.audio_samp_rate


class noisesiggen(gr.top_block):

    def __init__(
        self,
        wav_file,
        sample_file,
        samp_rate,
        audio_samp_rate,
        audio_gain,
        audio_interp=4,
    ):
        gr.top_block.__init__(self, "noisesiggen", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate
        self.audio_samp_rate = audio_samp_rate
        self.audio_interp = audio_interp
        self.audio_gain = audio_gain

        ##################################################
        # Blocks
        ##################################################

        self.rational_resampler_xxx_2 = grfilter.rational_resampler_ccc(
            interpolation=samp_rate,
            decimation=audio_samp_rate,
            taps=[],
            fractional_bw=0,
        )
        self.low_pass_filter_1 = grfilter.interp_fir_filter_ccf(
            1,
            grfilter.firdes.low_pass(
                0.5, audio_samp_rate, 5000, 400, window.WIN_HAMMING, 6.76
            ),
        )
        self.blocks_wavfile_source_0 = blocks.wavfile_source(wav_file, False)
        self.blocks_multiply_xx_2 = blocks.multiply_vcc(1)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_float_to_complex_0 = blocks.float_to_complex(1)
        self.blocks_add_const_vxx_0 = blocks.add_const_cc(0.5)
        self.analog_sig_source_x_2 = analog.sig_source_c(
            audio_samp_rate, analog.GR_COS_WAVE, 0, 1, 0, 0
        )
        self.analog_const_source_x_0 = analog.sig_source_f(
            0, analog.GR_CONST_WAVE, 0, 0, 0
        )
        self.blocks_sigmf_sink_minimal_0 = blocks.sigmf_sink_minimal(
            item_size=gr.sizeof_gr_complex,
            filename=sample_file,
            sample_rate=samp_rate,
            center_freq=CENTER_FREQ,
            author="",
            description="",
            hw_info="",
            is_complex=True,
        )
        self.analog_fastnoise_source_x_0 = analog.fastnoise_source_c(
            analog.GR_GAUSSIAN, 1, 0, 8192
        )

        ##################################################
        # Connections
        ##################################################
        self.connect(
            (self.analog_const_source_x_0, 0), (self.blocks_float_to_complex_0, 1)
        )
        self.connect((self.analog_sig_source_x_2, 0), (self.blocks_multiply_xx_2, 1))
        self.connect((self.blocks_add_const_vxx_0, 0), (self.blocks_multiply_xx_2, 0))
        self.connect((self.blocks_float_to_complex_0, 0), (self.low_pass_filter_1, 0))
        self.connect((self.blocks_multiply_xx_2, 0), (self.rational_resampler_xxx_2, 0))
        self.connect(
            (self.blocks_wavfile_source_0, 0), (self.blocks_float_to_complex_0, 0)
        )
        self.connect((self.low_pass_filter_1, 0), (self.blocks_add_const_vxx_0, 0))
        self.connect((self.rational_resampler_xxx_2, 0), (self.blocks_multiply_xx_0, 0))
        self.connect(
            (self.analog_fastnoise_source_x_0, 0), (self.blocks_multiply_xx_0, 1)
        )
        self.connect(
            (self.blocks_multiply_xx_0, 0), (self.blocks_sigmf_sink_minimal_0, 0)
        )

    def bw(self):
        return self.samp_rate


def run_siggen(
    siggen,
    int_count,
    sample_file,
    samp_rate,
    audio_samp_rate,
    audio_gain,
):
    numbers = [str(random.randint(0, 1000)) for _ in range(int_count)]
    if sample_file is None:
        sample_file = siggen
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_file = os.path.join(tmpdir, "test.wav")
        try:
            p = Popen(
                ["text2wave", "-F", str(audio_samp_rate), "-o", wav_file],
                stdout=PIPE,
                stdin=PIPE,
                stderr=PIPE,
                text=True,
            )
        except FileNotFoundError:
            print("error running text2wave: need festival installed")
            sys.exit(-1)
        print(f"writing {int_count} numbers")
        print(p.communicate(input=" ".join(numbers))[0])
        out_samp_rate, audio_data = wavfile.read(wav_file)
        audio_secs = audio_data.shape[0] / out_samp_rate
        print(f"{audio_secs} seconds of audio")
        siggen_cls = getattr(sys.modules[__name__], siggen + "siggen")
        print(f"using {siggen_cls}")
        tb = siggen_cls(wav_file, sample_file, samp_rate, audio_samp_rate, audio_gain)
        tb.start()
        tb.wait()
        sigmf_meta_filename = sample_file + ".sigmf-meta"
        sigmf_meta = sigmf.sigmffile.fromfile(sigmf_meta_filename, skip_checksum=True)
        upper_edge = CENTER_FREQ + tb.bw() / 2
        lower_edge = CENTER_FREQ - tb.bw() / 2
        sigmf_meta.add_annotation(
            0,
            sigmf_meta.sample_count,
            metadata={
                sigmf.SigMFFile.LABEL_KEY: siggen,
                sigmf.SigMFFile.FHI_KEY: upper_edge,
                sigmf.SigMFFile.FLO_KEY: lower_edge,
            },
        )
        sigmf_meta.tofile(sigmf_meta_filename, skip_validate=True)


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--siggen",
        dest="siggen",
        type=str,
        default="fmsiggen",
        help="signal generator class to use",
    )
    parser.add_argument(
        "--sample_file",
        dest="sample_file",
        type=str,
        default=None,
        help="base file name for sample file name output",
    )
    parser.add_argument(
        "--samp_rate",
        dest="samp_rate",
        type=int,
        default=int(20.48e6),
        help="sample rate",
    )
    parser.add_argument(
        "--int_count",
        dest="int_count",
        type=int,
        default=int(10),
        help="number of random integers to say",
    )
    parser.add_argument(
        "--audio_samp_rate",
        dest="audio_samp_rate",
        type=int,
        default=int(44100),
        help="audio sample rate",
    )
    parser.add_argument(
        "--audio_gain",
        dest="audio_gain",
        type=int,
        default=int(20),
        help="audio gain",
    )
    return parser


def main():
    options = argument_parser().parse_args()
    run_siggen(
        options.siggen,
        options.int_count,
        options.sample_file,
        options.samp_rate,
        options.audio_samp_rate,
        options.audio_gain,
    )


if __name__ == "__main__":
    main()

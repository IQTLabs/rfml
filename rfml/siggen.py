#!/usr/bin/env python3

import os
import random
import sys
import tempfile
from argparse import ArgumentParser
from subprocess import Popen, PIPE, STDOUT
from gnuradio import analog
from gnuradio import blocks
from gnuradio import filter as grfilter
from gnuradio import gr
from scipy.io import wavfile


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
            center_freq=100e6,
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


def run_siggen(
    siggen_cls, int_count, sample_file, samp_rate, audio_samp_rate, audio_gain
):
    numbers = [str(random.randint(0, 1000)) for _ in range(int_count)]
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
        tb = siggen_cls(wav_file, sample_file, samp_rate, audio_samp_rate, audio_gain)
        tb.start()
        tb.wait()


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--sample_file",
        dest="sample_file",
        type=str,
        default="siggen",
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
        fmsiggen,
        options.int_count,
        options.sample_file,
        options.samp_rate,
        options.audio_samp_rate,
        options.audio_gain,
    )


if __name__ == "__main__":
    main()

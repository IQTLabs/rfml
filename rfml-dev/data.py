import os
import re
import gzip
import cv2 as cv
import copy
import bz2
import zstandard
import json
import numpy as np
import shutil
import warnings
import yaml
import matplotlib.pyplot as plt

from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from python_on_whales import docker


from auto_label import auto_label, auto_label_configs
from zst_parse import parse_zst_filename
from spectrogram import spectrogram, spectrogram_cmap

SIGMF_META_DEFAULT = {
    "global": {  # https://github.com/sigmf/SigMF/blob/sigmf-v1.x/sigmf-spec.md#global-object
        "core:version": "1.0.0",
        "core:datatype": None,
        "core:sample_rate": None,
        "core:dataset": None,
    },
    "captures": [  # https://github.com/sigmf/SigMF/blob/sigmf-v1.x/sigmf-spec.md#capture-segment-objects
        {
            "core:frequency": None,
            "core:sample_start": 0,
        }
    ],
    "annotations": [],  # https://github.com/sigmf/SigMF/blob/sigmf-v1.x/sigmf-spec.md#annotation-segment-objects
    "spectrograms": {},
    # spectrogram should have (img_filename, sample_start, sample_count, nfft, augmentations)
}

SIGMF_ANNOTATION_DEFAULT = {
    "core:sample_start": None,
    "core:sample_count": None,
    "core:freq_lower_edge": None,
    "core:freq_upper_edge": None,
    "core:label": None,
    # "core:comment": None,
}

SPECTROGRAM_METADATA_DEFAULT = {
    "sample_start": None,
    "sample_count": None,
    "nfft": None,
    "augmentations": None,
}

LABELME_SHAPE_DEFAULT = {
    "label": None,
    "text": "",
    "points": [],
    "group_id": None,
    "shape_type": "rectangle",
    "flags": {},
}

LABELME_DEFAULT = {
    "version": "0.3.3",
    "flags": {},
    "shapes": [],
    "imagePath": None,
    "imageData": None,
    "imageHeight": None,
    "imageWidth": None,
}

SIGMF_TO_NP = {
    "ci8": "<i1",
    "ci16_le": "<i2",
    "ci32_le": "<i4",
    "cu8": "<u1",
    "cu16_le": "<u2",
    "cu32_le": "<u4",
    "cf32_le": "<f4",
    "cf64_le": "<f8",
}

CV_CMAPS = {
    "turbo": cv.COLORMAP_TURBO,
    "viridis": cv.COLORMAP_VIRIDIS,
}


class Data:
    """
    Create an object representing an I/Q signal recording and associated metadata.

    Args:
        filename (str): The path of the I/Q recording or metadata. File can be
            of type .sigmf-meta, .sigmf-data, or .zst (using GamutRF format).

    Attributes:
        data_filename (str): The path of the I/Q recording.
        sigmf_meta_filename (str): The path of the .sigmf-meta file. Will be created
            and populated if it does not exist.
        metadata (dict): The metadata associated with the I/Q recording.
    """

    def __init__(self, filename):
        """
        Creates and/or loads metadata from .sigmf-meta file.
        """

        self.filename = filename

        if not os.path.isfile(self.filename):
            raise ValueError(f"File: {self.filename} is not a valid file.")

        if self.filename.lower().endswith(".sigmf-meta"):
            self.sigmf_meta_filename = self.filename
            self.data_filename = (
                f"{os.path.splitext(self.sigmf_meta_filename)[0]}.sigmf-data"
            )
            if not os.path.isfile(self.data_filename):
                raise ValueError(f"File: {self.data_filename} is not a valid file.")
        elif self.filename.lower().endswith(".sigmf-data"):
            self.data_filename = self.filename
            self.sigmf_meta_filename = (
                f"{os.path.splitext(self.data_filename)[0]}.sigmf-meta"
            )
            if not os.path.isfile(self.sigmf_meta_filename):
                raise ValueError(
                    f"File: {self.sigmf_meta_filename} is not a valid vile."
                )
        elif self.filename.lower().endswith(".zst"):
            self.data_filename = self.filename
            self.sigmf_meta_filename = (
                f"{os.path.splitext(self.data_filename)[0]}.sigmf-meta"
            )
            if not os.path.isfile(self.sigmf_meta_filename):
                self.zst_to_sigmf_meta()
        else:
            raise ValueError(
                f"Extension: {os.path.splitext(self.filename)[1]} of file: {self.filename} unknown."
            )

        # print(f"\nData file: {self.data_filename}")
        # print(f"\nSigMF-meta file: {self.sigmf_meta_filename}")

        self.metadata = json.load(open(self.sigmf_meta_filename))

    def auto_label_spectrograms(self, signal_type):
        """
        Label all spectrogram images (from current metadata) using the auto_label.py
        script and update metadata with the converted annotations.

        Args:
            signal_type (str): The signal type used by auto_label.py to choose the labelling
                configuration. See auto_label_configs in auto_label.py for signal types.
        """

        if signal_type not in auto_label_configs:
            raise ValueError(
                f"{signal_type=} must be in {list(auto_label_configs.keys())}"
            )

        for spectrogram_filename, spectrogram_metadata in self.metadata[
            "spectrograms"
        ].items():
            yolo_label_filepath = auto_label(
                spectrogram_filename, **auto_label_configs[signal_type]["args"]
            )

            self.import_label(
                "yolo",
                yolo_label_filepath,
                spectrogram_filename,
                spectrogram_metadata,
                yolo_class_labels=auto_label_configs[signal_type]["yolo_class_labels"],
                overwrite=False,
            )

    def get_sample_reader(self):
        """
        Create file object or reader depending on the type of data file.

        Returns:
            file or reader: The file object or reader.
        """

        # nosemgrep:github.workflows.config.useless-inner-function
        def bz2_reader(x):
            return bz2.open(x, "rb")

        # nosemgrep:github.workflows.config.useless-inner-function
        def gzip_reader(x):
            return gzip.open(x, "rb")

        # nosemgrep:github.workflows.config.useless-inner-function
        def zst_reader(x):
            return zstandard.ZstdDecompressor().stream_reader(
                open(x, "rb"), read_across_frames=True
            )

        def default_reader(x):
            return open(x, "rb")

        if self.data_filename.endswith(".bz2"):
            return bz2_reader
        if self.data_filename.endswith(".gz"):
            return gzip_reader
        if self.data_filename.endswith(".zst"):
            return zst_reader

        return default_reader

    def get_samples(self, n_seek_samples=0, n_samples=None):
        """
        Read bytes from binary I/Q files and format them as complex vectors.

        Args:
            n_seek_samples (int): The index of the sample to begin reading from,
                (i.e., the number of samples to seek through the file before
                reading). (Default = 0)
            n_samples (int): Number of samples to read from the I/Q file.
                (Default = None, read all samples)

        Returns:
            np.array: Complex vector of I/Q samples.
        """

        reader = self.get_sample_reader()

        np_dtype = SIGMF_TO_NP[self.metadata["global"]["core:datatype"]]
        sample_dtype = np.dtype([("i", np_dtype), ("q", np_dtype)])

        with reader(self.data_filename) as infile:
            if n_seek_samples:
                infile.seek(int(n_seek_samples * sample_dtype.itemsize))

            if n_samples:
                sample_buffer = infile.read(int(n_samples * sample_dtype.itemsize))
            else:
                sample_buffer = infile.read()

            n_buffered_samples = int(len(sample_buffer) / sample_dtype.itemsize)
            if len(sample_buffer) % sample_dtype.itemsize != 0:
                raise ValueError(
                    f"Size mismatch. Sample bytes are not a multiple of sample dtype size."
                )

            if n_buffered_samples == 0:
                # raise ValueError(f"No samples could be read from {self.data_filename}.")
                # warnings.warn(f"{n_seek_samples} samples read. No more samples could be read from {self.data_filename}.")
                # reached end of file
                return None
            if n_samples and n_buffered_samples != n_samples:
                warnings.warn(
                    f"Could only read {n_buffered_samples}/{n_samples} samples from {self.data_filename}."
                )
                return None
            x1d = np.frombuffer(
                sample_buffer, dtype=sample_dtype, count=n_buffered_samples
            )
            return x1d["i"] + np.csingle(1j) * x1d["q"]

    def write_sigmf_meta(self, sigmf_meta):
        """
        Write metadata to .sigmf-meta file.

        Args:
            sigmf_meta (dict): Data to write.
        """

        with open(self.sigmf_meta_filename, "w") as outfile:
            print(f"Saving {self.sigmf_meta_filename}\n")
            outfile.write(json.dumps(sigmf_meta, indent=4))

    def zst_to_sigmf_meta(self):
        """
        Parse metadata from .zst filename (GamutRF format) and write .sigmf-meta file.
        """
        file_info = parse_zst_filename(self.data_filename)
        sigmf_meta = copy.deepcopy(SIGMF_META_DEFAULT)

        sigmf_meta["global"]["core:dataset"] = self.data_filename
        sigmf_meta["global"]["core:datatype"] = file_info["sigmf_datatype"]
        sigmf_meta["global"]["core:sample_rate"] = int(file_info["sample_rate"])
        sigmf_meta["captures"][0]["core:frequency"] = int(file_info["freq_center"])
        sigmf_meta["captures"][0]["core:datetime"] = (
            datetime.fromtimestamp(float(file_info["timestamp"]))
            .astimezone(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

        self.write_sigmf_meta(sigmf_meta)

    def generate_gamutrf_spectrograms(
        self,
        n_samples,
        n_fft,
        image_outdir=None,
        cmap_str="turbo",
        overwrite=False,
    ):
        """
        Create spectrogram images using GamutRF and save metadata.

        Args:
            n_samples (int): Number of samples to use per spectrogram.
            n_fft (int): Length of the FFT.
            image_outdir (str): Directory to save spectrogram images.
                (Default = None, will create new directory called {data_filename}_images)
            cmap_str (str): The colormap to use for generating the spectrogram.
                (Default = turbo)
            overwrite (bool): Whether to overwrite metadata. (Default = False)
        """

        if image_outdir is None:
            image_outdir = f"{self.data_filename}_images"
        image_outdir = Path(image_outdir)
        image_outdir.mkdir(parents=True, exist_ok=True)

        docker.run(
            "iqtlabs/gamutrf:latest",
            interactive=True,
            tty=True,
            volumes=[
                (
                    f"{self.data_filename}",
                    f"/gamutrf/data/{os.path.basename(self.data_filename)}",
                ),
                (f"{image_outdir}", f"/gamutrf/data/{os.path.basename(image_outdir)}"),
            ],
            command=[
                "gamutrf-offline",
                f"/gamutrf/data/{os.path.basename(self.data_filename)}",
                f"--nfft={int(n_fft)}",
                f"--tune-step-fft={int(n_samples/n_fft)}",
                f"--colormap={CV_CMAPS[cmap_str]}",
                "--fft_batch_size=1",
                "--rotate_secs=0",
                f"--inference_output_dir=/gamutrf/data/{os.path.basename(image_outdir)}",
            ],
        )

        image_files = [
            image_file
            for image_file in os.listdir(image_outdir)
            if image_file.endswith(".png")
        ]

        for image_filename in sorted(image_files):
            reg = re.compile(r"^image_(\d+)_([\d.]+)_(\d+)x(\d+)_(\d+)Hz\.png$")

            fft_count = int(reg.match(image_filename).group(1))

            spectrogram_metadata = copy.deepcopy(SPECTROGRAM_METADATA_DEFAULT)
            spectrogram_metadata["sample_start"] = n_fft * fft_count
            spectrogram_metadata["sample_count"] = n_samples
            spectrogram_metadata["nfft"] = n_fft

            self.import_image(
                str(Path(image_outdir, image_filename)),
                spectrogram_metadata,
                overwrite=overwrite,
            )

    def generate_spectrograms(
        self,
        n_samples,
        n_fft,
        image_outdir=None,
        n_overlap=0,
        cmap_str="turbo",
        overwrite=False,
    ):
        """
        Create spectrogram images and save metadata.

        Args:
            n_samples (int): Number of samples to use per spectrogram.
            n_fft (int): Length of the FFT.
            image_outdir (str): Directory to save spectrogram images.
                (Default = None, will create new directory called {data_filename}_images)
            n_overlap (int): Number of samples to overlap between FFTs.
            cmap_str (str): The colormap to use for generating the spectrogram.
                (Default = turbo)
            overwrite (bool): Whether to overwrite metadata. (Default = False)
        """

        cmap = plt.get_cmap(cmap_str)

        if image_outdir is None:
            image_outdir = f"{self.data_filename}_images"
        image_outdir = Path(image_outdir)
        image_outdir.mkdir(parents=True, exist_ok=True)

        n_seek_samples = 0
        while True:
            image_filename = (
                f"{os.path.basename(self.data_filename)}_{n_seek_samples}.png"
            )
            image_filepath = str(Path(image_outdir, image_filename))

            if "spectrograms" not in self.metadata:
                self.metadata["spectrograms"] = {}

            if (
                not overwrite
                and image_filepath in self.metadata["spectrograms"]
                and os.path.isfile(image_filepath)
            ):
                n_seek_samples += n_samples
                continue

            samples = self.get_samples(
                n_seek_samples=n_seek_samples, n_samples=n_samples
            )
            if samples is None:
                break

            spectrogram_data, spectrogram_raw = spectrogram(
                samples,
                self.metadata["global"]["core:sample_rate"],
                n_fft,
                n_overlap,
            )
            spectrogram_color = spectrogram_cmap(spectrogram_data, cmap)

            spectrogram_image = Image.fromarray(spectrogram_color)
            spectrogram_image.save(image_filepath)

            spectrogram_metadata = copy.deepcopy(SPECTROGRAM_METADATA_DEFAULT)
            spectrogram_metadata["sample_start"] = n_seek_samples
            spectrogram_metadata["sample_count"] = n_samples
            spectrogram_metadata["nfft"] = n_fft

            self.import_image(image_filepath, spectrogram_metadata, overwrite=overwrite)

            n_seek_samples += n_samples

    def sigmf_to_yolo(self, annotation, spectrogram):
        """
        Convert a SigMF annotation to YOLO label format. See https://github.com/sigmf/SigMF/blob/sigmf-v1.x/sigmf-spec.md#annotation-segment-objects
        for SigMF annotation specification. See https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format
        for YOLO label specification.

        Args:
            annotation (dict): SigMF annotation.
            spectrogram (dict): Spectrogram metadata.

        Returns:
            string: YOLO label string.
        """

        if "annotation_labels" not in self.metadata:
            self.metadata["annotation_labels"] = []

        if annotation["core:label"] not in self.metadata["annotation_labels"]:
            self.metadata["annotation_labels"].append(annotation["core:label"])

        label_idx = self.metadata["annotation_labels"].index(annotation["core:label"])

        freq_dim = spectrogram["nfft"]
        time_dim = spectrogram["sample_count"] / freq_dim
        sample_rate = self.metadata["global"]["core:sample_rate"]
        freq_center = self.metadata["captures"][0]["core:frequency"]
        min_freq = freq_center - (sample_rate / 2)
        max_freq = freq_center + (sample_rate / 2)
        freq_space = list(np.linspace(start=min_freq, stop=max_freq, num=int(freq_dim)))
        sample_space = list(
            np.linspace(
                start=(spectrogram["sample_start"] + spectrogram["sample_count"]),
                stop=spectrogram["sample_start"],
                num=int(time_dim) + 1,
            )
        )

        # (freq_space[963]-freq_space[60])/(freq_space[1]-freq_space[0])/1024
        # (freq_space[963]-freq_space[60])/((max_freq-min_freq)/(1024-1))/1024
        # width = (annotation["core:freq_upper_edge"] - annotation["core:freq_lower_edge"]) / ((max_freq - min_freq)/(freq_dim-1))/freq_dim
        width = (
            freq_space.index(annotation["core:freq_upper_edge"])
            - freq_space.index(annotation["core:freq_lower_edge"])
        ) / freq_dim

        # (((freq_space[963]+freq_space[60])/2)-min_freq) / ((max_freq-min_freq)/(1024-1))/1024
        x_center = (
            (
                freq_space.index(annotation["core:freq_upper_edge"])
                + freq_space.index(annotation["core:freq_lower_edge"])
            )
            / 2
        ) / freq_dim

        # (sample_space.index(2008064)-1 - sample_space.index(2008064+23552)) / 512
        height = (
            (sample_space.index(annotation["core:sample_start"]) - 1)
            - sample_space.index(
                annotation["core:sample_start"] + annotation["core:sample_count"]
            )
        ) / time_dim

        # ((sample_space.index(2008064)-1 + sample_space.index(2008064+23552))/2) / 512
        y_center = (
            (
                (sample_space.index(annotation["core:sample_start"]) - 1)
                + sample_space.index(
                    annotation["core:sample_start"] + annotation["core:sample_count"]
                )
            )
            / 2
        ) / time_dim

        yolo_label = f"{label_idx} {x_center} {y_center} {width} {height}"

        return yolo_label

    def sigmf_to_labelme(self, annotation, spectrogram, spectrogram_filename):
        """
        Convert a SigMF annotation to LabelMe JSON format. See https://github.com/sigmf/SigMF/blob/sigmf-v1.x/sigmf-spec.md#annotation-segment-objects
        for SigMF annotation specification. See https://roboflow.com/formats/labelme-json for an example LabelMe JSON.

        Args:
            annotation (dict): SigMF annotation.
            spectrogram (dict): Spectrogram metadata.
            spectrogram_filename (str): Path of spectrogram file.

        Returns:
            dict: LabelMe JSON data as a dictionary.
        """

        if "annotation_labels" not in self.metadata:
            self.metadata["annotation_labels"] = []

        if annotation["core:label"] not in self.metadata["annotation_labels"]:
            self.metadata["annotation_labels"].append(annotation["core:label"])

        with Image.open(spectrogram_filename) as spectrogram_image:
            img_width, img_height = spectrogram_image.size

        freq_dim = spectrogram["nfft"]
        time_dim = spectrogram["sample_count"] / freq_dim
        sample_rate = self.metadata["global"]["core:sample_rate"]
        freq_center = self.metadata["captures"][0]["core:frequency"]
        min_freq = freq_center - (sample_rate / 2)
        max_freq = freq_center + (sample_rate / 2)
        freq_space = list(np.linspace(start=min_freq, stop=max_freq, num=int(freq_dim)))
        sample_space = list(
            np.linspace(
                start=(spectrogram["sample_start"] + spectrogram["sample_count"]),
                stop=spectrogram["sample_start"],
                num=int(time_dim) + 1,
            )
        )

        labelme_label = copy.deepcopy(LABELME_DEFAULT)

        labelme_label["imagePath"] = os.path.basename(spectrogram_filename)
        labelme_label["imageHeight"] = int(img_height)
        labelme_label["imageWidth"] = int(img_width)

        labelme_shape = copy.deepcopy(LABELME_SHAPE_DEFAULT)
        labelme_label["shapes"].append(labelme_shape)
        labelme_label["shapes"][0]["label"] = annotation["core:label"]

        freq_min = freq_space.index(annotation["core:freq_lower_edge"])
        freq_max = freq_space.index(annotation["core:freq_upper_edge"])

        time_min = sample_space.index(annotation["core:sample_start"]) - 1
        time_max = sample_space.index(
            annotation["core:sample_count"] + annotation["core:sample_start"]
        )

        x_rescale = img_width / freq_dim
        y_rescale = img_height / time_dim

        x_min = freq_min * x_rescale
        x_max = freq_max * x_rescale
        y_min = time_max * y_rescale
        y_max = time_min * y_rescale

        points = [[x_min, y_min], [x_max, y_max]]
        labelme_label["shapes"][0]["points"] = points

        return labelme_label

    def find_matching_spectrograms(self, sample_start, sample_count):
        """
        Find spectrograms that contain the specified samples.

        Args:
            sample_start (int): Index of first sample relative to the start of the I/Q file.
            sample_count (int): Number of samples.

        Returns:
            list: Matching spectrogram file paths.
        """

        matching_spectrograms = []
        for spectrogram_filename, spectrogram in self.metadata["spectrograms"].items():
            if (sample_start >= spectrogram["sample_start"]) and (
                (sample_start + sample_count)
                <= (spectrogram["sample_start"] + spectrogram["sample_count"])
            ):
                matching_spectrograms.append(spectrogram_filename)
        return matching_spectrograms

    def export_labelme(self, label_outdir, image_outdir=None):
        """
        Create LabelMe JSON files and update metadata with LabelMe JSON files and any new images.
        Starts by converting all SigMF annotations to LabelMe JSON format if not already available.

        Args:
            label_outdir (str): Directory to save LabelMe JSON files.
            image_outdir (str): Directory to save copies of images. (Default = None, will only copy images if not None)
        """

        self.convert_all_sigmf_to_labelme()

        Path(label_outdir).mkdir(parents=True, exist_ok=True)
        if image_outdir:
            Path(image_outdir).mkdir(parents=True, exist_ok=True)

        new_image = 0
        new_metadata = 0
        for spectrogram_filename, spectrogram in (
            self.metadata["spectrograms"].copy().items()
        ):
            if "labels" not in spectrogram:
                continue
            if "labelme" not in spectrogram["labels"]:
                continue

            basefilename = os.path.splitext(os.path.basename(spectrogram_filename))[0]
            labelme_filename = f"{basefilename}.json"
            labelme_filepath = Path(label_outdir, labelme_filename)
            with open(labelme_filepath, "w") as outfile:
                print(f"Saving {labelme_filepath}\n")
                outfile.write(json.dumps(spectrogram["labels"]["labelme"], indent=4))

                if (
                    "labelme_file"
                    not in self.metadata["spectrograms"][spectrogram_filename]["labels"]
                ):
                    self.metadata["spectrograms"][spectrogram_filename]["labels"][
                        "labelme_file"
                    ] = str(labelme_filepath)
                    new_metadata += 1

            if image_outdir:
                # copy image file into new directory
                new_spectrogram_filename = str(
                    Path(image_outdir, os.path.basename(spectrogram_filename))
                )
                try:
                    shutil.copy2(spectrogram_filename, new_spectrogram_filename)
                    # copy entry in metadata
                    self.metadata["spectrograms"][
                        new_spectrogram_filename
                    ] = copy.deepcopy(spectrogram)
                    self.metadata["spectrograms"][new_spectrogram_filename]["labels"][
                        "labelme_file"
                    ] = str(labelme_filepath)
                    new_image += 1
                except shutil.SameFileError:
                    pass

        if new_image or new_metadata:
            self.write_sigmf_meta(self.metadata)

    def export_yolo(self, label_outdir, image_outdir=None):
        """
        Create YOLO label .txt files and update metadata with YOLO label files and any new images.
        Starts by converting all SigMF annotations to YOLO format if not already available.

        Args:
            label_outdir (str): Directory to save YOLO .txt files.
            image_outdir (str): Directory to save copies of images. (Default = None, will only copy images if not None)
        """

        self.convert_all_sigmf_to_yolo()

        Path(label_outdir).mkdir(parents=True, exist_ok=True)
        if image_outdir:
            Path(image_outdir).mkdir(parents=True, exist_ok=True)

        new_image = 0
        new_metadata = 0
        for spectrogram_filename, spectrogram in (
            self.metadata["spectrograms"].copy().items()
        ):
            if "labels" not in spectrogram:
                continue
            if "yolo" not in spectrogram["labels"]:
                continue

            basefilename = os.path.splitext(os.path.basename(spectrogram_filename))[0]
            yolo_filename = f"{basefilename}.txt"
            yolo_filepath = Path(label_outdir, yolo_filename)
            with open(yolo_filepath, "w") as f:
                for annotation in spectrogram["labels"]["yolo"]:
                    f.write(f"{annotation}\n")
                print(f"Saving {yolo_filepath}\n")
                if (
                    "yolo_file"
                    not in self.metadata["spectrograms"][spectrogram_filename]["labels"]
                ):
                    self.metadata["spectrograms"][spectrogram_filename]["labels"][
                        "yolo_file"
                    ] = str(yolo_filepath)
                    new_metadata += 1

            if image_outdir:
                # copy image file into new directory
                new_spectrogram_filename = str(
                    Path(image_outdir, os.path.basename(spectrogram_filename))
                )
                try:
                    shutil.copy2(spectrogram_filename, new_spectrogram_filename)
                    # copy entry in metadata
                    self.metadata["spectrograms"][
                        new_spectrogram_filename
                    ] = copy.deepcopy(spectrogram)
                    self.metadata["spectrograms"][new_spectrogram_filename]["labels"][
                        "yolo_file"
                    ] = str(yolo_filepath)
                    new_image += 1
                except shutil.SameFileError:
                    pass

        if new_image or new_metadata:
            self.write_sigmf_meta(self.metadata)

    def convert_all_sigmf_to_yolo(self):
        """
        Convert all SigMF annotations to YOLO label format and update metadata.
        """

        if not self.metadata["spectrograms"]:
            raise ValueError("No spectrograms found.")

        if not self.metadata["annotations"]:
            raise ValueError("No annotations found.")

        new_annotations = 0

        for annotation in self.metadata["annotations"]:
            sample_start = annotation["core:sample_start"]
            sample_count = annotation["core:sample_count"]

            matching_spectrograms = self.find_matching_spectrograms(
                sample_start, sample_count
            )

            if not matching_spectrograms:
                warnings.warn(
                    f"Matching spectrogram could not be found for annotation: {annotation}"
                )
                continue

            for spectrogram_filename in matching_spectrograms:
                spectrogram = self.metadata["spectrograms"][spectrogram_filename]

                if "labels" not in spectrogram:
                    spectrogram["labels"] = {}
                if "yolo" not in spectrogram["labels"]:
                    spectrogram["labels"]["yolo"] = []

                yolo_label = self.sigmf_to_yolo(annotation, spectrogram)

                if yolo_label not in spectrogram["labels"]["yolo"]:
                    spectrogram["labels"]["yolo"].append(yolo_label)
                    new_annotations += 1

        if new_annotations:
            self.write_sigmf_meta(self.metadata)

    def convert_all_sigmf_to_labelme(self):
        """
        Convert all SigMF annotations to LabelMe JSON format and update metadata.
        """

        if not self.metadata["spectrograms"]:
            raise ValueError("No spectrograms found.")

        if not self.metadata["annotations"]:
            raise ValueError("No annotations found.")

        new_annotations = 0

        for annotation in self.metadata["annotations"]:
            sample_start = annotation["core:sample_start"]
            sample_count = annotation["core:sample_count"]

            matching_spectrograms = self.find_matching_spectrograms(
                sample_start, sample_count
            )

            if not matching_spectrograms:
                warnings.warn(
                    f"Matching spectrograms could not be found for annotation: {annotation}"
                )
                continue

            for spectrogram_filename in matching_spectrograms:
                spectrogram = self.metadata["spectrograms"][spectrogram_filename]

                if "labels" not in spectrogram:
                    spectrogram["labels"] = {}
                if "labelme" not in spectrogram["labels"]:
                    spectrogram["labels"]["labelme"] = {}

                labelme_label = self.sigmf_to_labelme(
                    annotation, spectrogram, spectrogram_filename
                )

                if not spectrogram["labels"]["labelme"]:
                    spectrogram["labels"]["labelme"] = labelme_label
                    new_annotations += 1
                elif (
                    labelme_label["shapes"][0]
                    not in spectrogram["labels"]["labelme"]["shapes"]
                ):
                    spectrogram["labels"]["labelme"]["shapes"].append(
                        labelme_label["shapes"][0]
                    )
                    new_annotations += 1

        if new_annotations:
            self.write_sigmf_meta(self.metadata)

    def labelme_to_sigmf(self, labelme_json, img_filename):
        """
        Convert LabelMe JSON to SigMF annotation and update metadata.
        See https://roboflow.com/formats/labelme-json for an example LabelMe JSON.
        See https://github.com/sigmf/SigMF/blob/sigmf-v1.x/sigmf-spec.md#annotation-segment-objects
        for SigMF annotation specification.

        Args:
            labelme_json (dict): LabelMe JSON data.
            img_filename (str): Path of spectrogram image file.
        """

        spectrogram_metadata = self.metadata["spectrograms"][img_filename]

        sample_rate = self.metadata["global"]["core:sample_rate"]
        unix_timestamp = datetime.strptime(
            self.metadata["captures"][0]["core:datetime"], "%Y-%m-%dT%H:%M:%S.%f%z"
        ).timestamp()
        sample_start_time = unix_timestamp + (
            spectrogram_metadata["sample_start"] / sample_rate
        )
        sample_end_time = unix_timestamp + (
            (
                spectrogram_metadata["sample_start"]
                + spectrogram_metadata["sample_count"]
            )
            / sample_rate
        )
        freq_dim = spectrogram_metadata["nfft"]
        time_dim = spectrogram_metadata["sample_count"] / freq_dim
        time_space = np.linspace(
            start=sample_end_time, stop=sample_start_time, num=int(time_dim)
        )
        sample_space = np.linspace(
            start=(
                spectrogram_metadata["sample_start"]
                + spectrogram_metadata["sample_count"]
            ),
            stop=spectrogram_metadata["sample_start"],
            num=int(time_dim) + 1,
        )

        freq_center = self.metadata["captures"][0]["core:frequency"]
        min_freq = freq_center - (sample_rate / 2)
        max_freq = freq_center + (sample_rate / 2)
        freq_space = np.linspace(start=min_freq, stop=max_freq, num=int(freq_dim))

        img_width = labelme_json["imageWidth"]
        img_height = labelme_json["imageHeight"]

        new_annotations = 0
        for labelme_annotation in labelme_json["shapes"]:
            x_min = labelme_annotation["points"][0][0]
            y_min = labelme_annotation["points"][0][1]
            x_max = labelme_annotation["points"][1][0]
            y_max = labelme_annotation["points"][1][1]

            x_rescale = freq_dim / img_width  # 1024 / 640
            y_rescale = time_dim / img_height  # 512 / 640

            freq_min = x_min * x_rescale
            freq_max = x_max * x_rescale
            time_min = y_min * y_rescale
            time_max = y_max * y_rescale

            sigmf_annotation = copy.deepcopy(SIGMF_ANNOTATION_DEFAULT)

            sigmf_annotation["core:sample_start"] = int(
                sample_space[min(int(time_max) + 1, int(time_dim))]
            )
            sigmf_annotation["core:sample_count"] = (
                int(sample_space[int(time_min)]) - sigmf_annotation["core:sample_start"]
            )
            sigmf_annotation["core:freq_lower_edge"] = freq_space[
                int(freq_min)
            ]  # min_freq
            sigmf_annotation["core:freq_upper_edge"] = freq_space[
                int(freq_max)
            ]  # max_freq
            sigmf_annotation["core:label"] = labelme_annotation["label"]
            # sigmf_annotation["core:comment"] = "labelme"

            found_in_sigmf = False
            for annot in self.metadata["annotations"]:
                if sigmf_annotation.items() <= annot.items():
                    if "labelme" not in annot["core:comment"]:
                        annot["core:comment"] += ",labelme"
                        new_annotations += 1
                    found_in_sigmf = True

            sigmf_annotation["core:comment"] = "labelme"
            if not found_in_sigmf:
                new_annotations += 1
                self.metadata["annotations"].append(sigmf_annotation)

        if new_annotations:
            self.write_sigmf_meta(self.metadata)

    def yolo_to_sigmf(self, yolo_txt, img_filename, yolo_class_labels):
        """
        Convert YOLO label .txt file content to SigMF annotation and update metadata.
        See https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format for YOLO label specification.
        See https://github.com/sigmf/SigMF/blob/sigmf-v1.x/sigmf-spec.md#annotation-segment-objects
        for SigMF annotation specification.

        Args:
            yolo_txt (list): Contents of the YOLO label file with each line as a string.
            img_filename (str): Path of spectrogram image file.
            yolo_class_labels (list): Label class names.
        """

        spectrogram_metadata = self.metadata["spectrograms"][img_filename]

        freq_dim = spectrogram_metadata["nfft"]
        time_dim = spectrogram_metadata["sample_count"] / freq_dim
        sample_rate = self.metadata["global"]["core:sample_rate"]
        freq_center = self.metadata["captures"][0]["core:frequency"]
        min_freq = freq_center - (sample_rate / 2)
        max_freq = freq_center + (sample_rate / 2)
        width = int(freq_dim)
        height = int(time_dim)

        sample_space = np.linspace(
            start=(
                spectrogram_metadata["sample_start"]
                + spectrogram_metadata["sample_count"]
            ),
            stop=spectrogram_metadata["sample_start"],
            num=int(time_dim) + 1,
        )
        freq_space = np.linspace(start=min_freq, stop=max_freq, num=int(freq_dim))

        new_annotations = 0
        for line in yolo_txt:
            sigmf_annotation = copy.deepcopy(SIGMF_ANNOTATION_DEFAULT)

            values = line.split()

            class_id = int(values[0])
            x_center, y_center, w, h = [float(value) for value in values[1:]]

            x_min = (x_center - 0.5 * w) * width
            y_min = (y_center - 0.5 * h) * height
            x_max = (x_center + 0.5 * w) * width
            y_max = (y_center + 0.5 * h) * height

            sigmf_annotation["core:sample_start"] = int(
                sample_space[min(int(y_max) + 1, int(time_dim))]
            )
            sigmf_annotation["core:sample_count"] = (
                int(sample_space[int(y_min)]) - sigmf_annotation["core:sample_start"]
            )
            sigmf_annotation["core:freq_lower_edge"] = freq_space[
                int(x_min)
            ]  # min_freq
            sigmf_annotation["core:freq_upper_edge"] = freq_space[
                min(int(x_max), int(freq_dim - 1))
            ]  # max_freq
            sigmf_annotation["core:label"] = yolo_class_labels[class_id]
            # sigmf_annotation["core:comment"] = "yolo"

            found_in_sigmf = False
            for annot in self.metadata["annotations"]:
                if sigmf_annotation.items() <= annot.items():
                    if "yolo" not in annot["core:comment"]:
                        annot["core:comment"] += ",yolo"
                        new_annotations += 1
                    found_in_sigmf = True

            sigmf_annotation["core:comment"] = "yolo"
            if not found_in_sigmf:
                new_annotations += 1
                self.metadata["annotations"].append(sigmf_annotation)

        if new_annotations:
            self.write_sigmf_meta(self.metadata)

    def import_image(self, image_filepath, spectrogram_metadata, overwrite=False):
        """
        Update metadata with spectrogram image and spectrogram metadata.

        Args:
            image_filepath (str): Path of spectrogram image file.
            spectrogram_metadata (dict): Metadata for spectrogram image.
            overwrite (bool): Forces metadata to be overwritten. (Default = False)
        """

        # if image not in sigmf["spectrograms"] or current metadata is not a subset
        if (
            overwrite
            or (image_filepath not in self.metadata["spectrograms"])
            or not (
                spectrogram_metadata.items()
                <= self.metadata["spectrograms"][image_filepath].items()
            )
        ):
            self.metadata["spectrograms"][image_filepath] = spectrogram_metadata
            self.write_sigmf_meta(self.metadata)

    def import_label(
        self,
        label_type,
        label_filepath,
        image_filepath,
        spectrogram_metadata,
        yolo_class_labels=None,
        overwrite=False,
    ):
        """
        Update metadata with spectrogram image, original label, equivalent
        SigMF annotation, and other spectrogram metadata.

        Args:
            label_type (str): Type of label. Currently support yolo and labelme.
            label_filepath (str): Path of label file.
            image_filepath (str): Path of spectrogram image.
            spectrogram_metadata (dict): Metadata for spectrogram image.
            yolo_class_labels (list): Label class names. (Default = None, only required if label_type == yolo)
            overwrite (bool): Forces metadata to be overwritten. (Default = False)
        """

        if "yolo" in label_type and yolo_class_labels is None:
            raise ValueError("Must define yolo_class_labels when using Yolo dataset")

        if "labelme" in label_type.lower():
            label_type = "labelme"
            label_metadata = json.load(open(label_filepath))
        elif "yolo" in label_type.lower():
            label_type = "yolo"
            with open(label_filepath) as yolo_file:
                label_metadata = [line.rstrip() for line in yolo_file]
            # with open(yolo_dataset_yaml, "r") as stream:
            #     dataset_yaml = yaml.safe_load(stream)
            #     class_labels = dataset_yaml["names"]
            if "annotation_labels" not in self.metadata:
                self.metadata["annotation_labels"] = yolo_class_labels
                self.write_sigmf_meta(self.metadata)

        new_label = {
            label_type: label_metadata,
            f"{label_type}_file": label_filepath,
        }
        # Checks if spectrogram_metadata is a subset of any spectrogram in SigMF
        if (image_filepath in self.metadata["spectrograms"]) and (
            spectrogram_metadata.items()
            <= self.metadata["spectrograms"][image_filepath].items()
        ):
            if "labels" not in self.metadata["spectrograms"][image_filepath]:
                self.metadata["spectrograms"][image_filepath]["labels"] = {}

            if (
                label_type
                not in self.metadata["spectrograms"][image_filepath]["labels"]
                or self.metadata["spectrograms"][image_filepath]["labels"][label_type]
                != label_metadata
            ):
                self.metadata["spectrograms"][image_filepath]["labels"].update(
                    new_label
                )
                self.write_sigmf_meta(self.metadata)

        else:
            spectrogram_metadata["labels"] = new_label
            self.metadata["spectrograms"][image_filepath] = spectrogram_metadata
            self.write_sigmf_meta(self.metadata)

        if "labelme" in label_type.lower():
            self.labelme_to_sigmf(label_metadata, image_filepath)
        elif "yolo" in label_type.lower():
            self.yolo_to_sigmf(label_metadata, image_filepath, yolo_class_labels)


def labels_to_sigmf(metadata_getter, label_type, yolo_dataset_yaml=None):
    """
    Batch import of labels.

    Args:
        metadata_getter (generator): Function that yields tuples of (image path (str),
            label path (str), sample recording path (str), spectrogram metadata (dict)). This generator is
            responsible for mapping images and labels to sample recording files and populating the
            necessary fields for the spectrogram metadata dictionary.
        label_type (str): Type of label. Currently support yolo and labelme.
        yolo_dataset_yaml (str): Path of YOLO dataset yaml file. (Default = None, only required if label_type == yolo)
    """

    if "yolo" in label_type and yolo_dataset_yaml is None:
        raise ValueError("Must define yolo_dataset_yaml when using Yolo dataset")

    if "yolo" in label_type:
        with open(yolo_dataset_yaml, "r") as stream:
            dataset_yaml = yaml.safe_load(stream)
            yolo_class_labels = dataset_yaml["names"]

    for (
        image_filepath,
        label_filepath,
        sample_filepath,
        spectrogram_metadata,
    ) in metadata_getter:
        data_object = Data(sample_filepath)
        data_object.import_label(
            label_type,
            label_filepath,
            image_filepath,
            spectrogram_metadata,
            yolo_class_labels=yolo_class_labels,
        )


def images_to_sigmf(metadata_getter):
    """
    Batch import of images.

    Args:
        metadata_getter (generator): Function that yields tuples of (image path (str),
            spectrogram metadata (dict), sample recording path (str)). This generator is
            responsible for mapping images to sample recording files and populating the
            necessary fields for the spectrogram metadata dictionary.
    """

    for image_filepath, spectrogram_metadata, sample_filepath in metadata_getter:
        data_object = Data(sample_filepath)
        data_object.import_image(image_filepath, spectrogram_metadata)


def yield_image_metadata_from_filename(images_directory, samples_directory):
    """
    Yield filenames and metadata by parsing metadata from filenames.

    Args:
        images_directory (str): Directory that contains spectrogram images.
        samples_directory (str): Directory that contains the original sample recordings
            for the spectrograms in images_directory.

    Note:
        Spectrogram metadata dictionary must minimally contain keys "sample_start",
        "sample_count", and "nfft". The value of "sample_start" must be the absolute index from
        the original sample recording of the first sample used in generating the spectrogram.

    Returns:
        generator: (image file name (str), spectrogram metadata (dict), sample file name (str))
    """

    # IMAGES
    image_files = [
        image_file
        for image_file in os.listdir(images_directory)
        if image_file.endswith(".png")
    ]

    for image_filename in image_files:
        # Get sample file
        reg = re.compile(r"^(.*)_id(\d+)_batch(\d+)\.png$")
        sample_filename = reg.match(image_filename).group(1)

        # Get sample count for image
        sample_count = 1024 * 1024

        # Get nfft for image
        nfft = 1024

        # Get start sample for image
        sample_start = int(sample_count * int(reg.match(image_filename).group(3)))

        spectrogram_metadata = copy.deepcopy(SPECTROGRAM_METADATA_DEFAULT)
        spectrogram_metadata["sample_start"] = sample_start
        spectrogram_metadata["sample_count"] = sample_count
        spectrogram_metadata["nfft"] = nfft

        yield str(Path(images_directory, image_filename)), spectrogram_metadata, str(
            Path(samples_directory, sample_filename)
        )


def yield_image_metadata_from_json(
    image_directory, metadata_directory, samples_directory
):
    """
    Yield filenames and metadata by parsing metadata from custom JSON files.

    Args:
        image_directory (str): Directory that contains spectrogram images.
        metadata_directory (str): Directory that contains custom JSON metadata files.
        samples_directory (str): Directory that contains the original sample recordings
            for the spectrograms in image_directory.

    Note:
        Spectrogram metadata dictionary must minimally contain keys "sample_start",
        "sample_count", and "nfft". The value of "sample_start" must be the absolute index from
        the original sample recording of the first sample used in generating the spectrogram.

    Returns:
        generator: (image file name (str), spectrogram metadata (dict), sample file name (str))
    """

    # IMAGES
    image_files = [
        image_file
        for image_file in os.listdir(image_directory)
        if image_file.endswith(".png")
    ]

    for image_filename in image_files:
        spectrogram_metadata, sample_filename = get_custom_metadata(
            image_filename, metadata_directory
        )

        yield str(Path(image_directory, image_filename)), spectrogram_metadata, str(
            Path(samples_directory, sample_filename)
        )


def yield_label_metadata(
    label_ext,
    label_directory,
    image_directory,
    samples_directory,
    metadata_directory,
):
    """
    Yield filenames and metadata by parsing metadata from custom JSON files.

    Args:
        label_ext (str): File extension for label files.
        label_directory (str): Directory containing label files.
        image_directory (str): Directory containing spectrogram images.
        samples_directory (str): Directory containing the original sample recordings
            for the spectrograms in image_directory.
        metadata_directory (str): Directory containing custom JSON metadata files.

    Note:
        Spectrogram metadata dictionary must minimally contain keys "sample_start",
        "sample_count", and "nfft". The value of "sample_start" must be the absolute index from
        the original sample recording of the first sample used in generating the spectrogram.

    Returns:
        generator: (image file name (str), label file name (str), sample file name (str), spectrogram metadata (dict))
    """

    # LABELS
    label_files = [
        label_file
        for label_file in os.listdir(label_directory)
        if label_file.endswith(label_ext)
    ]

    for label_filename in label_files:
        # get metadata
        spectrogram_metadata, sample_filename = get_custom_metadata(
            label_filename, metadata_directory
        )
        image_filename = f"{os.path.splitext(label_filename)[0]}.png"
        sample_filename = os.path.basename(sample_filename)

        yield (
            str(Path(image_directory, image_filename)),
            str(Path(label_directory, label_filename)),
            str(Path(samples_directory, sample_filename)),
            spectrogram_metadata,
        )


def get_custom_metadata(filename, metadata_directory):
    """
    Load metadata from custom JSON files.

    Args:
        filename (str): Path of the JSON, image, or label file. (Assumes common path name minus extension)
        metadata_directory (str): Directory that contains the custom JSON files.

    Returns:
        spectrogram_metadata (dict): Dictionary containing metadata about the spectrogram.
            Must minimally contain keys "sample_start", "sample_count", and "nfft". The
            value of "sample_start" must be the absolute index from the original sample
            recording of the first sample used in generating the spectrogram.
        sample_filename (str): Path of the sample recording associated with the spectrogram.
    """

    metadata_filename = f"{os.path.splitext(filename)[0]}.json"
    if metadata_filename not in os.listdir(metadata_directory):
        raise ValueError(f"Could not find metadata file {metadata_filename}")

    metadata = json.load(open(Path(metadata_directory, metadata_filename)))

    spectrogram_metadata = {
        "sample_start": metadata["sample_start_idx"],
        "sample_count": metadata["mini_batch_size"],
        "nfft": metadata["nfft"],
        "augmentations": {
            "snr": metadata["snr"],
        },
    }

    sample_filename = metadata["sample_file"]["filename"]

    return spectrogram_metadata, sample_filename


if __name__ == "__main__":
    # /Users/ltindall/data/gamutrf/gamutrf-arl/01_30_23/mini2/snr_noise_floor/

    directory = "/Users/ltindall/data_test/snr_noise_floor/"

    # images_to_sigmf(
    #     yield_image_metadata_from_json(
    #         directory + "png", directory + "metadata", directory
    #     )
    # )

    directory = "/Users/ltindall/data_test/snr_noise_floor/"
    label_ext = ".txt"
    label_type = "yolo"
    label_directory = directory + "png/YOLODataset/labels/train/"
    image_directory = directory + "png/YOLODataset/images/train/"
    samples_directory = directory
    metadata_directory = directory + "metadata/"
    yolo_dataset_yaml = directory + "png/YOLODataset/dataset.yaml"
    labels_to_sigmf(
        yield_label_metadata(
            label_ext,
            label_directory,
            image_directory,
            samples_directory,
            metadata_directory,
        ),
        label_type,
        yolo_dataset_yaml,
    )

    # directory = "/Users/ltindall/data_test/snr_noise_floor/"
    # label_ext = ".json"
    # label_type = "labelme"
    # label_directory = directory + "png/"
    # image_directory = directory + "png/"
    # samples_directory = directory
    # metadata_directory = directory + "metadata/"
    # labels_to_sigmf(
    #     yield_label_metadata(
    #         label_ext,
    #         label_directory,
    #         image_directory,
    #         samples_directory,
    #         metadata_directory,
    #     ),
    #     label_type,
    # )

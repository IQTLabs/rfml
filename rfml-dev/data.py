import os
import re
import json
import numpy as np
import shutil
import warnings
import yaml

from datetime import datetime, timezone
from pathlib import Path

from zst_parse import parse_zst_filename

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


class Data:
    def __init__(self, filename):
        # filename is a .zst, .sigmf-meta, or .sigmf-data
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

    def write_sigmf_meta(self, sigmf_meta):
        with open(self.sigmf_meta_filename, "w") as outfile:
            print(f"Saving {self.sigmf_meta_filename}\n")
            outfile.write(json.dumps(sigmf_meta, indent=4))

    def zst_to_sigmf_meta(self):
        file_info = parse_zst_filename(self.data_filename)
        sigmf_meta = SIGMF_META_DEFAULT.copy()

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

    def generate_spectrograms(self):
        # will update sigmf-meta with image metadata
        raise NotImplementedError

    def sigmf_to_labelme(self):
        # assume existing images and annotations
        raise NotImplementedError

    def sigmf_to_yolo(self, annotation, spectrogram):
        # {
        #     "core:sample_start": 2008064,
        #     "core:sample_count": 23552,
        #     "core:freq_lower_edge": 5726461173.020528,
        #     "core:freq_upper_edge": 5744538826.979472,
        #     "core:label": "mini2_video",
        #     "core:comment": "labelme,yolo"
        # },

        # "/Users/ltindall/data_test/snr_noise_floor/png/YOLODataset/images/train/gamutrf_recording_ettus__gain40_1675088974_5735500000Hz_20480000sps.s16.zst_id25_batch25.png": {
        #     "sample_start": 1572864,
        #     "sample_count": 524288,
        #     "nfft": 1024,
        #     "augmentations": {
        #         "snr": -10
        #     },
        #     "labels": {
        #         "yolo": [
        #             "0 0.49951171875 0.146484375 0.8818359375 0.04296875",
        #             "0 0.50048828125 0.3017578125 0.8818359375 0.041015625",
        #             "0 0.50048828125 0.5361328125 0.8818359375 0.044921875",
        #             "0 0.49951171875 0.6923828125 0.8818359375 0.041015625",
        #             "0 0.50048828125 0.927734375 0.8818359375 0.04296875",
        #             "1 0.45166015625 0.458984375 0.0556640625 0.01953125",
        #             "1 0.54931640625 0.8505859375 0.0556640625 0.021484375",
        #             "1 0.54931640625 0.9970703125 0.0556640625 0.001953125"
        #         ]
        #     }
        # },

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
        sample_space = np.linspace(
            start=(spectrogram["sample_start"] + spectrogram["sample_count"]),
            stop=spectrogram["sample_start"],
            num=int(time_dim) + 1,
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
        # {
        #     "core:sample_start": 2008064,
        #     "core:sample_count": 23552,
        #     "core:freq_lower_edge": 5726461173.020528,
        #     "core:freq_upper_edge": 5744538826.979472,
        #     "core:label": "mini2_video",
        #     "core:comment": "labelme,yolo"
        # },

        # "/Users/ltindall/data_test/snr_noise_floor/png/YOLODataset/images/train/gamutrf_recording_ettus__gain40_1675088974_5735500000Hz_20480000sps.s16.zst_id25_batch25.png": {
        #     "sample_start": 1572864,
        #     "sample_count": 524288,
        #     "nfft": 1024,
        #     "augmentations": {
        #         "snr": -10
        #     },
        #     "labels": {
        #         "yolo": [
        #             "0 0.49951171875 0.146484375 0.8818359375 0.04296875",
        #             "0 0.50048828125 0.3017578125 0.8818359375 0.041015625",
        #             "0 0.50048828125 0.5361328125 0.8818359375 0.044921875",
        #             "0 0.49951171875 0.6923828125 0.8818359375 0.041015625",
        #             "0 0.50048828125 0.927734375 0.8818359375 0.04296875",
        #             "1 0.45166015625 0.458984375 0.0556640625 0.01953125",
        #             "1 0.54931640625 0.8505859375 0.0556640625 0.021484375",
        #             "1 0.54931640625 0.9970703125 0.0556640625 0.001953125"
        #         ]
        #     }
        # },

        # "/Users/ltindall/data_test/snr_noise_floor/png/gamutrf_recording_ettus__gain40_1675088974_5735500000Hz_20480000sps.s16.zst_id25_batch25.png": {
        #     "sample_start": 1572864,
        #     "sample_count": 524288,
        #     "nfft": 1024,
        #     "augmentations": {
        #         "snr": -10
        #     },
        #     "labels": {
        #         "labelme": {
        #             "version": "0.3.3",
        #             "flags": {},
        #             "shapes": [
        #                 {
        #                     "label": "mini2_video",
        #                     "text": "",
        #                     "points": [
        #                         [
        #                             60,
        #                             64
        #                         ],
        #                         [
        #                             963,
        #                             86
        #                         ]
        #                     ],
        #                     "group_id": null,
        #                     "shape_type": "rectangle",
        #                     "flags": {}
        #                 },
        #                 {
        #                     "label": "mini2_video",
        #                     "text": "",
        #                     "points": [
        #                         [
        #                             61,
        #                             144
        #                         ],
        #                         [
        #                             964,
        #                             165
        #                         ]
        #                     ],
        #                     "group_id": null,
        #                     "shape_type": "rectangle",
        #                     "flags": {}
        #                 },
        #                 {
        #                     "label": "mini2_video",
        #                     "text": "",
        #                     "points": [
        #                         [
        #                             61,
        #                             263
        #                         ],
        #                         [
        #                             964,
        #                             286
        #                         ]
        #                     ],
        #                     "group_id": null,
        #                     "shape_type": "rectangle",
        #                     "flags": {}
        #                 },
        #                 {
        #                     "label": "mini2_video",
        #                     "text": "",
        #                     "points": [
        #                         [
        #                             60,
        #                             344
        #                         ],
        #                         [
        #                             963,
        #                             365
        #                         ]
        #                     ],
        #                     "group_id": null,
        #                     "shape_type": "rectangle",
        #                     "flags": {}
        #                 },
        #                 {
        #                     "label": "mini2_video",
        #                     "text": "",
        #                     "points": [
        #                         [
        #                             61,
        #                             464
        #                         ],
        #                         [
        #                             964,
        #                             486
        #                         ]
        #                     ],
        #                     "group_id": null,
        #                     "shape_type": "rectangle",
        #                     "flags": {}
        #                 },
        #                 {
        #                     "label": "mini2_telem",
        #                     "text": "",
        #                     "points": [
        #                         [
        #                             434,
        #                             230
        #                         ],
        #                         [
        #                             491,
        #                             240
        #                         ]
        #                     ],
        #                     "group_id": null,
        #                     "shape_type": "rectangle",
        #                     "flags": {}
        #                 },
        #                 {
        #                     "label": "mini2_telem",
        #                     "text": "",
        #                     "points": [
        #                         [
        #                             534,
        #                             430
        #                         ],
        #                         [
        #                             591,
        #                             441
        #                         ]
        #                     ],
        #                     "group_id": null,
        #                     "shape_type": "rectangle",
        #                     "flags": {}
        #                 },
        #                 {
        #                     "label": "mini2_telem",
        #                     "text": "",
        #                     "points": [
        #                         [
        #                             534,
        #                             510
        #                         ],
        #                         [
        #                             591,
        #                             511
        #                         ]
        #                     ],
        #                     "group_id": null,
        #                     "shape_type": "rectangle",
        #                     "flags": {}
        #                 }
        #             ],
        #             "imagePath": "gamutrf_recording_ettus__gain40_1675088974_5735500000Hz_20480000sps.s16.zst_id25_batch25.png",
        #             "imageData": null,
        #             "imageHeight": 512,
        #             "imageWidth": 1024
        #         }
        #     }
        # },

        if "annotation_labels" not in self.metadata:
            self.metadata["annotation_labels"] = []

        if annotation["core:label"] not in self.metadata["annotation_labels"]:
            self.metadata["annotation_labels"].append(annotation["core:label"])

        freq_dim = spectrogram["nfft"]
        time_dim = spectrogram["sample_count"] / freq_dim
        sample_rate = self.metadata["global"]["core:sample_rate"]
        freq_center = self.metadata["captures"][0]["core:frequency"]
        min_freq = freq_center - (sample_rate / 2)
        max_freq = freq_center + (sample_rate / 2)
        freq_space = list(np.linspace(start=min_freq, stop=max_freq, num=int(freq_dim)))
        sample_space = np.linspace(
            start=(spectrogram["sample_start"] + spectrogram["sample_count"]),
            stop=spectrogram["sample_start"],
            num=int(time_dim) + 1,
        )

        labelme_label = LABELME_DEFAULT.copy()
        labelme_label["imagePath"] = os.path.basename(spectrogram_filename)
        labelme_label["imageHeight"] = int(time_dim)
        labelme_label["imageWidth"] = int(freq_dim)

        labelme_label["shapes"].append(LABELME_SHAPE_DEFAULT.copy())

        labelme_label["shapes"][0]["label"] = annotation["core:label"]

        x_min = freq_space.index(annotation["core:freq_lower_edge"])
        x_max = freq_space.index(annotation["core:freq_upper_edge"])

        y_max = sample_space.index(annotation["core:sample_start"]) - 1
        y_min = sample_space.index(
            annotation["core:sample_count"] + annotation["core:sample_start"]
        )

        points = [[x_min, y_min], [x_max, y_max]]
        labelme_label["shapes"][0]["points"] = points

        return labelme_label

    def find_matching_spectrogram(self, sample_start, sample_count):
        """
        sample_start and sample_count are from annotation
        """
        for spectrogram_filename, spectrogram in self.metadata["spectrograms"]:
            if (sample_start >= spectrogram["sample_start"]) and (
                (sample_start + sample_count)
                <= (spectrogram["sample_start"] + spectrogram["sample_count"])
            ):
                return spectrogram_filename
        return ""

    def convert_all_sigmf_to_yolo(self):
        # assume existing images and annotations/yolo
        if not self.metadata["spectrograms"]:
            raise ValueError("No spectrograms found.")

        if not self.metadata["annotations"]:
            raise ValueError("No annotations found.")

        new_annotations = 0

        for annotation in self.metadata["annotations"]:
            sample_start = annotation["core:sample_start"]
            sample_count = annotation["core:sample_count"]
            spectrogram_filename = self.find_matching_spectrogram(
                sample_start, sample_count
            )
            if not spectrogram_filename:
                warnings.warn(
                    f"Matching spectrogram could not be found for annotation: {annotation}"
                )
                continue
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

    def export_labelme(self, label_outdir, image_outdir=None):
        # assume existing images and annotations/labelme
        # will convert annotations if necessary
        self.convert_all_sigmf_to_labelme()

        new_image = 0
        for spectrogram_filename, spectrogram in self.metadata["spectrograms"]:
            if "labels" not in spectrogram:
                continue
            if "labelme" not in spectrogram["labels"]:
                continue

            basefilename = os.path.splitext(os.path.basename(spectrogram_filename))[0]
            labelme_filename = f"{basefilename}.json"
            with open(Path(label_outdir, labelme_filename), "w") as outfile:
                print(f"Saving {Path(label_outdir, labelme_filename)}\n")
                outfile.write(json.dumps(spectrogram["labels"]["labelme"], indent=4))

            if image_outdir:
                # copy image file into new directory
                new_spectrogram_filename = Path(
                    image_outdir, os.path.basename(spectrogram_filename)
                )
                shutil.copy2(spectrogram_filename, new_spectrogram_filename)
                # copy entry in metadata
                self.metadata["spectrograms"][new_spectrogram_filename] = spectrogram
                new_image += 1

        if new_image:
            self.write_sigmf_meta(self.metadata)

    def export_yolo(self, label_outdir, image_outdir=None):
        self.convert_all_sigmf_to_yolo()

        new_image = 0
        for spectrogram_filename, spectrogram in self.metadata["spectrograms"]:
            if "labels" not in spectrogram:
                continue
            if "yolo" not in spectrogram["labels"]:
                continue

            basefilename = os.path.splitext(os.path.basename(spectrogram_filename))[0]
            yolo_filename = f"{basefilename}.txt"
            with open(Path(label_outdir, yolo_filename), "w") as f:
                for annotation in spectrogram["labels"]["yolo"]:
                    f.write(f"{annotation}\n")
                print(f"Saving {Path(label_outdir, yolo_filename)}\n")

            if image_outdir:
                # copy image file into new directory
                new_spectrogram_filename = Path(
                    image_outdir, os.path.basename(spectrogram_filename)
                )
                shutil.copy2(spectrogram_filename, new_spectrogram_filename)
                # copy entry in metadata
                self.metadata["spectrograms"][new_spectrogram_filename] = spectrogram
                new_image += 1

        if new_image:
            self.write_sigmf_meta(self.metadata)

    def convert_all_sigmf_to_labelme(self):
        # assume existing images and annotations/labelme
        if not self.metadata["spectrograms"]:
            raise ValueError("No spectrograms found.")

        if not self.metadata["annotations"]:
            raise ValueError("No annotations found.")

        new_annotations = 0

        for annotation in self.metadata["annotations"]:
            sample_start = annotation["core:sample_start"]
            sample_count = annotation["core:sample_count"]
            spectrogram_filename = self.find_matching_spectrogram(
                sample_start, sample_count
            )
            if not spectrogram_filename:
                warnings.warn(
                    f"Matching spectrogram could not be found for annotation: {annotation}"
                )
                continue
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

        new_annotations = 0
        for labelme_annotation in labelme_json["shapes"]:
            sigmf_annotation = SIGMF_ANNOTATION_DEFAULT.copy()

            sigmf_annotation["core:sample_start"] = int(
                sample_space[int(labelme_annotation["points"][1][1]) + 1]
            )
            sigmf_annotation["core:sample_count"] = (
                int(sample_space[int(labelme_annotation["points"][0][1])])
                - sigmf_annotation["core:sample_start"]
            )
            sigmf_annotation["core:freq_lower_edge"] = freq_space[
                int(labelme_annotation["points"][0][0])
            ]  # min_freq
            sigmf_annotation["core:freq_upper_edge"] = freq_space[
                int(labelme_annotation["points"][1][0])
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

    def yolo_to_sigmf(self, yolo_txt, img_filename, class_labels):
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
            sigmf_annotation = SIGMF_ANNOTATION_DEFAULT.copy()

            values = line.split()

            class_id = int(values[0])
            x_center, y_center, w, h = [float(value) for value in values[1:]]

            x_min = (x_center - 0.5 * w) * width
            y_min = (y_center - 0.5 * h) * height
            x_max = (x_center + 0.5 * w) * width
            y_max = (y_center + 0.5 * h) * height
            points = [[x_min, y_min], [x_max, y_max]]

            sigmf_annotation["core:sample_start"] = int(
                sample_space[int(points[1][1]) + 1]
            )
            sigmf_annotation["core:sample_count"] = (
                int(sample_space[int(points[0][1])])
                - sigmf_annotation["core:sample_start"]
            )
            sigmf_annotation["core:freq_lower_edge"] = freq_space[
                int(points[0][0])
            ]  # min_freq
            sigmf_annotation["core:freq_upper_edge"] = freq_space[
                int(points[1][0])
            ]  # max_freq
            sigmf_annotation["core:label"] = class_labels[class_id]
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


class DtypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.dtype):
            return obj.descr
        return json.JSONEncoder.default(self, obj)


def images_to_sigmf(metadata_getter):
    """Instantiates a Data object and then creates or appends to a SigMF-meta file
        (for the original sample recording) using metadata from a spectrogram image.

    Args:
        metadata_getter (generator): Function that yields tuples of (image path (str),
            spectrogram metadata (dict), sample recording path (str)). This generator is
            responsible for mapping images to sample recording files and populating the
            necessary fields for the spectrogram metadata dictionary.

    """
    for image_filepath, spectrogram_metadata, sample_filepath in metadata_getter:
        data_object = Data(sample_filepath)

        # if image not in sigmf["spectrograms"] or current metadata is not a subset
        if (image_filepath not in data_object.metadata["spectrograms"]) or not (
            spectrogram_metadata.items()
            <= data_object.metadata["spectrograms"][image_filepath].items()
        ):
            data_object.metadata["spectrograms"][image_filepath] = spectrogram_metadata
            data_object.write_sigmf_meta(data_object.metadata)


def yield_image_metadata_from_filename(images_directory, samples_directory):
    """Yields filenames and metadata by parsing metadata from filenames.

    Args:
        image_directory (str): A directory that contains spectrogram images.
        samples_directory (str): A directory that contains the original sample recordings
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

        spectrogram_metadata = SPECTROGRAM_METADATA_DEFAULT.copy()
        spectrogram_metadata["sample_start"] = sample_start
        spectrogram_metadata["sample_count"] = sample_count
        spectrogram_metadata["nfft"] = nfft

        yield str(Path(images_directory, image_filename)), spectrogram_metadata, str(
            Path(samples_directory, sample_filename)
        )


def get_custom_metadata(filename, metadata_directory):
    """Loads metadata from custom json files.

    Args:
        filename (str): Path of the json, image, or label file.
            (Assumes common path name minus extension)
        metadata_directory (str): Directory that contains the custom json files.

    Returns:
        spectrogram_metadata (dict): Dictionary containing metadata about the spectrogram.
            must minimally contain keys "sample_start", "sample_count", and "nfft". The
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


def yield_image_metadata_from_json(
    image_directory, metadata_directory, samples_directory
):
    """Yields filenames and metadata by parsing metadata from json files.

    Args:
        image_directory (str): A directory that contains spectrogram images.
        metadata_directory (str): A directory that contains json metadata files.
        samples_directory (str): A directory that contains the original sample recordings
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

        yield (
            str(Path(image_directory, image_filename)),
            str(Path(label_directory, label_filename)),
            str(Path(samples_directory, sample_filename)),
            spectrogram_metadata,
        )


def labels_to_sigmf(metadata_getter, label_type, yolo_dataset_yaml=None):
    if "yolo" in label_type and yolo_dataset_yaml is None:
        raise ValueError("Must define yolo_dataset_yaml when using Yolo dataset")

    for (
        image_filepath,
        label_filepath,
        sample_filepath,
        spectrogram_metadata,
    ) in metadata_getter:
        data_object = Data(sample_filepath)

        if "labelme" in label_type.lower():
            label_metadata = json.load(open(label_filepath))
        elif "yolo" in label_type.lower():
            with open(label_filepath) as yolo_file:
                label_metadata = [line.rstrip() for line in yolo_file]
            with open(yolo_dataset_yaml, "r") as stream:
                dataset_yaml = yaml.safe_load(stream)
                class_labels = dataset_yaml["names"]
                if "annotation_labels" not in data_object.metadata:
                    data_object.metadata["annotation_labels"] = class_labels
                    data_object.write_sigmf_meta(data_object.metadata)

        # Checks if spectrogram_metadata is a subset of any spectrogram in SigMF
        if (image_filepath in data_object.metadata["spectrograms"]) and (
            spectrogram_metadata.items()
            <= data_object.metadata["spectrograms"][image_filepath].items()
        ):
            if "labels" not in data_object.metadata["spectrograms"][image_filepath]:
                data_object.metadata["spectrograms"][image_filepath]["labels"] = {}

            if (
                label_type
                not in data_object.metadata["spectrograms"][image_filepath]["labels"]
                or data_object.metadata["spectrograms"][image_filepath]["labels"][
                    label_type
                ]
                != label_metadata
            ):
                data_object.metadata["spectrograms"][image_filepath]["labels"].update(
                    {label_type: label_metadata}
                )
                data_object.write_sigmf_meta(data_object.metadata)

        else:
            spectrogram_metadata["labels"] = {label_type: label_metadata}
            data_object.metadata["spectrograms"][image_filepath] = spectrogram_metadata
            data_object.write_sigmf_meta(data_object.metadata)

        if "labelme" in label_type.lower():
            data_object.labelme_to_sigmf(label_metadata, image_filepath)
        elif "yolo" in label_type.lower():
            data_object.yolo_to_sigmf(label_metadata, image_filepath, class_labels)


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

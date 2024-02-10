import os
import json
import numpy as np
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


class Data:
    def __init__(self, filename):
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
            start=spectrogram_metadata["sample_start"],
            stop=(
                spectrogram_metadata["sample_start"]
                + spectrogram_metadata["sample_count"]
            ),
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
                sample_space[int(labelme_annotation["points"][0][1])]
            )
            sigmf_annotation["core:sample_count"] = (
                int(sample_space[int(labelme_annotation["points"][1][1]) + 1])
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
            start=spectrogram_metadata["sample_start"],
            stop=(
                spectrogram_metadata["sample_start"]
                + spectrogram_metadata["sample_count"]
            ),
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

            sigmf_annotation["core:sample_start"] = int(sample_space[int(points[0][1])])
            sigmf_annotation["core:sample_count"] = (
                int(sample_space[int(points[1][1]) + 1])
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


def images_to_sigmf(directory):
    # takes directory of spectrogram images
    # create Data object from associated sample file (creates sigmf-meta if doesn't exist)
    # populates sigmf-meta["spectrogams"] object
    # spectrogram should have (img_filename, sample_start, sample_count, nfft, augmentations)

    # IMAGES
    image_dir = Path(directory, "png")
    image_files = [
        image_file
        for image_file in os.listdir(image_dir)
        if image_file.endswith(".png")
    ]

    # METADATA
    # if no metadata saved then must be parsed from images or generated in some way
    metadata_dir = Path(directory, "metadata")
    metadata_files = os.listdir(metadata_dir)

    for image_filename in image_files:
        metadata_filename = f"{os.path.splitext(image_filename)[0]}.json"
        if metadata_filename not in metadata_files:
            raise ValueError(
                f"Could not find matching metadata file {metadata_filename} for {image_filename}"
            )

        metadata = json.load(open(Path(metadata_dir, metadata_filename)))

        # # fix the img_file and sample_file directories
        # metadata["img_file"] = str(Path(image_dir,os.path.basename(metadata['img_file'])))
        # metadata["sample_file"]["filename"] = str(Path(directory, os.path.basename(metadata['sample_file']['filename'])))
        # json_object = json.dumps(metadata, indent=4, cls=DtypeEncoder)
        # with open(Path(metadata_dir, metadata_filename), "w") as outfile:
        #     outfile.write(json_object)

        data_object = Data(metadata["sample_file"]["filename"])

        spectrogram_metadata = {
            # "filename": metadata["img_file"],
            "sample_start": metadata["sample_start_idx"],
            "sample_count": metadata["mini_batch_size"],
            "nfft": metadata["nfft"],
            "augmentations": {
                "snr": metadata["snr"],
            },
        }

        # Checks if spectrogram_metadata is a subset of any spectrogram in SigMF

        if (metadata["img_file"] not in data_object.metadata["spectrograms"]) or not (
            spectrogram_metadata.items()
            <= data_object.metadata["spectrograms"][metadata["img_file"]].items()
        ):
            data_object.metadata["spectrograms"][
                metadata["img_file"]
            ] = spectrogram_metadata
            data_object.write_sigmf_meta(data_object.metadata)

        # for sigmf_spectrogram_metadata in data_object.metadata["spectrograms"]:
        #     if spectrogram_metadata.items() <= sigmf_spectrogram_metadata.items():
        #         found_in_sigmf = True
        #         break

        # if not found_in_sigmf:
        #     data_object.metadata["spectrograms"].append(spectrogram_metadata)
        #     data_object.write_sigmf_meta(data_object.metadata)


def labels_to_sigmf(directory):
    # takes directory of labels
    # create Data object from associated sample file (creates sigmf-meta if doesn't exist)
    # populates sigmf-meta["spectrogams"] object
    # spectrogram should have (img_filename, sample_start, sample_count, nfft, augmentations)

    # LABELS
    label_dir = Path(directory, "png")

    # if labelme
    label_files = [
        label_file
        for label_file in os.listdir(label_dir)
        if label_file.endswith(".json")
    ]
    # if yolo
    # label_files = [label_file for label_file i n os.listdir(label_dir) if label_file.endswith(".txt")]

    # METADATA
    # if no metadata saved then must be parsed from images or generated in some way
    metadata_dir = Path(directory, "metadata")
    metadata_files = os.listdir(metadata_dir)

    for label_filename in label_files:
        # get metadata
        metadata_filename = f"{os.path.splitext(label_filename)[0]}.json"
        if metadata_filename not in metadata_files:
            raise ValueError(
                f"Could not find matching metadata file {metadata_filename} for {label_filename}"
            )
        metadata = json.load(open(Path(metadata_dir, metadata_filename)))

        # create Data object
        data_object = Data(metadata["sample_file"]["filename"])

        # format spectrogram metadata
        spectrogram_metadata = {
            # "filename": metadata["img_file"],
            "sample_start": metadata["sample_start_idx"],
            "sample_count": metadata["mini_batch_size"],
            "nfft": metadata["nfft"],
            "augmentations": {
                "snr": metadata["snr"],
            },
        }

        label_metadata = json.load(open(Path(label_dir, label_filename)))

        data_object.labelme_to_sigmf(label_metadata, metadata["img_file"])

        # Checks if spectrogram_metadata is a subset of any spectrogram in SigMF
        if (metadata["img_file"] in data_object.metadata["spectrograms"]) and (
            spectrogram_metadata.items()
            <= data_object.metadata["spectrograms"][metadata["img_file"]].items()
        ):
            if (
                "labels"
                not in data_object.metadata["spectrograms"][metadata["img_file"]]
            ):
                data_object.metadata["spectrograms"][metadata["img_file"]][
                    "labels"
                ] = {}

            if (
                "labelme"
                not in data_object.metadata["spectrograms"][metadata["img_file"]][
                    "labels"
                ]
                or data_object.metadata["spectrograms"][metadata["img_file"]]["labels"][
                    "labelme"
                ]
                != label_metadata
            ):
                data_object.metadata["spectrograms"][metadata["img_file"]][
                    "labels"
                ].update({"labelme": label_metadata})
                data_object.write_sigmf_meta(data_object.metadata)

        else:
            spectrogram_metadata["labels"] = {"labelme": label_metadata}
            data_object.metadata["spectrograms"][
                metadata["img_file"]
            ] = spectrogram_metadata
            data_object.write_sigmf_meta(data_object.metadata)


def general_labels_to_sigmf(
    label_dir, label_type, metadata_dir, yolo_dataset_yaml=None
):
    # takes directory of labels
    # create Data object from associated sample file (creates sigmf-meta if doesn't exist)
    # populates sigmf-meta["spectrogams"] object
    # spectrogram should have (img_filename, sample_start, sample_count, nfft, augmentations)

    # LABELS
    # label_dir = Path(directory, "png")

    if "labelme" in label_type.lower():
        label_ext = ".json"
        label_type = "labelme"
    elif "yolo" in label_type.lower():
        if yolo_dataset_yaml is None:
            raise ValueError("Must define yolo_dataset_yaml when using Yolo dataset")
        label_ext = ".txt"
        label_type = "yolo"
    else:
        raise NotImplementedError

    label_files = [
        label_file
        for label_file in os.listdir(label_dir)
        if label_file.endswith(label_ext)
    ]

    # METADATA
    # if no metadata saved then must be parsed from images or generated in some way
    # metadata_dir = Path(directory, "metadata")
    metadata_files = os.listdir(metadata_dir)

    for label_filename in label_files:
        # get metadata
        metadata_filename = f"{os.path.splitext(label_filename)[0]}.json"
        if metadata_filename not in metadata_files:
            raise ValueError(
                f"Could not find matching metadata file {metadata_filename} for {label_filename}"
            )
        metadata = json.load(open(Path(metadata_dir, metadata_filename)))

        # create Data object
        data_object = Data(metadata["sample_file"]["filename"])

        # format spectrogram metadata
        spectrogram_metadata = {
            # "filename": metadata["img_file"],
            "sample_start": metadata["sample_start_idx"],
            "sample_count": metadata["mini_batch_size"],
            "nfft": metadata["nfft"],
            "augmentations": {
                "snr": metadata["snr"],
            },
        }

        if "labelme" in label_type.lower():
            label_metadata = json.load(open(Path(label_dir, label_filename)))
            data_object.labelme_to_sigmf(label_metadata, metadata["img_file"])
        elif "yolo" in label_type.lower():
            with open(Path(label_dir, label_filename)) as yolo_file:
                label_metadata = [line.rstrip() for line in yolo_file]
            with open(yolo_dataset_yaml, "r") as stream:
                dataset_yaml = yaml.safe_load(stream)
                class_labels = dataset_yaml["names"]

            data_object.yolo_to_sigmf(
                label_metadata, metadata["img_file"], class_labels
            )

        # Checks if spectrogram_metadata is a subset of any spectrogram in SigMF
        if (metadata["img_file"] in data_object.metadata["spectrograms"]) and (
            spectrogram_metadata.items()
            <= data_object.metadata["spectrograms"][metadata["img_file"]].items()
        ):
            if (
                "labels"
                not in data_object.metadata["spectrograms"][metadata["img_file"]]
            ):
                data_object.metadata["spectrograms"][metadata["img_file"]][
                    "labels"
                ] = {}

            if (
                label_type
                not in data_object.metadata["spectrograms"][metadata["img_file"]][
                    "labels"
                ]
                or data_object.metadata["spectrograms"][metadata["img_file"]]["labels"][
                    label_type
                ]
                != label_metadata
            ):
                data_object.metadata["spectrograms"][metadata["img_file"]][
                    "labels"
                ].update({label_type: label_metadata})
                data_object.write_sigmf_meta(data_object.metadata)

        else:
            spectrogram_metadata["labels"] = {label_type: label_metadata}
            data_object.metadata["spectrograms"][
                metadata["img_file"]
            ] = spectrogram_metadata
            data_object.write_sigmf_meta(data_object.metadata)


if __name__ == "__main__":
    # /Users/ltindall/data/gamutrf/gamutrf-arl/01_30_23/mini2/snr_noise_floor/

    directory = "/Users/ltindall/data_test/snr_noise_floor/"
    # images_to_sigmf(directory)
    # labels_to_sigmf(directory)
    general_labels_to_sigmf(
        directory + "png/YOLODataset/labels/train/",
        "yolo",
        directory + "metadata/",
        directory + "png/YOLODataset/dataset.yaml",
    )

import matplotlib.pyplot as plt
import numpy as np

import json
import os
import zstandard


from torchsig.utils.types import SignalCapture, SignalDescription, SignalData
from torchsig.utils.types import SignalCapture, SignalData
from torchsig.utils.dataset import SignalDataset
import torchsig.utils.reader as reader
import torchsig.utils.index as indexer
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
import numpy as np
import torch


def reader_from_zst(signal_capture: SignalCapture) -> SignalData:
    """
    Args:
        signal_capture:

    Returns:
        signal_data: SignalData object with meta-data parsed from sigMF file

    """
    with zstandard.ZstdDecompressor().stream_reader(
        open(signal_capture.absolute_path, "rb"), read_across_frames=True
    ) as file_object:
        file_object.seek(signal_capture.byte_offset)
        return SignalData(
            data=file_object.read(signal_capture.num_bytes),
            item_type=signal_capture.item_type,
            data_type=np.dtype(np.complex128)
            if signal_capture.is_complex
            else np.dtype(np.float64),
            signal_description=signal_capture.signal_description,
        )


class SigMFDataset(SignalDataset):
    """SigMFDataset is meant to make a mappable (index-able) dataset from
    a set of annotated sigmf files

    Args:
        root:
            Root file path to search recursively for files

        index_filter:
            Given an index, remove certain elements

        *\\*kwargs:**
            Keyword arguments

    """

    def __init__(
        self,
        root: str,
        index_filter: Optional[Callable[[Tuple[Any, SignalCapture]], bool]] = None,
        class_list: Optional[List[str]] = None,
        allowed_filetypes: Optional[List[str]] = [".sigmf-data"],
        **kwargs,
    ):
        super(SigMFDataset, self).__init__(**kwargs)
        self.class_list = class_list if class_list else []
        self.allowed_filetypes = allowed_filetypes
        self.index = self.indexer_from_sigmf_annotations(root)

        if index_filter:
            self.index = list(filter(index_filter, self.index))

    def get_data(self, signal_capture: SignalCapture) -> SignalData:
        if signal_capture.absolute_path.endswith(".sigmf-data"):
            return reader.reader_from_sigmf(signal_capture)
        elif signal_capture.absolute_path.endswith(".zst"):
            return reader_from_zst(signal_capture)
        else:
            raise ValueError(
                f"Could not read {signal_capture.absolute_path}. Check file type."
            )

    def __getitem__(self, item: int) -> Tuple[np.ndarray, Any]:  # type: ignore
        target, signal_capture = self.index[item]
        signal_data = self.get_data(signal_capture)

        if self.transform:
            signal_data = self.transform(signal_data)

        if self.target_transform:
            target = self.target_transform(target)

        return signal_data.iq_data, target  # type: ignore

    def __len__(self) -> int:
        return len(self.index)

    def indexer_from_sigmf_annotations(
        self, root: str
    ) -> List[Tuple[Any, SignalCapture]]:
        """An indexer the reads in the annotations from the sigmf-meta files in the provided directory

        Args:
            root:

        Returns:
            index: tuple of label, SignalCapture pairs

        """
        # go through directories and find files
        non_empty_dirs = [
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        ]
        non_empty_dirs.append(".")
        print(non_empty_dirs)
        # non_empty_dirs = [d for d in non_empty_dirs if os.listdir(os.path.join(root, d))]
        # print(non_empty_dirs)
        # Identify all files associated with each class
        index = []
        for dir_idx, dir_name in enumerate(non_empty_dirs):
            data_dir = os.path.join(root, dir_name)

            # Find files with allowed filetype
            proper_sigmf_files = list(
                filter(
                    lambda x: os.path.splitext(x)[1] in self.allowed_filetypes
                    and os.path.isfile(os.path.join(data_dir, x))
                    and os.path.isfile(
                        os.path.join(data_dir, f"{os.path.splitext(x)[0]}.sigmf-meta")
                    ),
                    os.listdir(data_dir),
                )
            )

            # Go through each file and create and index
            for f in proper_sigmf_files:
                index = index + self._parse_sigmf_annotations(os.path.join(data_dir, f))

        print(f"Class List: {self.class_list}")
        return index

    def _get_name_to_idx(self, name: str) -> int:
        try:
            idx = self.class_list.index(name)
        except ValueError:
            print(f"Adding {name} to class list")
            self.class_list.append(name)
            idx = self.class_list.index(name)
        return idx

    def _parse_sigmf_annotations(self, absolute_file_path: str) -> List[SignalCapture]:
        """
        Args:
            absolute_file_path: absolute file path of sigmf-data file for which to create Captures
            It will find the associated sigmf-meta file and parse the annotations

        Returns:
            signal_files:

        """

        meta_file_name = f"{os.path.splitext(absolute_file_path)[0]}.sigmf-meta"
        # meta_file_name = "{}{}".format(absolute_file_path.split("sigmf-data")[0], "sigmf-meta")
        meta = json.load(open(meta_file_name, "r"))
        item_type = indexer.SIGMF_DTYPE_MAP[meta["global"]["core:datatype"]]
        sample_size = item_type.itemsize * (
            2 if "c" in meta["global"]["core:datatype"] else 1
        )
        total_num_samples = os.path.getsize(absolute_file_path) // sample_size

        # It's quite common for there to be only a single "capture" in sigMF
        index = []
        if len(meta["captures"]) == 1:
            for annotation in meta["annotations"]:
                sample_start = annotation["core:sample_start"]
                sample_count = 4096  # annotation["core:sample_count"]
                signal_description = SignalDescription(
                    sample_rate=meta["global"]["core:sample_rate"],
                )
                signal_description.upper_frequency = annotation["core:freq_upper_edge"]
                signal_description.lower_frequency = annotation["core:freq_lower_edge"]

                label = annotation["core:label"]
                comment = annotation.get("core:comment", None)

                signal = SignalCapture(
                    absolute_path=absolute_file_path,
                    num_bytes=sample_size * sample_count,
                    byte_offset=sample_size * sample_start,
                    item_type=item_type,
                    is_complex=True
                    if "c" in meta["global"]["core:datatype"]
                    else False,
                    signal_description=signal_description,
                )
                index.append((self._get_name_to_idx(label), signal))
                # print(f"Signal {label}  {signal.num_bytes} {signal.byte_offset} {signal.item_type} {signal.is_complex} ")
        else:
            print(
                "Not Clear how we should handle the annotations when there is more than one capture"
            )
        # If there's more than one, we construct a list of captures
        return index

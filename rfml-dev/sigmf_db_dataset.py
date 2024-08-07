import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import lmdb
import numpy as np

from torchsig.datasets import conf
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.transforms import Identity
from torchsig.utils.types import SignalData

class SigMFDB:
    """The Official Sig53 dataset

    Args:
        root (string):
            Root directory of dataset. A folder will be created for the
            requested version of the dataset, an mdb file inside contains the
            data and labels.

        train (bool, optional):
            If True, constructs the corresponding training set, otherwise
            constructs the corresponding val set

        impaired (bool, optional):
            If True, will construct the impaired version of the dataset, with
            data passed through a seeded channel model

        eb_no (bool, optional):
            If True, will define SNR as Eb/No; If False, will define SNR as Es/No

        transform (callable, optional):
            A function/transform that takes in a complex64 ndarray and returns
            a transformed version

        target_transform (callable, optional):
            A function/transform that takes in the target class (int) and
            returns a transformed version

        use_signal_data (bool, optional):
            If True, data will be converted to SignalData objects as read in.
            Default: False. Sig53

    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_signal_data: bool = False,
    ):
        self.root = Path(root)
        self.use_signal_data = use_signal_data
        self.T = transform if transform else Identity()
        self.TT = target_transform if target_transform else Identity()

        # cfg: conf.Sig53Config = (
        #     "Sig53"  # type: ignore
        #     + ("Impaired" if impaired else "Clean")
        #     + ("EbNo" if (impaired and eb_no) else "")
        #     + ("Train" if train else "Val")
        #     + "Config"
        # )

        # cfg = getattr(conf, cfg)()  # type: ignore

        self.path = self.root #/ "data.mbd" #cfg.name


    def __open_lmdb(self):
        self.env = lmdb.Environment(
            str(self.path).encode(), map_size=int(1e12), max_dbs=2, lock=False
        )

        self.data_db = self.env.open_db(b"data")
        self.label_db = self.env.open_db(b"label")
        with self.env.begin(db=self.data_db) as data_txn:
            self.length = data_txn.stat()["entries"]

    def __len__(self) -> int:
        if not hasattr(self, "data_db"):
            self.__open_lmdb()
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Any]:
        if not hasattr(self, "data_db"):
            self.__open_lmdb()

        encoded_idx = pickle.dumps(idx)
        with self.env.begin(db=self.data_db) as data_txn:
            iq_data = pickle.loads(data_txn.get(encoded_idx)).numpy()

        with self.env.begin(db=self.label_db) as label_txn:
            label = pickle.loads(label_txn.get(encoded_idx))

        label = str("".join(label))
        #label = int(label.numpy())
        print(f"label: {label}")
        if self.use_signal_data:
            signal_desc = SignalDescription(
                class_name=label, #self._idx_to_name_dict[mod],
                #class_index=1, # mod,
                #snr=snr,
            )
            
            data: SignalData = SignalData(
                data=deepcopy(iq_data.tobytes()),
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[signal_desc],
            )
            data = self.T(data)  # type: ignore
            target = self.TT(data.signal_description)  # type: ignore
            assert data.iq_data is not None
            sig_iq_data: np.ndarray = data.iq_data
            return sig_iq_data, target

        np_data: np.ndarray = self.T(iq_data)  # type: ignore
        target = self.TT(label) #(self.TT(mod), snr)  # type: ignore

        return np_data, target

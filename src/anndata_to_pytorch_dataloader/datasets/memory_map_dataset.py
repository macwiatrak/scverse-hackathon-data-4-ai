# implementation taken from https://github.com/laminlabs/lamindb/blob/main/lamindb/dev/_mapped_dataset.py#L6
from collections import Counter
from typing import Any, Optional, Union

import h5py
import numpy as np
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset


class Registry:
    """Registry for storage backends."""

    # TODO: add storage backends from
    # https://github.com/laminlabs/lamindb/blob/main/lamindb/dev/storage/_backed_access.py

    def open(self, storage_backend: str, *args, **kwargs):
        """Open a storage backend."""
        pass


registry = Registry()

ArrayTypes = Union[csr_matrix, np.ndarray, np.memmap]
GroupTypes = [h5py.Group]


class MemoryMapDataset(Dataset):
    """Dataset for memory mapped files."""

    # TODO: add support for multiple file types
    # TODO: test the dataset
    # TODO: do speed comparisons against AnnLoader and AnnTorchLoader from scvi-tools
    # TODO: add support for reading and using multiple files

    def __init__(
        self,
        input_file_path: str,
        label_keys: Optional[Union[str, list[str]]] = None,
        encode_labels: bool = True,
    ):
        super().__init__()

        self.conn, self.storage = registry.open("h5py", input_file_path)

        X = self.storage["X"]
        if isinstance(X, ArrayTypes):  # type: ignore
            self.n_obs = X.shape[0]
        else:
            self.n_obs = X.attrs["shape"][0]

        self.indices = np.arange(self.n_obs)

        self.encode_labels = encode_labels
        if isinstance(label_keys, str):
            label_keys = [label_keys]
        self.label_keys = label_keys
        if self.label_keys is not None:
            if self.encode_labels:
                self.encoders = []
                for label in self.label_keys:
                    cats = self.get_merged_categories(label)
                    self.encoders.append({cat: i for i, cat in enumerate(cats)})

        self._closed = False

    def __len__(self):
        return self.n_obs

    def __getitem__(self, idx):
        obs_idx = self.indices[idx]
        out = [self.get_data_idx(self.storage, obs_idx)]
        if self.label_keys is not None:
            for i, label in enumerate(self.label_keys):
                label_idx = self.get_label_idx(self.storage, obs_idx, label)
                if self.encode_labels:
                    label_idx = self.encoders[i][label_idx]
                out.append(label_idx)
        return out

    def get_data_idx(self, storage: StorageType, idx: int, layer_key: Optional[str] = None):  # type: ignore # noqa
        """Get the index for the data."""
        layer = storage["X"] if layer_key is None else storage["layers"][layer_key]  # type: ignore
        if isinstance(layer, ArrayTypes):  # type: ignore
            return layer[idx]
        else:  # assume csr_matrix here
            data = layer["data"]
            indices = layer["indices"]
            indptr = layer["indptr"]
            s = slice(*(indptr[idx : idx + 2]))
            layer_idx = np.zeros(layer.attrs["shape"][1])
            layer_idx[indices[s]] = data[s]
            return layer_idx

    def get_label_idx(self, storage: StorageType, idx: int, label_key: str):  # type: ignore # noqa
        """Get the index for the label by key."""
        obs = storage["obs"]  # type: ignore
        # how backwards compatible do we want to be here actually?
        if isinstance(obs, ArrayTypes):  # type: ignore
            label = obs[idx][obs.dtype.names.index(label_key)]
        else:
            labels = obs[label_key]
            if isinstance(labels, ArrayTypes):  # type: ignore
                label = labels[idx]
            else:
                label = labels["codes"][idx]

        cats = self.get_categories(storage, label_key)
        if cats is not None:
            label = cats[label]
        if isinstance(label, bytes):
            label = label.decode("utf-8")
        return label

    def get_label_weights(self, label_key: str):
        """Get all weights for a given label key."""
        labels = self.get_merged_labels(label_key)
        counter = Counter(labels)  # type: ignore
        weights = np.array([counter[label] for label in labels]) / len(labels)
        return weights

    def get_merged_labels(self, label_key: str):
        """Get merged labels."""
        labels_merge = []
        decode = np.frompyfunc(lambda x: x.decode("utf-8"), 1, 1)

        codes = self.get_codes(self.storage, label_key)
        labels = decode(codes) if isinstance(codes[0], bytes) else codes
        cats = self.get_categories(self.storage, label_key)
        if cats is not None:
            cats = decode(cats) if isinstance(cats[0], bytes) else cats
            labels = cats[labels]
        labels_merge.append(labels)
        return np.hstack(labels_merge)

    def get_merged_categories(self, label_key: str):
        """Get merged categories."""
        cats_merge = set()
        decode = np.frompyfunc(lambda x: x.decode("utf-8"), 1, 1)

        cats = self.get_categories(self.storage, label_key)
        if cats is not None:
            cats = decode(cats) if isinstance(cats[0], bytes) else cats
            cats_merge.update(cats)
        else:
            codes = self.get_codes(self.storage, label_key)
            codes = decode(codes) if isinstance(codes[0], bytes) else codes
            cats_merge.update(codes)
        return cats_merge

    def get_categories(self, storage: Any, label_key: str):  # type: ignore
        """Get categories."""
        obs = storage["obs"]  # type: ignore
        if isinstance(obs, ArrayTypes):  # type: ignore
            cat_key_uns = f"{label_key}_categories"
            if cat_key_uns in storage["uns"]:  # type: ignore
                return storage["uns"][cat_key_uns]  # type: ignore
            else:
                return None
        else:
            if "__categories" in obs:
                cats = obs["__categories"]
                if label_key in cats:
                    return cats[label_key]
                else:
                    return None
            labels = obs[label_key]
            if isinstance(labels, GroupTypes):  # type: ignore
                if "categories" in labels:
                    return labels["categories"]
                else:
                    return None
            else:
                if "categories" in labels.attrs:
                    return labels.attrs["categories"]
                else:
                    return None
        return None

    def get_codes(self, storage: Any, label_key: str):  # type: ignore
        """Get codes."""
        obs = storage["obs"]  # type: ignore
        if isinstance(obs, ArrayTypes):  # type: ignore
            label = obs[label_key]
        else:
            label = obs[label_key]
            if isinstance(label, ArrayTypes):  # type: ignore
                return label[...]
            else:
                return label["codes"][...]

    def close(self):
        """Close connection to array streaming backend."""
        if hasattr(self.storage, "close"):
            self.storage.close()

        if hasattr(self.conn, "close"):
            self.conn.close()
        self._closed = True

    @property
    def closed(self):
        """Return True if the connection to the array streaming backend is closed."""
        return self._closed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

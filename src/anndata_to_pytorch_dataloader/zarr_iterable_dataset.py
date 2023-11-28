import math
from typing import Callable

import pandas as pd
import torch
import zarr
from torch.utils.data import DataLoader, IterableDataset


def worker_init_fn():
    """Function to initialize the random seed for each worker."""
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


class ZarrDataset(IterableDataset):
    """Iterable dataset for zarr files."""

    def __init__(
        self,
        zarr_file_path: str,
        metadata_file_path: str = None,
        transform_fn: Callable = None,
        start: int = None,
        end: int = None,
    ):
        super().__init__()

        self.metadata = None
        if metadata_file_path is not None:
            self.metadata = pd.read_parquet(metadata_file_path)
        self.transform_fn = transform_fn

        store = zarr.DirectoryStore(zarr_file_path)
        self.zarr_array = zarr.open(store, mode="r")

        start = 0 if start is None else start
        end = self.zarr_array.shape[0] if end is None else end
        assert end > start

        self.start = start
        self.end = end

    def __iter__(self):
        iter = self.zarr_array.islice(self.start, self.end)

        if self.metadata is not None:
            iter = zip(iter, self.metadata.iloc[self.start : self.end].iterrows())

        for item in iter:
            yield self.transform_fn(item)


def main():
    """Example of using the ZarrDataset."""
    n_obs = 10000
    n_genes = 36000
    chunks = 1000

    z = zarr.zeros((n_obs, n_genes), chunks=chunks, dtype="i4")
    zarr.save("/tmp/example.zarr", z)

    dataset = ZarrDataset(
        zarr_file_path="/tmp/example.zarr",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        # TODO: add collate_fn
    )

    for batch in dataloader:
        print(batch)

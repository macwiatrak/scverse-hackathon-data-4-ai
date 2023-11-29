from pytorch_lightning import LightningDataModule
import anndata as ad
import scanpy as sc
from typing import List
from functools import partial

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np


class SimpleDataModule(LightningDataModule):
    """
    Data module for data created from adata. Uses simple data loaders but
    AnnDataset in the background. Suited for when you have one or more datafields with
    the same shape or length. Always uses a single dataloader.
    """

    def __init__(
        self,
        train_adata,
        test_adata=None,
        val_adata=None,
        encoders=None,
        batch_size=32,
        shuffle: bool = True,
        num_workers=1,
    ):
        super(SimpleDataModule, self).__init__()

        if val_adata is not None:
            self.val_data = val_adata
        if test_adata is not None:
            self.test_data = val_adata

        self.train_data = train_adata
        self.encoders = encoders

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def train_dataloader(self):
        return ad.experimental.AnnLoader(
            self.train_data,
            convert=self.encoders,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return ad.experimental.AnnLoader(
            self.val_data,
            convert=self.encoders,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return ad.experimental.AnnLoader(
            self.test_data,
            convert=self.encoders,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()


def setup_ad_anndata_module(
    adata: ad.AnnData,
    train_frac: float = 0.0,
    val_frac: float = 0.0,
    test_frac: float = 0.0,
    include_exprs: bool = True,
    obs_fields: List[str] = None,
    var_fields: List[str] = None,
    layer_fields: List[str] = None,
    batch_size: int = 512,
    shuffle: bool = True,
    num_workers: int = 1,
    random_state: int = 11,
) -> SimpleDataModule:
    anndata = adata.copy()

    # Transform categorical data
    encoder_study = OneHotEncoder(sparse=False, dtype=np.float32)
    encoder_study.fit(adata.obs["study"].to_numpy()[:, None])
    encoder_celltype = LabelEncoder()
    encoder_celltype.fit(adata.obs["cell_type"])

    def study_encoder(s):
        return encoder_study.transform(s.to_numpy()[:, None])

    def cell_type_encoder(s):
        return encoder_celltype.transform

    encoders = {
        "obs": {
            "study": study_encoder,
            "cell_type": cell_type_encoder,
        }
    }

    # Split into train val test
    if train_frac + val_frac + test_frac != 1:
        raise Exception("train, test and validation splits should add up to 1.")
    else:
        train_adata = sc.pp.subsample(adata, fraction=train_frac, copy=True)
        rest = adata[~(adata.obs_names.isin(train_adata.obs_names)), :]
        val_adata = sc.pp.subsample(rest, fraction=val_frac / (test_frac + val_frac), copy=True)
        test_adata = rest[~(rest.obs_names.isin(val_adata.obs_names)), :]

    return SimpleDataModule(train_adata=train_adata, val_adata=val_adata, test_adata=test_adata, encoders=encoders)

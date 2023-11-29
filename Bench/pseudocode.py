from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import numpy as np
from typing import Union, List
import scanpy as sc


class AnnDataModule(LightningDataModule):
    def __init__(self, train, val, test, trained_transformer=None, batch_size=32, num_workers=1):
        super(AnnDataModule, self).__init__()
        self.train = train
        self.val = val
        self.test = test
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()


class AnnDataSet(Dataset):
    def __init__(self, X, y):
        """
        Parameters
        ----------

        """
        self.X = X
        self.y = y

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return self.X.shape[0]


def setup_anndata_datamodule(
    adata,
    train_frac: float = 0.0,
    val_frac: float = 0.0,
    test_frac: float = 0.0,
    include_exprs: bool = True,
    obs_fields: List[str] = None,
    var_fields: List[str] = None,
    layers_fields: List[str] = None,
    target: dict[str, str] = None,
    batch_size=512,
    num_workers=1,
    cofactor_transform=True,
    random_state=11,
):
    anndata = adata.copy()

    # Split into train val test
    if train_frac + val_frac + test_frac > 0:
        adata_list = split_adata(adata=anndata, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac)
    else:
        adata_list = [anndata]

    # Get features and target

    # Optional
    # Train scaler on training data

    dataset_list = [AnnDatasetFromAnnData(adt) for adt in adata_list]

    ds_train = AnnDataSet(X=X_train, y=y_train, scaler=trained_scaler)

    ds_val = AnnDataSet(X=X_val, y=y_val, scaler=trained_scaler)

    ds_test = AnnDataSet(X=X_test, y=y_test, scaler=trained_scaler)

    return AnnDataModule(
        datasets=dataset_list,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def split_adata(adata, train_frac, val_frac, test_frac):
    full_length = len(adata)

    if train_frac + val_frac + test_frac == 1:
        train_adata = sc.pp.subsample(adata, fraction=train_frac, copy=True)
        rest = adata[adata.obs_names.isin(train_adata.obs_names), :]

        val_adata = sc.pp.subsample(rest, n_obs=int(val_frac * full_length), copy=True)
        test_adata = rest[rest.obs_names.isin(val_adata.obs_names), :]

        return [train_adata, val_adata, test_adata]

    elif (
        (train_frac + val_frac == 1)
        or (train_frac + test_frac == 1)
        or (train_frac + val_frac + test_frac == train_frac)
    ):
        train_adata = sc.pp.subsample(adata, fraction=train_frac, copy=True)
        rest_adata = adata[adata.obs_names.isin(train_adata.obs_names), :]

        return [train_adata, rest_adata]

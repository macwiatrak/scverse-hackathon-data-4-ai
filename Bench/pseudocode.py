from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import numpy as np
from typing import Union, List


class AnnDataModule(LightningDataModule):
    def __init__(self, train, val, test, trained_transformer=None, batch_size=32, num_workers=1):
        super(AnnDataModule, self).__init__()
        self.train = train
        self.val = val
        self.test = test
        self.transformer = trained_transformer
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

    def state_dict(self):
        # track whatever you want here
        # will be saved upon checkpointing
        state = {"trained_transformer": self.scaler}
        return state

    def load_state_dict(self, state_dict):
        # restore the state based on what you tracked in (def state_dict)
        self.scaler = state_dict["trained_transformer"]


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
    batch_size=512,
    num_workers=1,
    cofactor_transform=True,
    random_state=11,
):
    # Split into train val test

    # Get features and target

    # Optional
    # Train scaler on training data

    # Pass to each get_cytof_dataset function

    ds_train = AnnDataSet(X=X_train, y=y_train, scaler=trained_scaler)

    ds_val = AnnDataSet(X=X_val, y=y_val, scaler=trained_scaler)

    ds_test = AnnDataSet(X=X_test, y=y_test, scaler=trained_scaler)

    return AnnDataModule(
        train=ds_train,
        val=ds_val,
        test=ds_test,
        trained_scaler=trained_scaler,
        batch_size=batch_size,
        num_workers=num_workers,
    )

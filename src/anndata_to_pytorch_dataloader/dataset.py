import numpy as np
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from pytorch_lightning import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader

from typing import List, Tuple


def preprocess_anndata(adata: AnnData):
    """Preprocess anndata for input into the model."""
    adata.obs["size_factors"] = adata.X.sum(1)

    encoder_study = OneHotEncoder(sparse=False, dtype=np.float32)
    encoder_study.fit(adata.obs["study"].to_numpy()[:, None])

    encoder_celltype = LabelEncoder()
    encoder_celltype.fit(adata.obs["cell_type"])

    # TODO: transform the data


class SimpleDataset(Dataset):
    def __init__(self, X):
        """
        Parameters
        ----------
        X:
            Data
        """
        self.X = X
        self.X = Tensor(np.array(self.X))

    def __getitem__(self, i):
        return self.X[i]

    def __len__(self):
        return self.X.shape[0]


class AnnDataset(Dataset):
    def __init__(self, data_list, data_names):
        """
        Parameters
        ----------
        X:
            If having multiple data field this gives you a
            dictonary. Otherwise returns just the single datafield
        """
        self.X = [Tensor(np.array(d)) for d in data_list]
        self.data_names = data_names
        self.n_data = len(self.data_names)

        if "X" in data_names:
            self.length_data_name = "X"
        elif sum([n.startswith("obs") for n in self.data_names]) > 0:
            self.length_data_name = self.data_names[[n.startswith("obs") for n in self.data_names]][0]
        else:
            raise (("Data doesn't contain fields along the `obs` axis,", " can't split into different sets"))

    def __getitem__(self, i):
        if self.n_data == 1:
            return self.X[i]
        else:
            item = {self.data_names[n]: self.X[n][i] for n in range(self.n_data)}

            return item

    def __len__(self):
        return self.X[self.data_names.index(self.length_data_name)].shape[0]


class AnnDataModule(LightningDataModule):
    """
    Data module for data created from adata. Works by combining dataloaders.
    This means that this module is suited for datafields or datasets with different
    lengths or sizes. The CombinedDataloader returns a dictonary in the case of
    multiple data fields. In the case of one datafield, a single dataloader is used.
    """

    def __init__(self, splitted_data_lists, data_names, batch_size=32, shuffle: bool = True, num_workers=1):
        super(AnnDataModule, self).__init__()

        self.test_data_l = splitted_data_lists[2]
        self.val_data_l = splitted_data_lists[1]
        self.train_data_l = splitted_data_lists[0]

        self.data_names = data_names
        self.n_data = len(data_names)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def train_dataloader(self):
        if self.n_data > 1:
            dataloaders = CombinedLoader(
                {
                    self.data_names[i]: DataLoader(
                        self.train_data_l[i],
                        batch_size=self.batch_size,
                        shuffle=self.shuffle,
                        num_workers=self.num_workers,
                    )
                    for i in range(self.n_data)
                }
            )
            return dataloaders
        else:
            return DataLoader(
                self.train_data_l[0],
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
            )

    def val_dataloader(self):
        if self.n_data > 1:
            dataloaders = CombinedLoader(
                {
                    self.data_names[i]: DataLoader(
                        self.val_data_l[i],
                        batch_size=self.batch_size,
                        shuffle=self.shuffle,
                        num_workers=self.num_workers,
                    )
                    for i in range(self.n_data)
                }
            )
            return dataloaders
        else:
            return DataLoader(
                self.val_data_l[0],
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
            )

    def test_dataloader(self):
        if self.n_data > 1:
            dataloaders = CombinedLoader(
                {
                    self.data_names[i]: DataLoader(
                        self.test_data_l[i],
                        batch_size=self.batch_size,
                        shuffle=self.shuffle,
                        num_workers=self.num_workers,
                    )
                    for i in range(self.n_data)
                }
            )
            return dataloaders
        else:
            return DataLoader(
                self.test_data_l[0],
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
            )

    def predict_dataloader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()


class SimpleDataModule(LightningDataModule):
    """
    Data module for data created from adata. Uses simple data loaders but
    AnnDataset in the background. Suited for when you have one or more datafields with
    the same shape or length. Always uses a single dataloader.
    """

    def __init__(self, data_list: List, batch_size=32, shuffle: bool = True, num_workers=1):
        super(SimpleDataModule, self).__init__()

        self.test_data = data_list[2]
        self.val_data = data_list[1]
        self.train_data = data_list[0]

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()


def setup_anndata_datamodule(
    adata: AnnData,
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
) -> AnnDataModule:
    """
    Use this function to setup AnnData with a combined dataloader,
    which can handle different dataset lengths/sizes between the different
    data fields.
    Parameters
    -----------
        adata:
    """

    anndata = adata.copy()

    # Extract data
    data_list, data_names = extract_from_adata(
        adata=anndata,
        include_exprs=include_exprs,
        layer_fields=layer_fields,
        obs_fields=obs_fields,
        var_fields=var_fields,
    )

    # Split into train val test
    if train_frac + val_frac + test_frac > 0:
        splitted_data_lists = split_data(
            data_list=data_list,
            data_names=data_names,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            random_state=random_state,
        )
    else:
        splitted_data_lists = [data_list]

    # TODO: add transformations of training data
    # and apply to test and val data
    # pass trained transformer or parameters to adatamodule

    for i, data_l in enumerate(splitted_data_lists):
        splitted_data_lists[i] = [SimpleDataset(d) for d in data_l]

    return AnnDataModule(
        splitted_data_lists=splitted_data_lists,
        data_names=data_names,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def setup_simple_datamodule(
    adata: AnnData,
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
) -> AnnDataModule:
    anndata = adata.copy()

    # Extract data
    data_list, data_names = extract_from_adata(
        adata=anndata,
        include_exprs=include_exprs,
        layer_fields=layer_fields,
        obs_fields=obs_fields,
        var_fields=var_fields,
    )

    # Split into train val test
    if train_frac + val_frac + test_frac > 0:
        splitted_data_lists = split_data(
            data_list=data_list,
            data_names=data_names,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            random_state=random_state,
        )
    else:
        splitted_data_lists = [data_list]

    # TODO: add transformations of training data
    # and apply to test and val data
    # pass trained transformer or parameters to adatamodule

    data_list = []
    for data_l in splitted_data_lists:
        data_list.append(AnnDataset(data_list=data_l, data_names=data_names))

    return SimpleDataModule(
        data_list=data_list,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def split_data(
    data_list: list, data_names: list, train_frac: float, val_frac: float, test_frac: float, random_state: int
) -> List[List]:
    """Always returns train, test and validation sets"""

    if "X" in data_names:
        full_length = data_list[data_names.index("X")].shape[0]
    elif sum([n.startswith("obs") for n in data_names]) > 0:
        name = data_names[[n.startswith("obs") for n in data_names]][0]
        full_length = len(data_list[data_names.index(name)])
    else:
        raise (("Data doesn't contain fields along the `obs` axis,", " can't split into different sets"))

    # All three provided should add up to one
    if train_frac + val_frac + test_frac == 1:
        train_idx, test_val_idx = train_test_split(range(full_length), train_size=train_frac, random_state=random_state)
        val_idx, test_idx = train_test_split(test_val_idx, test_size=test_frac / (test_frac + val_frac))
    # Only train frac provided
    elif train_frac + val_frac + test_frac == train_frac:
        train_idx, test_val_idx = train_test_split(range(full_length), train_size=train_frac, random_state=random_state)
        val_idx, test_idx = train_test_split(test_val_idx, test_size=0.5)
    # No test frac
    elif test_frac == 0 & train_frac != 0 and val_frac != 0:
        train_idx, test_val_idx = train_test_split(range(full_length), train_size=train_frac, random_state=random_state)
        val_idx, test_idx = train_test_split(test_val_idx, test_size=val_frac / (1 - train_frac + val_frac))
    # No val frac
    elif test_frac != 0 & train_frac != 0 and val_frac == 0:
        train_idx, test_val_idx = train_test_split(range(full_length), train_size=train_frac, random_state=random_state)
        test_idx, val_idx = train_test_split(test_val_idx, test_size=test_frac / (1 - train_frac + test_frac))
    else:
        raise (("Please supple valid train, test and validation fractions. With at least a training fraction.",))
    index_list = [train_idx, val_idx, test_val_idx]

    train_data = []
    val_data = []
    test_data = []
    for i, data in enumerate(data_list):
        if (data_names[i] == "X") or (data_names[i].startswith("obs")):
            train_data.append(data[index_list[0]])
            val_data.append(data[index_list[1]])
            test_data.append(data[index_list[2]])
        else:
            train_data.append(data)
            val_data.append(data)
            test_data.append(data)
    return [train_data, val_data, test_data]


def extract_from_adata(
    adata: AnnData, include_exprs: bool, layer_fields: List[str], obs_fields: List[str], var_fields: list[str]
) -> Tuple[list, list]:
    data_list = []
    data_names = []
    if include_exprs:
        expr = adata.X
        if issparse(expr):
            expr = expr.toarray()
        data_list.append(expr)
        data_names.append("X")
    if layer_fields is not None:
        for field in layer_fields:
            data_list.append(np.array(adata.obs[field]))
            data_names.append(f"layer_{field}")
    if obs_fields is not None:
        for field in obs_fields:
            data_list.append(np.array(adata.obs[field]))
            data_names.append(f"obs_{field}")
    if var_fields is not None:
        for field in var_fields:
            data_list.append(np.array(adata.var[field]))
            data_names.append(f"var_{field}")

    return data_list, data_names

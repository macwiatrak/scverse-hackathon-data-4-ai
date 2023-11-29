import numpy as np
from anndata import AnnData
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import Dataset


def preprocess_anndata(adata: AnnData):
    """Preprocess anndata for input into the model."""
    adata.obs["size_factors"] = adata.X.sum(1)

    encoder_study = OneHotEncoder(sparse=False, dtype=np.float32)
    encoder_study.fit(adata.obs["study"].to_numpy()[:, None])

    encoder_celltype = LabelEncoder()
    encoder_celltype.fit(adata.obs["cell_type"])

    # TODO: transform the data


class AnnDatasetFromAnnData(Dataset):
    """Extension of torch dataset to get tensors from anndata."""

    def __init__(
        self,
        adata: AnnData,
        obs_fields: list[str] = None,
        var_fields: list[str] = None,
    ):
        self.X = adata.X

        self.obs_fields = obs_fields
        self.var_fields = var_fields

        if obs_fields is not None:
            self.obs = adata.obs[obs_fields]

        if var_fields is not None:
            self.var = adata.var[var_fields]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = self.X[idx]

        if self.obs_fields is not None:
            # TODO: add the obs fields to the sample
            pass

        if self.var_fields is not None:
            # TODO: add the var fields to the sample
            pass
        return sample

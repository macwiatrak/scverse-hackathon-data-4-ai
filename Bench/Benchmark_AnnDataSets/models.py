import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, out_dim):
        super().__init__()

        modules = []
        for in_size, out_size in zip([input_dim] + hidden_dims, hidden_dims):
            modules.append(nn.Linear(in_size, out_size))
            modules.append(nn.LayerNorm(out_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p=0.05))
        modules.append(nn.Linear(hidden_dims[-1], out_dim))
        self.fc = nn.Sequential(*modules)

    def forward(self, *inputs):
        input_cat = torch.cat(inputs, dim=-1)
        return self.fc(input_cat)


class scLightning(pl.LightningModule):
    """
    Pytorch lightning implementation of basic cell type classifier
    """

    def __init__(self, n_vars, n_classes, feature_var: str, label_var: str):
        super().__init__()

        self.n_classes = n_classes
        self.feature_name = feature_var
        self.label_name = label_var

        self.model = MLP(input_dim=n_vars, hidden_dims=[128, 64, 32], out_dim=self.n_classes)

        for stage in ["train", "test", "val"]:
            setattr(self, f"{stage}_loss", [])
            setattr(self, f"{stage}_label_true", [])
            setattr(self, f"{stage}_label_pred", [])

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x = batch[self.feature_name]
        y = batch[self.label_name].long()

        out = self(x)
        self.train_label_true.append(y)
        self.train_label_pred.append(out)

        loss = F.cross_entropy(out, y)

        self.train_loss.append(loss)
        self.log(f"train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.feature_name]
        y = batch[self.label_name].long()

        out = self(x)

        self.val_label_true.append(y)
        self.val_label_pred.append(out)

        loss = F.cross_entropy(out, y)

        self.val_loss.append(loss)
        self.log(f"val/loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x = batch[self.feature_name]
        y = batch[self.label_name].long()

        out = self(x)
        self.test_label_true.append(y)
        self.test_label_pred.append(out)

        loss = F.cross_entropy(out, y)

        self.test_loss.append(loss)
        self.log(f"test/loss", loss, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        self.end_epoch(stage="train")

    def on_validation_epoch_end(self):
        self.end_epoch(stage="val")

    def on_test_epoch_end(self):
        self.end_epoch(stage="test")

    def end_epoch(self, stage):
        epoch_loss = getattr(self, f"{stage}_loss")
        epoch_loss = torch.stack(epoch_loss).sum()

        epoch_label_true = getattr(self, f"{stage}_label_true")
        epoch_label_true = torch.cat(epoch_label_true)

        epoch_label_pred = getattr(self, f"{stage}_label_pred")
        epoch_label_pred = torch.cat(epoch_label_pred)
        epoch_label_pred = torch.argmax(epoch_label_pred, dim=-1)

        # calculate accuracy
        epoch_acc = (epoch_label_true == epoch_label_pred).float().mean()
        self.log(f"{stage}/acc", epoch_acc, prog_bar=True)

        # Reset
        setattr(self, f"{stage}_loss", [])
        setattr(self, f"{stage}_label_true", [])
        setattr(self, f"{stage}_label_pred", [])

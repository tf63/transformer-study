import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class TransformerLightning(pl.LightningModule):

    def __init__(self, model, lr, dropout=0.1):
        super().__init__()
        self.model = model
        self.lr = lr
        self.dropot = dropout

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _calculate_loss(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

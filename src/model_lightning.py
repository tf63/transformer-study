import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class TransformerLightning(pl.LightningModule):

    def __init__(self, model, lr, dec_vocab_size, mask_size):
        super().__init__()
        self.model = model
        self.lr = lr
        self.dec_vocab_size = dec_vocab_size
        self.mask_size = mask_size

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _calculate_loss(self, batch):
        x, dec_input, target = batch
        mask = nn.Transformer.generate_square_subsequent_mask(self.mask_size).to(self.device)  # マスクの作成

        dec_output = self.model(x, dec_input, mask)
        target = F.one_hot(target, self.dec_vocab_size).to(torch.float32)
        loss = F.cross_entropy(target=target, input=dec_output)

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

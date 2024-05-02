import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class TransformerLightning(pl.LightningModule):
    """
    TransformerのPyTorch Lightningモデル
    pl.LightningModuleをオーバーライドすることでPyTorchの学習を実装する
    """

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

        # マスクを作成
        mask = nn.Transformer.generate_square_subsequent_mask(self.mask_size).to(self.device) 

        # モデルへ入力
        dec_output = self.model(x, dec_input, mask)

        # 損失を計算
        target = F.one_hot(target, self.dec_vocab_size).to(torch.float32)
        loss = F.cross_entropy(target=target, input=dec_output)

        return loss

    def training_step(self, batch, batch_idx):
        """trainステップ Trainer.fit(*)で呼ばれる この関数でlossを返すとbackwardされるようになっている"""
        loss = self._calculate_loss(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """validationステップ  Trainer.fit(*)で呼ばれる"""
        loss = self._calculate_loss(batch)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        """testステップ  Trainer.test(*)で呼ばれる"""
        loss = self._calculate_loss(batch)
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

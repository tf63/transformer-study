from functools import partial
import click

from torch.utils.data import DataLoader

import pytorch_lightning
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger

from src.model import Transformer
from src.model_lightning import TransformerLightning
from src.dataset import ReverseDataset


@click.command()
@click.option("--accelerator", default=None, help="accelerator for training [gpu|cpu|tpu|ipu|None (default)]")
@click.option("--devices", default=None, help="number of devices (default: None)")
@click.option("--lr", default=5e-4, help="learning rate (hparams)")
def main(accelerator, devices, lr):

    # setting
    pytorch_lightning.seed_everything(42)

    # dataloaderを作成
    num_categories = 10
    seq_len = 16
    dataset = partial(ReverseDataset, num_categories, seq_len)
    train_loader = DataLoader(dataset(50000), batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(dataset(1000), batch_size=128)
    test_loader = DataLoader(dataset(10000), batch_size=128)

    # modelを作成
    device = "cuda" if devices is not None else "cpu"
    model = Transformer(
        enc_vocab_size=num_categories,
        dec_vocab_size=num_categories,
        dim=32,
        head_num=1,
        device=device).to(device)
    model_lightning = TransformerLightning(model=model, lr=lr, dec_vocab_size=num_categories, mask_size=seq_len - 1)

    # loggerを作成
    wandb_logger = WandbLogger(project="transformer-study", save_dir="logs/")

    # Trainerを作成
    trainer = Trainer(logger=wandb_logger,
                      devices=devices,
                      accelerator=accelerator,
                      max_epochs=1000)

    # Train
    trainer.fit(model_lightning, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test
    trainer.test(model_lightning, test_loader)


if __name__ == "__main__":
    main()

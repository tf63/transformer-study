from functools import partial
import click

import torch
from torch.utils.data import DataLoader

import pytorch_lightning
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model import Transformer
from src.model_lightning import TransformerLightning
from src.dataset import ReverseDataset


@click.command()
@click.option("--accelerator", default="gpu", help="accelerator for training [gpu|cpu|tpu|ipu|None] (default: gpu)")
@click.option("--devices", default="1", help="number of devices (default: 1)")
@click.option("--lr", default=0.01, help="learning rate")
@click.option("--max_epochs", default=100, help="epoch")
@click.option("--batch_size", default=128, help="batch size")
@click.option("--num_heads", default=1, help="Headの数")
@click.option("--dim", default=32, help="embedding dimension")
@click.option("--num_categories", default=10, help="vocab (今回は0 ~ 9なので10個)")
@click.option("--seq_len", default=16, help="系列長")
@click.option("--debug", is_flag=True, help="デバックモードで実行")
def main(accelerator, devices, lr, max_epochs, num_heads, dim, batch_size, num_categories, seq_len, debug):
    """長さseq_lenの数列を逆順に変換するタスクをTransformerで学習する"""

    # setting
    torch.set_float32_matmul_precision("high")
    pytorch_lightning.seed_everything(42)
    exp_name = f"head{num_heads}-dim{dim}-lr{lr}"
    device = "cuda" if devices is not None else "cpu"
    config = click.get_current_context().params

    # dataloaderを作成
    dataset = partial(ReverseDataset, num_categories, seq_len)
    train_loader = DataLoader(dataset(50000), batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=16)
    val_loader = DataLoader(dataset(1000), batch_size=batch_size, num_workers=16)
    test_loader = DataLoader(dataset(10000), batch_size=batch_size, num_workers=16)

    # modelを作成
    model = Transformer(device=device,
                        enc_vocab_size=num_categories,
                        dec_vocab_size=num_categories,
                        dim=dim,
                        num_heads=num_heads).to(device)
    model_lightning = TransformerLightning(model=model,
                                           lr=lr,
                                           dec_vocab_size=num_categories,
                                           mask_size=seq_len - 1)

    # loggerを作成
    wandb_logger = WandbLogger(project="transformer-study",
                               name=exp_name,
                               save_dir="logs/",
                               tags=["debug" if debug else "run"],
                               save_code=True)
    wandb_logger.log_hyperparams(config)
    checkpoint_callback = ModelCheckpoint(dirpath=f"ckpts/{exp_name}",
                                          monitor="val/loss_epoch",
                                          mode="min",
                                          save_top_k=10,
                                          filename="{epoch}")

    # Trainerを作成
    trainer = Trainer(logger=wandb_logger,
                      devices=devices,
                      accelerator=accelerator,
                      deterministic=False,
                      max_epochs=max_epochs,
                      callbacks=[checkpoint_callback])

    # Train
    trainer.fit(model_lightning, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test
    trainer.test(model_lightning, test_loader)


if __name__ == "__main__":
    main()

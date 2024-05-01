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
from src.dataset import SNDataset


@click.command()
@click.option("--accelerator", default="gpu", help="accelerator for training [gpu|cpu|tpu|ipu|None] (default: gpu)")
@click.option("--devices", default="1", help="number of devices (default: 1)")
@click.option("--lr", default=0.01, help="learning rate")
@click.option("--max_epochs", default=100, help="epoch")
@click.option("--num_datas", default=50000, help="data数")
@click.option("--batch_size", default=128, help="batch size")
@click.option("--num_heads", default=1, help="Headの数")
@click.option("--dim", default=32, help="embedding dimension")
@click.option("--num_categories", default=10, help="vocab (今回は0 ~ 9)")
@click.option("--seq_len", default=16, help="系列長")
@click.option("--debug", is_flag=True, help="デバックモードで実行")
def main(accelerator, devices, lr, max_epochs, num_datas, num_heads, dim, batch_size, num_categories, seq_len, debug):
    """2つの数字からその間の連番を作成するタスクをTransformerで学習する"""

    # setting
    torch.set_float32_matmul_precision("high")
    pytorch_lightning.seed_everything(42)
    exp_name = f"sn-data-{num_datas}-head{num_heads}-dim{dim}-lr{lr}"
    device = "cuda" if devices is not None else "cpu"
    config = click.get_current_context().params
    vocab_size = num_categories + 4  # 0 ~ 9 + 開始/終了/余白タグ と 偶数にするために+1
    assert seq_len > vocab_size, "今回はseq_lenがvocab_sizeより大きいことを想定"

    # dataloaderを作成
    dataset = partial(SNDataset, num_categories, seq_len)
    train_loader = DataLoader(dataset(num_datas), batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(dataset(5000), batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(dataset(10000), batch_size=batch_size, num_workers=4)

    # modelを作成
    model = Transformer(device=device,
                        enc_vocab_size=vocab_size,
                        dec_vocab_size=vocab_size,
                        dim=dim,
                        num_heads=num_heads).to(device)
    model_lightning = TransformerLightning(model=model,
                                           lr=lr,
                                           dec_vocab_size=vocab_size,
                                           mask_size=vocab_size + 1)

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

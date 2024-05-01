from functools import partial
import click

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from src.model import Transformer
from src.model_lightning import TransformerLightning
from src.dataset import SNDataset


@click.command()
@click.option("--accelerator", default="gpu", help="accelerator for training [gpu|cpu|tpu|ipu|None] (default: gpu)")
@click.option("--devices", default="1", help="number of devices (default: 1)")
@click.option("--lr", default=0.01, help="learning rate")
@click.option("--max_epochs", default=100, help="epoch")
@click.option("--batch_size", default=128, help="batch size")
@click.option("--num_heads", default=1, help="Headの数")
@click.option("--dim", default=32, help="embedding dimension")
@click.option("--num_categories", default=12, help="vocab (今回は0 ~ 9 + 開始/終了タグ なので12個)")
@click.option("--seq_len", default=16, help="系列長")
@click.option("--debug", is_flag=True, help="デバックモードで実行")
def main(accelerator, devices, lr, max_epochs, num_heads, dim, batch_size, num_categories, seq_len, debug):

    # setting
    device = "cuda" if devices is not None else "cpu"

    # dataloaderを作成
    dataset = partial(SNDataset, num_categories, seq_len)
    test_loader = DataLoader(dataset(1), batch_size=1, shuffle=True, drop_last=True, pin_memory=True)

    # modelを作成
    model = Transformer(device=device,
                        enc_vocab_size=num_categories,
                        dec_vocab_size=num_categories,
                        dim=dim,
                        num_heads=num_heads).to(device)

    model_lightning = TransformerLightning.load_from_checkpoint("ckpts/head1-dim32-lr0.0001/epoch=29.ckpt",
                                                                model=model,
                                                                lr=lr,
                                                                dec_vocab_size=num_categories,
                                                                mask_size=seq_len + 1)

    with torch.no_grad():
        x, dec_input, target = next(iter(test_loader))
        x, dec_input, target = x.to(device), dec_input.to(device), target.to(device)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len + 1).to(device)  # マスクの作成

        dec_output = model_lightning.model(x, dec_input, mask)
        dec_output = F.softmax(dec_output, dim=-1)

        print(f"x           : {x.tolist()}")
        print(f"dec_input   : {dec_input.tolist()}")
        print(f"target      : {target.tolist()}")
        print(f"prediction  : {dec_output.argmax(dim=-1).tolist()}")
        print(f"accuracy    : {torch.sum(dec_output.argmax(dim=-1) == target) / num_categories:.4}")
        print(f"chance rate : {1 / num_categories:.4}")


if __name__ == "__main__":
    main()

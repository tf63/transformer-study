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
@click.option("--lr", default=0.0001, help="learning rate")
@click.option("--max_epochs", default=100, help="epoch")
@click.option("--num_datas", default=50000, help="data数")
@click.option("--batch_size", default=128, help="batch size")
@click.option("--num_heads", default=8, help="Headの数")
@click.option("--dim", default=512, help="embedding dimension")
@click.option("--debug", is_flag=True, help="デバックモードで実行")
def main(accelerator, devices, lr, max_epochs, num_datas, num_heads, dim, batch_size, debug):

    # setting
    device = "cuda" if devices is not None else "cpu"

    num_categories = 10  # vocab (今回は0 ~ 9)
    seq_len = 16  # 系列長
    vocab_size = num_categories + 4  # 0 ~ 9 + 開始/終了/余白タグ と 偶数にするために+1
    assert seq_len > vocab_size, "今回はseq_lenがvocab_sizeより大きいことを想定"

    # dataloaderを作成
    dataset = partial(SNDataset, num_categories, seq_len)
    test_loader = DataLoader(dataset(1), batch_size=1, shuffle=True, drop_last=True, pin_memory=True)

    # modelを作成
    model = Transformer(device=device,
                        enc_vocab_size=vocab_size,
                        dec_vocab_size=vocab_size,
                        dim=dim,
                        num_heads=num_heads).to(device)
    model_lightning = TransformerLightning.load_from_checkpoint("ckpts/sn-data-50000-head8-dim512-lr0.0001/epoch=9.ckpt",
                                                                model=model,
                                                                lr=lr,
                                                                dec_vocab_size=vocab_size,
                                                                mask_size=seq_len)

    # サンプリング
    with torch.no_grad():
        x, _, target = next(iter(test_loader))

        # decoderへの入力を作成
        dec_input = torch.full_like(target, num_categories + 3)  # 余白タグで埋める
        dec_input[:, 0] = num_categories + 1  # 先頭を開始タグにする
        
        # マスクを作成
        mask = nn.Transformer.generate_square_subsequent_mask(vocab_size + 1).to(device) 

        # 貪欲法でサンプリング
        x, dec_input, target = x.to(device), dec_input.to(device), target.to(device)
        for i in range(seq_len - 2):
            dec_output = model_lightning.model(x, dec_input, mask) # 推論
            dec_output = F.softmax(dec_output, dim=-1) # i+1番目の数字を予測
            dec_input[:, i + 1] = dec_output.argmax(dim=-1)[:, i] # dec_inputのi+1番目を予測した数字で埋める

        # dec_inputにpaddingをつけたものがpredictionとなる
        # (targetのseq_lenが十分長いので末尾は必ずpaddingになる)
        prediction = torch.cat([dec_input[:, 1:], torch.tensor([[num_categories + 3]]).to(device)], dim=1)

        # 結果
        print(f"x           : {x.tolist()}")
        print(f"dec_input   : {dec_input.tolist()}")
        print(f"target      : {target.tolist()}")
        print(f"prediction  : {prediction.tolist()}")
        print(f"accuracy    : {torch.sum(prediction == target) / (seq_len - 1):.4}")
        print(f"chance rate : {1 / seq_len:.4}")


if __name__ == "__main__":
    main()

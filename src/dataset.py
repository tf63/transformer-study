# code from https://github.com/i14kwmr/practice-transformer/blob/main/ReverseDataset.py
import torch
import torch.utils.data as data


# 入力: 大きさseq_lenの配列inp_data, 出力（ラベル）: 逆順のinp_data
# sizeはデータ数
class ReverseDataset(data.Dataset):
    def __init__(self, num_categories, seq_len, size):
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len
        self.size = size

        self.data = torch.randint(self.num_categories, size=(self.size, self.seq_len))  # num_categories未満の整数

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.data[idx]  # 入力データ
        y = torch.flip(x, dims=(0,))  # 正解データ

        prefix = torch.tensor([10])
        suffix = torch.tensor([11])

        x = torch.cat([prefix, x, suffix], dim=0)
        y = torch.cat([prefix, y, suffix], dim=0)

        dec_input = y[:-1]  # decoderへの入力 (1つシフトする)
        target = y[1:]  # 正解ラベル

        return x, dec_input, target


if __name__ == "__main__":
    from functools import partial
    from torch.utils.data import DataLoader

    # dataloaderを作成
    num_categories = 12
    seq_len = 14
    dataset = partial(ReverseDataset, num_categories, seq_len)
    train_loader = DataLoader(dataset(1000), batch_size=2, shuffle=True, drop_last=True, pin_memory=True)

    for batch in train_loader:
        x, dec_input, target = batch
        print(f"shape: x {x.shape}, dec_input: {dec_input.shape}, target: {target.shape}")
        print(f"x[0]: {x[0]}")
        print(f"dec_input[0]: {dec_input[0]}")
        print(f"target[0]: {target[0]}")
        break

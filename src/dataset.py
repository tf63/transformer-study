import torch
import torch.utils.data as data
import torch.nn.functional as F


class SNDataset(data.Dataset):
    def __init__(self, num_categories, seq_len, size):
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len
        self.size = size

        self.data = torch.randint(self.num_categories, size=(self.size, 2))  # num_categories未満の整数

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.data[idx]  # 入力データ

        # x[0] から x[1] までの連続した整数を生成
        if x[0] < x[1]:
            y = torch.arange(x[0].item(), x[1].item() + 1)
        elif x[0] == x[1]:
            y = torch.tensor([x[0].item()])
        else:
            y = torch.flip(torch.arange(x[1].item(), x[0].item() + 1), dims=(0,))

        prefix = torch.tensor([self.num_categories + 1])  # 開始タグ
        suffix = torch.tensor([self.num_categories + 2])  # 終了タグ

        x = torch.cat([prefix, x, suffix], dim=0)
        y = torch.cat([prefix, y, suffix], dim=0)

        # padding
        x = F.pad(x, (0, self.seq_len - x.size(0)), value=self.num_categories + 3)
        y = F.pad(y, (0, self.seq_len - y.size(0)), value=self.num_categories + 3)

        dec_input = y[:-1]  # decoderへの入力 (1つシフトする)
        target = y[1:]  # 正解ラベル

        return x, dec_input, target


if __name__ == "__main__":
    from functools import partial
    from torch.utils.data import DataLoader

    # dataloaderを作成
    num_categories = 10
    seq_len = 14
    dataset = partial(SNDataset, num_categories, seq_len)
    train_loader = DataLoader(dataset(1000), batch_size=2, shuffle=True, drop_last=True, pin_memory=True)

    for batch in train_loader:
        x, dec_input, target = batch
        print(f"shape: x {x.shape}, dec_input: {dec_input.shape}, target: {target.shape}")
        print(f"x[0]: {x[0]}")
        print(f"dec_input[0]: {dec_input[0]}")
        print(f"target[0]: {target[0]}")
        break

import torch
import torch.utils.data as data
import torch.nn.functional as F


class SNDataset(data.Dataset):
    """
    Serial Number Dataset
    0からnum_categories-1までの整数のペアと、その間の連番を生成するデータセット
    初期化
        num_categories (int): 0からnum_categories-1までの整数が生成される
        seq_len (int): シーケンス長
        size (int): データセットのデータ数
    batchはx, dec_input, targetで構成される
        x: 0からnum_categories-1までのランダムな整数のペア
        dec_input: xから作成した連番 (1つシフト)
        target: xから作成した連番
    前処理としてprefix, suffix, paddingを追加している
        prefix: number_categories + 1を割り当て
        suffix: number_categories + 2を割り当て
        padding: number_categories + 3を割り当て
    例: 
        num_categories=10, seq_len=6
            x: [11, 3, 6, 12, 13, 13]
            dec_input: [11, 3, 4, 5, 6]
            target: [3, 4, 5, 6, 12]
    """

    def __init__(self, num_categories, seq_len, size):
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len
        self.size = size

        self.prefix = num_categories + 1
        self.suffix = num_categories + 2
        self.padding = num_categories + 3

        # ランダムな整数のペアを作成
        self.data = torch.randint(self.num_categories, size=(self.size, 2)) 

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.data[idx]

        # x[0] から x[1] までの連続した整数を生成
        if x[0] < x[1]:
            y = torch.arange(x[0].item(), x[1].item() + 1)
        elif x[0] == x[1]:
            y = torch.tensor([x[0].item()])
        else:
            y = torch.flip(torch.arange(x[1].item(), x[0].item() + 1), dims=(0,))

        # suffixとprefixを追加
        prefix = torch.tensor([self.prefix])
        suffix = torch.tensor([self.suffix])

        x = torch.cat([prefix, x, suffix], dim=0)
        y = torch.cat([prefix, y, suffix], dim=0)

        # padding
        x = F.pad(x, (0, self.seq_len - x.size(0)), value=self.padding)
        y = F.pad(y, (0, self.seq_len - y.size(0)), value=self.padding)

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

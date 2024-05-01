from functools import partial

from torch.utils.data import DataLoader

from src.dataset import ReverseDataset

# dataloaderを作成
num_categories = 10
seq_len = 16
dataset = partial(ReverseDataset, num_categories, seq_len)
train_loader = DataLoader(dataset(1000), batch_size=2, shuffle=True, drop_last=True, pin_memory=True)

for batch in train_loader:
    x, dec_input, target = batch
    print(x.shape, dec_input.shape, target.shape)
    print(x, dec_input, target)
    # prefix = torch.tensor([[-1]]).expand(x.shape[0], 1)
    # suffix = torch.tensor([[-2]]).expand(x.shape[0], 1)
    # x = torch.cat([prefix, x, suffix], dim=1)
    # print(x)
    break

import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from helper.numpy_helpers import load_np


# Need to override __init__, __len__, __getitem__
# as per datasets requirement
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.items = os.listdir(data_dir)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        x = load_np(os.path.join(self.data_dir, f"x{index}"))
        y = load_np(os.path.join(self.data_dir, f"y{index}"))
        return x, y


train_ds = CustomDataset(data_dir='../generated_data/fractals')
train_loader = DataLoader(
    train_ds,
    batch_size=1,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
)
print(train_loader)
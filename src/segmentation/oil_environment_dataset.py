from torch.utils.data import Dataset
import os
from helper.numpy_helpers import load_np
import torch
from torch.utils.data import DataLoader


# Need to override __init__, __len__, __getitem__
# as per datasets requirement
class OilEnvironmentDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.items = os.listdir(data_dir)

    def __len__(self):
        return len([item for item in self.items if 'x' in item])

    def __getitem__(self, index):
        x = torch.from_numpy(load_np(os.path.join(self.data_dir, f"x{index}"))).float()
        y = torch.from_numpy(load_np(os.path.join(self.data_dir, f"y{index}"))).float()
        x = torch.moveaxis(x, -1, 0)
        return x, y


def get_loaders(
        train_dir,
        val_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    train_ds = OilEnvironmentDataset(
        data_dir=train_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = OilEnvironmentDataset(
        data_dir=val_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

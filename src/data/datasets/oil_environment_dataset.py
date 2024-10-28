from torch.utils.data import Dataset
import os
from helper.numpy_helpers import load_np
import torch
from torch.utils.data import DataLoader


class OilEnvironmentDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.items = os.listdir(data_dir)

    def __len__(self):
        return len([item for item in self.items if 'x' in item])

    def __getitem__(self, index):
        x_np = load_np(os.path.join(self.data_dir, f"x{index}"))
        y_np = load_np(os.path.join(self.data_dir, f"ye{index}"))

        if self.transform:
            x_np = self.transform(image=x_np)['image']

        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).float()
        x = torch.moveaxis(x, -1, 0)
        return x, y


class OilEnvironmentDatasetClassificationAndEstimation(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.items = os.listdir(data_dir)

    def __len__(self):
        return len([item for item in self.items if 'x' in item])

    def __getitem__(self, index):
        x_np = load_np(os.path.join(self.data_dir, f"x{index}"))
        yc_np = load_np(os.path.join(self.data_dir, f"yc{index}"))
        ye_np = load_np(os.path.join(self.data_dir, f"ye{index}"))
        if self.transform:
            transform = self.transform(image=x_np, mask1=yc_np, mask2=ye_np)
            x_np = transform["image"]
            yc_np = transform["mask1"]
            ye_np = transform["mask2"]
        x = torch.from_numpy(x_np).float()
        yc = torch.from_numpy(yc_np).float()
        ye = torch.from_numpy(ye_np).long()
        x = torch.moveaxis(x, -1, 0)

        return x, (yc, ye)


def get_loaders(
        train_dir,
        val_dir,
        batch_size,
        train_transform,
        val_transform,
        dataset_class,
        num_workers=4,
        pin_memory=True,
):
    train_ds = dataset_class(
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

    val_ds = dataset_class(
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

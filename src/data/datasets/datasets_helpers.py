import torch
import albumentations as A


def get_mean_and_std(data_loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in data_loader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def get_normalize_transform(data_loader):
    mean, std = get_mean_and_std(data_loader)
    return A.Normalize(
        mean=mean,
        std=std,
        max_pixel_value=1,
    )


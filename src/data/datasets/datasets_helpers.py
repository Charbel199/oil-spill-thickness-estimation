import torch
import albumentations as A


def get_mean_and_std(data_loader):
    """
    Calculate the per-channel mean and standard deviation of a dataset.

    Parameters:
    data_loader (DataLoader): A PyTorch DataLoader providing batches of images in the format (batch, channels, height, width).

    Returns:
    tuple: (mean, std), where mean and std are tensors containing the per-channel mean and standard deviation.
    """
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


def get_max(data_loader):
    """
    Calculate the maximum value across all batches in a data loader.

    Parameters:
    data_loader (DataLoader): A PyTorch DataLoader providing batches of data.

    Returns:
    float: The maximum value across all data in the data loader.
    """
    max = float('-inf')
    for data, _ in data_loader:
        temp_max = torch.max(data)
        if temp_max > max:
            max = temp_max
    return max



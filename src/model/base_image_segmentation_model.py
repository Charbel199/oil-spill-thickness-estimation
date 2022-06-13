import torchvision.transforms.functional as TF
from tqdm import tqdm
import torch.nn as nn
from visualization.environment_oil_thickness_distribution import visualize_environment
import torch
from abc import abstractmethod

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ImageSegmentationModel(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], normalize_output=False
    ):
        super(ImageSegmentationModel, self).__init__()
        self.normalize_output = normalize_output
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        return torch.sigmoid(self.final_conv(x)) if self.normalize_output else self.final_conv(x)

    def train_fn(self, loader, optimizer, loss_fn, device='cpu'):
        loop = tqdm(loader)

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=device)
            targets = targets.float().to(device=device).unsqueeze(dim=1)
            # forward

            predictions = self(data)
            loss = loss_fn(predictions, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())

    @abstractmethod
    def process_prediction(self, predictions):
        pass

    def check_accuracy(self, loader, device="cuda"):
        num_correct = 0
        num_pixels = 0
        dice_score = 0
        self.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.float().to(device)
                predictions = self(x)
                predictions = self.process_prediction(predictions)

                num_correct += (predictions == y).sum()
                num_pixels += torch.numel(predictions)
                dice_score += (2 * (predictions * y).sum()) / (
                        (predictions + y).sum() + 1e-8
                )

        print(
            f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
        )
        print(f"Dice score: {dice_score / len(loader)}")
        self.train()

    def save_predictions_as_images(
            self, loader, folder="saved_images", device="cuda"
    ):
        self.eval()
        for idx, (x, y) in enumerate(loader):
            x = x.to(device=device)
            with torch.no_grad():
                predictions = self(x)
                predictions = self.process_prediction(predictions)

                index = idx * predictions.shape[0]
                for pred in predictions:
                    visualize_environment(environment=pred, save_fig=True, show_fig=False,
                                          output_file_name=f"{folder}/pred_{index}", file_type='jpeg')
                    index += 1

        self.train()

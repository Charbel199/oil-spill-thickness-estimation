import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
import torch
from data.datasets.oil_environment_dataset import get_loaders, OilEnvironmentDatasetClassificationAndEstimation


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.conv(x)


class ClassificationSegmentationModel(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[16, 32, 64], normalize_output=True
    ):
        super(ClassificationSegmentationModel, self).__init__()
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
        final = self.final_conv(x)
        return torch.sigmoid(final) if self.normalize_output else final


class EstimationSegmentationModel(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], normalize_output=False
    ):
        super(EstimationSegmentationModel, self).__init__()
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
        final = self.final_conv(x)
        return torch.sigmoid(final) if self.normalize_output else final


# Hyperparameters.
LEARNING_RATE = 1e-4
DEVICE = "cpu"
BATCH_SIZE = 10
NUM_EPOCHS = 5
NUM_WORKERS = 0
IMAGE_HEIGHT = 80  # 1280 originally
IMAGE_WIDTH = 80  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL_FROM_CHECKPOINT = False
SAVE = False
LOAD = True
MODEL_CHECKPOINT = "my_checkpoint2.pth.tar"
TRAIN_IMG_DIR = "assets/generated_data/variance_0.02/fractals_with_0_cascaded/training"
VAL_IMG_DIR = "assets/generated_data/variance_0.02/fractals_with_0_cascaded/validation"
PRED_IMG_DIR = "assets/generated_data/variance_0.02/fractals_with_0_cascaded/pred"
CLASSIFIER_MODEL_PATH = 'assets/generated_models/unet_highvariance_with_0_cascaded_classifier_unified_loss.pkl'
ESTIMATOR_MODEL_PATH = 'assets/generated_models/unet_highvariance_with_0_cascaded_estimator_unified_loss.pkl'
train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

val_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

train_loader, val_loader = get_loaders(
    TRAIN_IMG_DIR,
    VAL_IMG_DIR,
    BATCH_SIZE,
    train_transform,
    val_transforms,
    OilEnvironmentDatasetClassificationAndEstimation,
    NUM_WORKERS,
    PIN_MEMORY,
)


learning_rate = 1e-4

# Initialize models
classifier = ClassificationSegmentationModel(in_channels=4, out_channels=1, normalize_output=True).to(DEVICE)
estimator = EstimationSegmentationModel(in_channels=5, out_channels=11, normalize_output=False).to(DEVICE)
# Set optimizers
opt_classifier = optim.Adam(classifier.parameters(), lr=learning_rate)
opt_estimator = optim.Adam(estimator.parameters(), lr=learning_rate)
opt_all = optim.Adam([
    {'params': classifier.parameters()},
    {'params': estimator.parameters()}
], lr=learning_rate)


criterion_classifier = nn.BCELoss()
criterion_estimator = nn.CrossEntropyLoss()

if not LOAD:
    classifier.train()
    estimator.train()
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(NUM_EPOCHS):
            for batch_idx, (data, classification_targets, estimator_targets) in enumerate(train_loader):
                BATCH_SIZE = data.shape[0]


                classified_output = classifier(data)
                loss_classifier = criterion_classifier(classified_output.squeeze(), classification_targets)
                # opt_classifier.zero_grad()
                # loss_classifier.backward(retain_graph=True)
                # opt_classifier.step()

                estimator_input = torch.cat((data, classified_output.detach()), dim=1)
                estimated_output = estimator(estimator_input)
                loss_estimator = criterion_estimator(estimated_output, estimator_targets)



                # opt_estimator.zero_grad()
                # loss_estimator.backward(retain_graph=True)
                # opt_estimator.step()

                total_loss = 0.8*loss_classifier + loss_estimator
                opt_all.zero_grad()
                total_loss.backward(retain_graph=True)
                opt_all.step()

                print(f"Done batch {batch_idx}")
else:
    classifier = torch.load(CLASSIFIER_MODEL_PATH)
    estimator = torch.load(ESTIMATOR_MODEL_PATH)

if SAVE:
    torch.save(classifier, CLASSIFIER_MODEL_PATH)
    torch.save(estimator, ESTIMATOR_MODEL_PATH)

from visualization.environment_oil_thickness_distribution import visualize_environment
def check_accuracy(classifier,estimator, loader,folder, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    classifier.eval()
    estimator.eval()

    with torch.no_grad():
        for idx, (x, yc, ye) in enumerate(loader):
            classification = classifier(x)
            estimator_input = torch.cat((x, classification), dim=1)
            estimation = estimator(estimator_input)
            index = idx * estimation.shape[0]
            for pred in estimation:
                pred = torch.nn.functional.softmax(pred, dim=0)
                pred = torch.argmax(pred, dim=0)
                visualize_environment(environment=pred, save_fig=True, show_fig=False,
                                      output_file_name=f"{folder}/pred_estimation_{index}", file_type='jpeg')
                index += 1
            index = idx * classification.shape[0]
            for pred in classification:
                pred = pred.squeeze()
                visualize_environment(environment=pred, save_fig=True, show_fig=False,
                                      output_file_name=f"{folder}/pred_classification_{index}", file_type='jpeg')
                index += 1

    classifier.train()
    estimator.train()

check_accuracy(classifier=classifier,
               estimator=estimator,
               loader=val_loader,
               folder=PRED_IMG_DIR)
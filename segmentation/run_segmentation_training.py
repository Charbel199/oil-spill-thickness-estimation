import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
import torch
from segmentation.oil_environment_dataset import get_loaders
from helper.torch_helpers import load_checkpoint, save_checkpoint
from model.unet_model import UNET

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 3
NUM_WORKERS = 0
IMAGE_HEIGHT = 80  # 1280 originally
IMAGE_WIDTH = 80  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "../generated_data/fractals/"
VAL_IMG_DIR = "../generated_data/fractals/validation"
NUM_OF_CLASSES = 11

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

model = UNET(in_channels=4, out_channels=NUM_OF_CLASSES).to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loader, val_loader = get_loaders(
    TRAIN_IMG_DIR,
    VAL_IMG_DIR,
    BATCH_SIZE,
    train_transform,
    val_transforms,
    NUM_WORKERS,
    PIN_MEMORY,
)

if LOAD_MODEL:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

model.check_accuracy(val_loader, device=DEVICE)

for epoch in range(NUM_EPOCHS):
    model.train_fn(train_loader, optimizer, loss_fn)

    # save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint)

    # check accuracy
    model.check_accuracy(val_loader, device=DEVICE)

    # print some examples to a folder
    model.save_predictions_as_images(
        val_loader, folder="./generated_data", device=DEVICE
    )

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
import torch
from data.datasets.oil_environment_dataset import get_loaders, OilEnvironmentDataset
from helper.torch_helpers import load_checkpoint, save_checkpoint
from model.unet_model import UNET

# from model.unet_model_classification import UNETClassifier as UNET

# Parameters
# ==================================================================================================================
LEARNING_RATE = 1e-4
DEVICE = "cpu"
BATCH_SIZE = 10
NUM_EPOCHS = 5
NUM_WORKERS = 0
IMAGE_HEIGHT = 80  # 1280 originally
IMAGE_WIDTH = 80  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL_FROM_CHECKPOINT = False
SAVE_PREDICTION_IMAGES = False
EVALUATE_METRICS = True
MODEL_CHECKPOINT = "my_checkpoint2.pth.tar"
TRAIN_IMG_DIR = "assets/generated_data/variance_0.02/fractals_with_0/training"
VAL_IMG_DIR = "assets/generated_data/variance_0.02/fractals_with_0/validation"
PRED_IMG_DIR = "assets/generated_data/variance_0.02/fractals_with_0/pred"
NUM_OF_CLASSES = 11
SAVE = False
LOAD = True
MODEL_PATH = 'assets/generated_models/unet_highvariance_with_0.pkl'
# ==================================================================================================================

# Normalize if classification
model = UNET(in_channels=4, out_channels=NUM_OF_CLASSES, normalize_output=False).to(DEVICE)
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

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loader, val_loader = get_loaders(
    TRAIN_IMG_DIR,
    VAL_IMG_DIR,
    BATCH_SIZE,
    train_transform,
    val_transforms,
    OilEnvironmentDataset,
    NUM_WORKERS,
    PIN_MEMORY,
)


def _evaluate_model():
    # check accuracy
    if EVALUATE_METRICS:
        model.check_accuracy(val_loader, device=DEVICE)
        model.evaluate_metrics(val_loader)
    # print some examples to a folder
    if SAVE_PREDICTION_IMAGES:
        model.save_predictions_as_images(
            val_loader, folder=PRED_IMG_DIR, device=DEVICE)


if not LOAD:

    if LOAD_MODEL_FROM_CHECKPOINT:
        load_checkpoint(torch.load(MODEL_CHECKPOINT), model)

    for epoch in range(NUM_EPOCHS):
        print(f"Starting epoch {epoch}")
        model.train_fn(train_loader, optimizer, loss_fn)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        _evaluate_model()


else:
    model = torch.load(MODEL_PATH)
    model.eval()

    _evaluate_model()

if SAVE:
    torch.save(model, MODEL_PATH)

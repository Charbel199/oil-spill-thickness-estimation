import albumentations as A
from data.datasets.datasets_helpers import get_mean_and_std
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
COMPUTE_MEAN_AND_STD = False
BATCH_SIZE = 10
NUM_EPOCHS = 20
NUM_WORKERS = 0
IMAGE_HEIGHT = 80  # 1280 originally
IMAGE_WIDTH = 80  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL_FROM_CHECKPOINT = False
SAVE_PREDICTION_IMAGES = True
EVALUATE_METRICS = True
MODEL_CHECKPOINT = "my_checkpoint2.pth.tar"
TRAIN_IMG_DIR = "assets/generated_data/variance_0.02_windspeed_4/fractals_with_0/training"
VAL_IMG_DIR = "assets/generated_data/variance_0.02_windspeed_4/fractals_with_0/validation"
PRED_IMG_DIR = "assets/generated_data/variance_0.02_windspeed_4/fractals_with_0/pred"
NUM_OF_CLASSES = 11
SAVE = True
LOAD = False
MODEL_PATH = 'assets/generated_models/unet_highvariance_windspeed4_with_0.pkl'
# ==================================================================================================================

# Normalize if classification
model = UNET(in_channels=4, out_channels=NUM_OF_CLASSES, normalize_output=False).to(DEVICE)

train_transform = []
val_transform = []
if not COMPUTE_MEAN_AND_STD:
    train_transform = A.Compose(
        [
            A.Normalize(
                mean=[0.4678, 0.4022, 0.4325, 0.4570],
                std=[0.1877, 0.1894, 0.1965, 0.1972],
                max_pixel_value=1,
            ),
        ]
    )
    val_transform = A.Compose(
        [
            A.Normalize(
                mean=[0.4678, 0.4022, 0.4325, 0.4570],
                std=[0.1877, 0.1894, 0.1965, 0.1972],
                max_pixel_value=1.2243,
            )
        ]
    )

# Set criterion
loss_fn = nn.CrossEntropyLoss()
# Set optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loader, val_loader = get_loaders(
    TRAIN_IMG_DIR,
    VAL_IMG_DIR,
    BATCH_SIZE,
    train_transform,
    val_transform,
    OilEnvironmentDataset,
    NUM_WORKERS,
    PIN_MEMORY,
)

if COMPUTE_MEAN_AND_STD:
    print(f"Training dataset mean and std {get_mean_and_std(train_loader)}")
    print(f"Validation dataset mean and std {get_mean_and_std(val_loader)}")
    exit()


def _evaluate_model(save_images=False):
    # print some examples to a folder
    if SAVE_PREDICTION_IMAGES and save_images:
        model.save_predictions_as_images(
            val_loader, folder=PRED_IMG_DIR, device=DEVICE)
    # check accuracy
    if EVALUATE_METRICS:
        model.check_accuracy(val_loader, device=DEVICE)
        model.evaluate_metrics(val_loader)


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

        _evaluate_model(save_images=False)

    # Final evaluation
    _evaluate_model()


else:
    model = torch.load(MODEL_PATH)
    model.eval()

    _evaluate_model()

if SAVE:
    torch.save(model, MODEL_PATH)

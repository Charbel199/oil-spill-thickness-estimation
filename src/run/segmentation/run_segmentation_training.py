import albumentations as A
from data.datasets.datasets_helpers import get_mean_and_std
import torch.nn as nn
import torch.optim as optim
import torch
from data.datasets.oil_environment_dataset import get_loaders, OilEnvironmentDataset
from helper.torch_helpers import load_checkpoint, save_checkpoint
from model.unet_model import UNET
import os

# Parameters
# ==================================================================================================================
LEARNING_RATE = 8e-4
DEVICE = "cuda"
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_WORKERS = 0
PIN_MEMORY = True
LOAD_MODEL_FROM_CHECKPOINT = False
SAVE_PREDICTION_IMAGES = True
EVALUATE_METRICS = True
MODEL_CHECKPOINT = "my_checkpoint.pth.tar"
TRAIN_IMG_DIR = "assets/train"
VAL_IMG_DIR = "assets/val"
PRED_IMG_OUTPUT_DIR = "assets/pred"
NUM_OF_CLASSES = 11
SAVE = True
LOAD = False
MODEL_PATH = 'assets/unet_highvariance_all_windspeeds_normalized_10epochs_9freq.pkl'
# ==================================================================================================================

# Normalize if classification
model = UNET(in_channels=9, out_channels=NUM_OF_CLASSES, normalize_output=False, device=DEVICE).to(DEVICE)

train_transform = []
val_transform = []
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




def _evaluate_model(save_images=True):
    # print some examples to a folder
    if SAVE_PREDICTION_IMAGES and save_images:
        os.makedirs(PRED_IMG_OUTPUT_DIR, exist_ok=True)
        model.save_predictions_as_images(
            val_loader, folder=PRED_IMG_OUTPUT_DIR, device=DEVICE)
    # check accuracy
    if EVALUATE_METRICS:
        model.check_accuracy(val_loader, device=DEVICE)
        model.evaluate_metrics(val_loader, num_classes=NUM_OF_CLASSES)


if not LOAD:

    if LOAD_MODEL_FROM_CHECKPOINT:
        load_checkpoint(torch.load(MODEL_CHECKPOINT), model)

    for epoch in range(NUM_EPOCHS):
        print(f"Starting epoch {epoch}")
        model.train_fn(train_loader, optimizer, loss_fn, device=DEVICE)

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

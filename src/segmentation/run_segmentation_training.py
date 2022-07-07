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
NORMALIZE = True
BATCH_SIZE = 10
NUM_EPOCHS = 10
NUM_WORKERS = 0
IMAGE_HEIGHT = 80  # 1280 originally
IMAGE_WIDTH = 80  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL_FROM_CHECKPOINT = False
SAVE_PREDICTION_IMAGES = True
EVALUATE_METRICS = True
MODEL_CHECKPOINT = "my_checkpoint2.pth.tar"
TRAIN_IMG_DIR = "assets/generated_data/variance_0.02_windspeed_8/fluids_cascaded_9freq/training"
VAL_IMG_DIR = "assets/generated_data/variance_0.02_windspeed_8/fluids_cascaded_9freq/validation"
PRED_IMG_DIR = "assets/generated_data/variance_0.02_windspeed_8/fluids_cascaded_9freq/pred_non_cascaded"
NUM_OF_CLASSES = 11
SAVE = False
LOAD = True
MODEL_PATH = 'assets/generated_models/unet_highvariance_windspeed_8_cascaded_normalized_10epochs_9freq.pkl'
# ==================================================================================================================

# Normalize if classification
model = UNET(in_channels=9, out_channels=NUM_OF_CLASSES, normalize_output=False).to(DEVICE)

train_transform = []
val_transform = []
if not COMPUTE_MEAN_AND_STD and NORMALIZE:
    # Variance 0.02
    # mean = [0.4678, 0.4022, 0.4325, 0.4570]
    # std = [0.1877, 0.1894, 0.1965, 0.1972]


    # Variance 0.02 Windspeed 8 - 17 freq
    # mean = [0.5395, 0.5235, 0.5126, 0.5078, 0.5045, 0.5041, 0.5041, 0.5045, 0.5033,
    #     0.5033, 0.5006, 0.4939, 0.4886, 0.4812, 0.4713, 0.4614, 0.4491]
    # std = [0.1906, 0.1972, 0.1983, 0.1949, 0.1925, 0.1906, 0.1885, 0.1875, 0.1870,
    #     0.1861, 0.1854, 0.1835, 0.1826, 0.1805, 0.1796, 0.1830, 0.1854]

    # Variance 0.02 Windspeed 8 - 9 freq
    mean = [0.5410, 0.5144, 0.5050, 0.5052, 0.5038, 0.4999, 0.4904, 0.4743, 0.4515]
    std = [0.1906, 0.1978, 0.1927, 0.1887, 0.1871, 0.1842, 0.1811, 0.1793, 0.1853]

    train_transform = A.Compose(
        [
            A.Normalize(
                mean=mean,
                std=std,
                max_pixel_value=1,
            )
        ],
        additional_targets={'mask1': 'mask',
                            'mask2': 'mask', }
    )
    val_transform = A.Compose(
        [
            A.Normalize(
                mean=mean,
                std=std,
                max_pixel_value=1,
            )
        ],
        additional_targets={'mask1': 'mask',
                            'mask2': 'mask', }
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


def _evaluate_model(save_images=True):
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

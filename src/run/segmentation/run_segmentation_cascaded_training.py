import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
import torch
from data.datasets.oil_environment_dataset import get_loaders, OilEnvironmentDatasetClassificationAndEstimation
from model.unet_model_cascaded import SemanticSegmentationCascadedModel, DiceLoss
from model.unet_model import UNET
from model.unet_model_classification import UNETClassifier
from data.datasets.datasets_helpers import get_mean_and_std, get_max
from torch.utils.tensorboard import SummaryWriter


def _evaluate_model(save_images=True):
    if EVALUATE_METRICS:
        cascaded_model.evaluate_metrics(val_loader, device=DEVICE)
    if SAVE_PREDICTION_IMAGES and save_images:
        cascaded_model.save_predictions_as_images(
            val_loader, folder=PRED_IMG_DIR, device=DEVICE
        )


# Parameters
# ==================================================================================================================
LEARNING_RATE_COMBINED = 8e-4
LEARNING_RATE_ESTIMATOR = 1e-3
LEARNING_RATE_CLASSIFIER = 2e-4
DEVICE = "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 10
COMBINED_LOSS = True
COMPUTE_MEAN_AND_STD = False
# Variance 0.02 REAL DATA  NO VIBRATIONS - 9 freq
mean = [0.5852, 0.3543, 0.1974, 0.5254, 0.3596, 0.3773, 0.7751, 0.3998, 0.8537]
std = [0.1174, 0.1459, 0.0919, 0.1373, 0.0659, 0.1293, 0.4533, 0.2151, 0.4849]
NORMALIZE = True
NUM_WORKERS = 0
PIN_MEMORY = False
LOAD_MODEL_FROM_CHECKPOINT = False
SAVE = True
LOAD = False
EVALUATE_METRICS = True
NUMBER_OF_FEATURES = 9  # Ex: Number of input frequencies
MODEL_CHECKPOINT = "my_checkpoint3.pth.tar"

TRAIN_IMG_DIR = "assets/training"
VAL_IMG_DIR = "assets/validation"
PRED_IMG_DIR = "assets/pred_cascaded"


SAVE_PREDICTION_IMAGES = True
CLASSIFIER_MODEL_PATH = f'assets/unet_highvariance_all_windspeeds_cascaded_normalized_classifier_unified_loss_10epochs_{NUMBER_OF_FEATURES}freq_lr8e-4.pkl'
ESTIMATOR_MODEL_PATH = f'assets/unet_highvariance_all_windspeeds_cascaded_normalized_estimator_unified_loss_10epochs_{NUMBER_OF_FEATURES}freq_lr8e-4.pkl'
print(f"Working on {NUMBER_OF_FEATURES} features")

# Inputs: Parameters - TRAIN and Validation directories (Output from run_segmentation_dataset_generation.py) + Model paths
# Trains the cascaded model based on the parameters
# Output: Print the evaluation metrics + Save prediction images if specified
# ==================================================================================================================


writer = SummaryWriter(
    "assets/logs/unet_highvariance_all_windspeeds_cascaded_normalized_estimator_unified_loss_10epochs_9freq_lr8e-4")

classifier = UNETClassifier(in_channels=NUMBER_OF_FEATURES, out_channels=1, normalize_output=True,
                            features=[64, 128, 256, 512]).to(
    DEVICE)
estimator = UNET(in_channels=NUMBER_OF_FEATURES + 1, out_channels=11, normalize_output=False,
                 features=[32, 64, 128, 256, 512]).to(
    DEVICE)
cascaded_model = SemanticSegmentationCascadedModel(classifier=classifier,
                                                   estimator=estimator)
train_transform = []
val_transform = []

if not COMPUTE_MEAN_AND_STD and NORMALIZE:
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
criterion_classifier = DiceLoss()
criterion_estimator = nn.CrossEntropyLoss()
# Set optimizers
opt_classifier = optim.Adam(classifier.parameters(), lr=LEARNING_RATE_CLASSIFIER)
opt_estimator = optim.Adam(estimator.parameters(), lr=LEARNING_RATE_ESTIMATOR)
opt_all = optim.Adam([
    {'params': classifier.parameters()},
    {'params': estimator.parameters()}
], lr=LEARNING_RATE_COMBINED)

train_loader, val_loader = get_loaders(
    TRAIN_IMG_DIR,
    VAL_IMG_DIR,
    BATCH_SIZE,
    train_transform,
    val_transform,
    OilEnvironmentDatasetClassificationAndEstimation,
    NUM_WORKERS,
    PIN_MEMORY,
)

if COMPUTE_MEAN_AND_STD:
    print(f"Training dataset mean and std {get_mean_and_std(train_loader)}")
    print(f"Validation dataset mean and std {get_mean_and_std(val_loader)}")
    print(f"Training dataset max {get_max(train_loader)}")
    print(f"Validation dataset max  {get_max(val_loader)}")

    exit()

if not LOAD:
    classifier.train()
    estimator.train()
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(NUM_EPOCHS):
            print(f"Starting epoch {epoch}")
            cascaded_model.train_fn(loader=train_loader,
                                    opt_classifier=opt_classifier,
                                    opt_estimator=opt_estimator,
                                    criterion_estimator=criterion_estimator,
                                    criterion_classifier=criterion_classifier,
                                    combined_loss=COMBINED_LOSS,
                                    summary_writer=writer,
                                    opt_all=opt_all,
                                    device=DEVICE)

            # # save model
            # checkpoint = {
            #     "state_dict": model.state_dict(),
            #     "optimizer": optimizer.state_dict(),
            # }
            # save_checkpoint(checkpoint)

            # _evaluate_model(save_images=False)

        writer.flush()
        # Final evaluation
        _evaluate_model()



else:
    classifier = torch.load(CLASSIFIER_MODEL_PATH)
    estimator = torch.load(ESTIMATOR_MODEL_PATH)
    cascaded_model = SemanticSegmentationCascadedModel(classifier=classifier,
                                                       estimator=estimator)
    _evaluate_model()

if SAVE:
    torch.save(cascaded_model.classifier, CLASSIFIER_MODEL_PATH)
    torch.save(cascaded_model.estimator, ESTIMATOR_MODEL_PATH)

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
import torch
from data.datasets.oil_environment_dataset import get_loaders, OilEnvironmentDatasetClassificationAndEstimation
from model.unet_model_cascaded import SemanticSegmentationCascadedModel, DiceLoss
from model.unet_model import UNET
from model.unet_model_classification import UNETClassifier

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
SAVE = True
LOAD = False
MODEL_CHECKPOINT = "my_checkpoint2.pth.tar"
TRAIN_IMG_DIR = "assets/generated_data/variance_0.02/fractals_with_0_cascaded/training"
VAL_IMG_DIR = "assets/generated_data/variance_0.02/fractals_with_0_cascaded/validation"
PRED_IMG_DIR = "assets/generated_data/variance_0.02/fractals_with_0_cascaded/pred"
CLASSIFIER_MODEL_PATH = 'assets/generated_models/unet_highvariance_with_0_cascaded_classifier_unified_loss.pkl'
ESTIMATOR_MODEL_PATH = 'assets/generated_models/unet_highvariance_with_0_cascaded_estimator_unified_loss.pkl'
# ==================================================================================================================

classifier = UNETClassifier(in_channels=4, out_channels=1, normalize_output=True).to(DEVICE)
estimator = UNET(in_channels=5, out_channels=11, normalize_output=False).to(DEVICE)
cascaded_model = SemanticSegmentationCascadedModel(classifier=classifier,
                                                   estimator=estimator)
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

# Set optimizers
opt_classifier = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
opt_estimator = optim.Adam(estimator.parameters(), lr=LEARNING_RATE)
opt_all = optim.Adam([
    {'params': classifier.parameters()},
    {'params': estimator.parameters()}
], lr=LEARNING_RATE)

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

criterion_classifier = DiceLoss()
criterion_estimator = nn.CrossEntropyLoss()

if not LOAD:
    classifier.train()
    estimator.train()
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(NUM_EPOCHS):
            cascaded_model.train_fn(loader=train_loader,
                                    opt_classifier=opt_classifier,
                                    opt_estimator=opt_estimator,
                                    criterion_estimator=criterion_estimator,
                                    criterion_classifier=criterion_classifier,
                                    opt_all=opt_all,
                                    device=DEVICE)

            # # save model
            # checkpoint = {
            #     "state_dict": model.state_dict(),
            #     "optimizer": optimizer.state_dict(),
            # }
            # save_checkpoint(checkpoint)

            # check accuracy
            # cascaded_model.check_accuracy(val_loader, device=DEVICE)

            # print some examples to a folder
            cascaded_model.save_predictions_as_images(
                val_loader, folder=PRED_IMG_DIR, device=DEVICE
            )



else:
    classifier = torch.load(CLASSIFIER_MODEL_PATH)
    estimator = torch.load(ESTIMATOR_MODEL_PATH)
    cascaded_model = SemanticSegmentationCascadedModel(classifier=classifier,
                                                       estimator=estimator)
    # check accuracy
    # cascaded_model.check_accuracy(val_loader, device=DEVICE)

    # print some examples to a folder
    cascaded_model.save_predictions_as_images(
        val_loader, folder=PRED_IMG_DIR, device=DEVICE
    )
if SAVE:
    torch.save(cascaded_model.classifier, CLASSIFIER_MODEL_PATH)
    torch.save(cascaded_model.estimator, ESTIMATOR_MODEL_PATH)

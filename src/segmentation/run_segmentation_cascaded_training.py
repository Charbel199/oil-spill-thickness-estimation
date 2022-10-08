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
COMPUTE_MEAN_AND_STD = False
NORMALIZE = True
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_WORKERS = 0
IMAGE_HEIGHT = 80  # 1280 originally
IMAGE_WIDTH = 80  # 1918 originally
PIN_MEMORY = False
LOAD_MODEL_FROM_CHECKPOINT = False
SAVE = False
LOAD = True
COMBINED_LOSS = True
SAVE_PREDICTION_IMAGES = True
EVALUATE_METRICS = True
NUMBER_OF_FEATURES = 9
MODEL_CHECKPOINT = "my_checkpoint2.pth.tar"
TRAIN_IMG_DIR = f"assets/generated_data/variance_0.02_all_windspeeds/all_frequencies/{NUMBER_OF_FEATURES}/train"
VAL_IMG_DIR = f"assets/generated_data/real_data_updated/val_with_vibrations"
PRED_IMG_DIR = f"assets/generated_data/real_data_updated/pred_with_vibrations"
CLASSIFIER_MODEL_PATH = f'assets/generated_models/all_frequencies/unet_highvariance_all_windspeeds_cascaded_normalized_classifier_unified_loss_10epochs_{NUMBER_OF_FEATURES}freq_lr8e-4.pkl'
ESTIMATOR_MODEL_PATH = f'assets/generated_models/all_frequencies/unet_highvariance_all_windspeeds_cascaded_normalized_estimator_unified_loss_10epochs_{NUMBER_OF_FEATURES}freq_lr8e-4.pkl'
print(f"Working on {NUMBER_OF_FEATURES} features")
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
    # Variance 0.02
    # mean = [0.4678, 0.4022, 0.4325, 0.4570]
    # std = [0.1877, 0.1894, 0.1965, 0.1972]

    # Variance 0.02 Windspeed 8 - 17 freq
    # mean = [0.5395, 0.5235, 0.5126, 0.5078, 0.5045, 0.5041, 0.5041, 0.5045, 0.5033,
    #     0.5033, 0.5006, 0.4939, 0.4886, 0.4812, 0.4713, 0.4614, 0.4491]
    # std = [0.1906, 0.1972, 0.1983, 0.1949, 0.1925, 0.1906, 0.1885, 0.1875, 0.1870,
    #     0.1861, 0.1854, 0.1835, 0.1826, 0.1805, 0.1796, 0.1830, 0.1854]

    # Variance 0.02 Windspeed 8 - 9 freq
    # mean = [0.5410, 0.5144, 0.5050, 0.5052, 0.5038, 0.4999, 0.4904, 0.4743, 0.4515]
    # std = [0.1906, 0.1978, 0.1927, 0.1887, 0.1871, 0.1842, 0.1811, 0.1793, 0.1853]

    # Variance 0.02 REAL DATA  VIBRATIONS - 9 freq
    # mean = [0.4899, 0.1192, 0.0845, 0.3451, 0.3563, 0.4547, 0.9817, 0.0947, 1.7195]
    # std = [0.0807, 0.0406, 0.0422, 0.1055, 0.1207, 0.1324, 0.4198, 0.0825, 0.7989]

    # Variance 0.02 REAL DATA  NO VIBRATIONS - 9 freq
    mean = [0.5852, 0.3543, 0.1974, 0.5254, 0.3596, 0.3773, 0.7751, 0.3998, 0.8537]
    std = [0.1174, 0.1459, 0.0919, 0.1373, 0.0659, 0.1293, 0.4533, 0.2151, 0.4849]

    # Variance 0.02 All windspeeds - 9 freq
    # mean = [0.5716, 0.5487, 0.5370, 0.5315, 0.5248, 0.5176, 0.5071, 0.4919, 0.4735]
    # std = [0.1772, 0.1867, 0.1848, 0.1821, 0.1792, 0.1769, 0.1742, 0.1729, 0.1764]
    # if NUMBER_OF_FEATURES == 2:
    #     mean = [0.5786, 0.5281]
    #     std = [0.1735, 0.1764]
    # if NUMBER_OF_FEATURES == 3:
    #     mean = [0.5784, 0.5277, 0.5067]
    #     std = [0.1735, 0.1765, 0.1714]
    # if NUMBER_OF_FEATURES == 4:
    #     mean = [0.5788, 0.5445, 0.5280, 0.5072]
    #     std = [0.1735, 0.1817, 0.1764, 0.1713]
    # if NUMBER_OF_FEATURES == 5:
    #     mean = [0.5789, 0.5449, 0.5285, 0.5075, 0.4709]
    #     std = [0.1731, 0.1813, 0.1759, 0.1713, 0.1728]
    # if NUMBER_OF_FEATURES == 6:
    #     mean = [0.5790, 0.5573, 0.5448, 0.5282, 0.5074, 0.4705]
    #     std = [0.1732, 0.1827, 0.1817, 0.1762, 0.1713, 0.1730]
    # if NUMBER_OF_FEATURES == 7:
    #     mean = [0.5790, 0.5577, 0.5449, 0.5356, 0.5283, 0.5074, 0.4707]
    #     std = [0.1730, 0.1827, 0.1815, 0.1789, 0.1762, 0.1714, 0.1728]
    # if NUMBER_OF_FEATURES == 8:
    #     mean = [0.5790, 0.5574, 0.5447, 0.5354, 0.5281, 0.5179, 0.5074, 0.4707]
    #     std = [0.1732, 0.1826, 0.1817, 0.1790, 0.1761, 0.1738, 0.1713, 0.1729]
    # if NUMBER_OF_FEATURES == 9:
    #     mean = [0.5781, 0.5562, 0.5436, 0.5346, 0.5275, 0.5174, 0.5068, 0.4903, 0.4700]
    #     std = [0.1739, 0.1834, 0.1821, 0.1794, 0.1764, 0.1740, 0.1717, 0.1699, 0.1732]
    # if NUMBER_OF_FEATURES == 10:
    #     mean = [0.6062, 0.5786, 0.5568, 0.5443, 0.5349, 0.5277, 0.5175, 0.5069, 0.4902,0.4703]
    #     std = [0.1552, 0.1736, 0.1830, 0.1817, 0.1791, 0.1764, 0.1739, 0.1715, 0.1699,0.1730]
    # if NUMBER_OF_FEATURES == 11:
    #     mean = [0.6062, 0.5782, 0.5566, 0.5438, 0.5346, 0.5276, 0.5174, 0.5068, 0.4905,0.4704, 0.4519]
    #     std = [0.1551, 0.1735, 0.1831, 0.1819, 0.1792, 0.1764, 0.1740, 0.1713, 0.1697,0.1730, 0.1753]
    # if NUMBER_OF_FEATURES == 12:
    #     mean = [0.6062, 0.5782, 0.5566, 0.5440, 0.5349, 0.5276, 0.5180, 0.5070, 0.4904, 0.4704, 0.4518, 0.4361]
    #     std = [0.1552, 0.1738, 0.1832, 0.1819, 0.1791, 0.1763, 0.1738, 0.1714, 0.1698,0.1731, 0.1751, 0.1738]

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

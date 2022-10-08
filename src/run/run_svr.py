from data.data_loader import DataLoader
from model.svr_model import SVRModel
from helper.numpy_helpers import load_np, save_np
from visualization.environment_oil_thickness_distribution import visualize_environment
import numpy as np

# Parameters
# ==================================================================================================================
FILE_NAME = 'thickness-9freqs-variance0.02'
DATA_PATH = f"assets/generated_data/variance_0.02_all_windspeeds/{FILE_NAME}"
OUTPUT_FILE_NAME = 'assets/generated_models/svr_highvariance_all_windspeeds_with_0'
PRED_IMG_DIR = "assets/generated_data/variance_0.02_all_windspeeds/fluids_cascaded_9freq/custom_pred/svr"
SAVE = False
LOAD = True
# ==================================================================================================================

loader = DataLoader()
possible_output_values = [(0, 10, 1)]
max_number_of_rows = 10000
loader.load_data_from_file(
    file_name=DATA_PATH,
    file_format="{}-{}.txt",
    possible_output_values=possible_output_values,
    max_number_of_rows=max_number_of_rows)

svr = SVRModel(data_loader=loader, kernel='rbf', C=1)
svr.load_model_data(number_of_classes=11, is_classification_problem=False)
if not LOAD:
    svr.train_model(OUTPUT_FILE_NAME, save_file=SAVE)
else:
    svr.load_model(OUTPUT_FILE_NAME)

# Evaluation

x_all = []
y_all = []
for i in range(2):
    x_all.append(
        load_np(f"assets/generated_data/variance_0.02_all_windspeeds/fluids_cascaded_9freq/custom_training/x{i}"))
    y_all.append(
        load_np(f"assets/generated_data/variance_0.02_all_windspeeds/fluids_cascaded_9freq/custom_training/ye{i}"))
    print(f"Loaded image {i}")

# svr.evaluate_metrics(x_all, y_all, PRED_IMG_DIR)

# Specific output evaluation

x = load_np(f"assets/generated_data/variance_0.02_all_windspeeds/fluids_cascaded_9freq/validation/x14")
ye = load_np(f"assets/generated_data/variance_0.02_all_windspeeds/fluids_cascaded_9freq/validation/ye14")


def pred(x):
    a = x.reshape(1, -1)
    return svr.predict(a)


y_pred = np.apply_along_axis(pred, 2, x)

y_pred = np.squeeze(y_pred)
y_pred = np.rint(y_pred)
y_pred[y_pred > 10] = 10
y_pred[y_pred < 0] = 0
preds = []
for i, row in enumerate(ye):
    for j, point in enumerate(row):
        if point == 3:
            preds.append(y_pred[i][j])
preds_np = np.array(preds)
save_np(preds_np, 'svr_3mm')

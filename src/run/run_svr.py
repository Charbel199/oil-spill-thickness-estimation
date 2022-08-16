from data.data_loader import DataLoader
from model.svr_model import SVRModel
from helper.numpy_helpers import load_np
from visualization.environment_oil_thickness_distribution import visualize_environment
import numpy as np



# Parameters
# ==================================================================================================================
FILE_NAME = 'thickness-9freqs-variance0.02'
DATA_PATH = f"assets/generated_data/variance_0.02_all_windspeeds/{FILE_NAME}"
OUTPUT_FILE_NAME = 'assets/generated_models/svr_highvariance_all_windspeeds_with_0'
PRED_IMG_DIR = "assets/generated_data/variance_0.02_all_windspeeds/fluids_cascaded_9freq/pred/svr"
SAVE = True
LOAD = False
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

x_all = []
y_all = []
for i in range(63):
    x_all.append(load_np(f"assets/generated_data/variance_0.02_all_windspeeds/fluids_cascaded_9freq/validation/x{i}"))
    y_all.append(load_np(f"assets/generated_data/variance_0.02_all_windspeeds/fluids_cascaded_9freq/validation/ye{i}"))
    print(f"Loaded image {i}")

# def pred(x):
#     a = x.reshape(1, -1)
#     return svr.predict(a)
#
#
# tt = np.apply_along_axis(pred, 2, input_data)
# tt = np.squeeze(tt)
# tt = np.rint(tt)
# print("DONE")
# visualize_environment(tt)
# visualize_environment(gt)

svr.evaluate_metrics(x_all, y_all, PRED_IMG_DIR)
print("DONE")

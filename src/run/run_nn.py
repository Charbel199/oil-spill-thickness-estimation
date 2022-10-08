from data.data_loader import DataLoader
from model.nn_model import NNModel
from visualization import error_bars, points_cloud
from visualization import environment_oil_thickness_distribution as e
from visualization.environments import generate_circle_environment
import numpy as np
from helper.numpy_helpers import save_np, load_np
# Parameters
# ==================================================================================================================
NEW_MODEL = False
network_layers = [
    ["Input", 9],
    ["Dense", 12, "relu"],
    ["Dense", 16, "relu"],
    ["Dense", 12, "relu"],
    ["Dense", 1, "linear"]
]
FILE_NAME = 'thickness-9freqs-variance0.02'
DATA_PATH = f"assets/generated_data/variance_0.02_all_windspeeds/{FILE_NAME}"
OUTPUT_FILE_NAME = 'assets/generated_models/ann_highvariance_all_windspeeds_with_0'
PRED_IMG_DIR = "assets/generated_data/variance_0.02_all_windspeeds/fluids_cascaded_9freq/custom_pred/ann"
SAVE = False
LOAD = True
# ==================================================================================================================


# Data

# loader = DataLoader()
# loader.load_data_from_file(
#     file_name=f"generated_data/{FILE_NAME}",
#     file_format="{}permittivity{}-{}.txt",
#     possible_output_values=[(2.8, 3.3, 0.1), (1, 10, 1)],
#     max_number_of_rows=10000)

loader = DataLoader()
possible_output_values = [(0, 10, 1)]
max_number_of_rows = 20000
loader.load_data_from_file(
    file_name=DATA_PATH,
    file_format="{}-{}.txt",
    possible_output_values=possible_output_values,
    max_number_of_rows=max_number_of_rows)


# Training and evaluation

model = NNModel(data_loader=loader, network_layers=network_layers, loss='mean_squared_error', print_summary=True)
model.load_model_data(test_size=0.1, is_classification_problem=False, normalize_output=False)
if NEW_MODEL:
    model.train_model(output_file_name=OUTPUT_FILE_NAME, save_file=True, epochs=30)
else:
    model.load_model(f"{OUTPUT_FILE_NAME}")

# General evaluation

# model.evaluate_model(model_name=OUTPUT_FILE_NAME, log_evaluation=True, include_classification_metrics=False)
x_all = []
y_all = []
for i in range(2):
    x_all.append(load_np(f"assets/generated_data/variance_0.02_all_windspeeds/fluids_cascaded_9freq/custom_training/x{i}"))
    y_all.append(load_np(f"assets/generated_data/variance_0.02_all_windspeeds/fluids_cascaded_9freq/custom_training/ye{i}"))
    print(f"Loaded image {i}")

model.evaluate_metrics(x_all, y_all, PRED_IMG_DIR)

# Specific output evaluation

x = load_np(f"assets/generated_data/variance_0.02_all_windspeeds/fluids_cascaded_9freq/validation/x14")
ye = load_np(f"assets/generated_data/variance_0.02_all_windspeeds/fluids_cascaded_9freq/validation/ye14")

x = np.array(x)
x = np.reshape(x, (-1, 9))
y_pred = (model.predict(x))
y_pred = np.reshape(y_pred, (100, 100, -1))
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
save_np(preds_np, 'ann_3mm')

# # Error bars and point clouds evaluation

# save_figs = False
# selected_permittivity = 3
# observed_values = model.y_test[:, 1]
# predicted_values = model.y_pred[:, 1]
# points_cloud.plot_cloud(observed_values, predicted_values, "Observed thickness (mm)", "Predicted thickness (mm)", save_fig=save_figs, output_file_name="ThicknessCloud")
# error_bars.generate_error_bars(observed_values, predicted_values, "Observed thickness (mm)", "Predicted thickness (mm)", save_fig=save_figs, output_file_name="ThicknessErrorBars")
#
# observed_values = model.y_test[:, 0]
# predicted_values = model.y_pred[:, 0]
# error_bars.generate_error_bars(observed_values, predicted_values, "Observed permittivity", "Predicted permittivity", save_fig=save_figs, output_file_name="PermittivityErrorBars")


# # Circle visualization

# env = generate_circle_environment(size=200, smallest_thickness=1, step_size=1)
# e.visualize_environment(env)
# # populated_env_permittivity = e.fill_environment_with_reflectivity_data_2_outputs(env, data_loader=loader, model=model, is_multi_output=True, selected_permittivity=selected_permittivity)
# # print(np.average(populated_env_permittivity))
# populated_env_thickness = e.fill_environment_with_reflectivity_data_2_outputs(env, data_loader=loader, model=model, is_multi_output=True, is_thickness=True,
#                                                                               selected_permittivity=selected_permittivity)
# # e.compare_environments(env, populated_env_permittivity, save_fig=save_figs, output_file_name="Spill vs Permittivity view")
# e.compare_two_environments(env, populated_env_thickness, save_fig=save_figs, output_file_name="Spill vs Thickness view")

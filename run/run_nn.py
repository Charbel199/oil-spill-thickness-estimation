from data.data_loader import DataLoader
from model.nn_model import NNModel
from visualization import error_bars, points_cloud
from visualization import environment_oil_thickness_distribution as e

# Initial parameters
file_name = 'thickness-9freqs-variance0.001-'
model_name = 'new-WITHOUT0-nn-v13-2outputs-thickness-9freqs-variance0.001-10000'
network_layers = [
    ["Input", 9],
    ["Dense", 12, "relu"],
    ["Dense", 16, "relu"],
    ["Dense", 12, "relu"],
    ["Dense", 2, "linear"]
]
new_model = True

# Data
loader = DataLoader()
loader.load_data_from_file(
    file_name=f"generated_data/{file_name}",
    file_format="{}permittivity{}-{}.txt",
    possible_output_values=[(2.8, 3.3, 0.1), (1, 10, 1)],
    max_number_of_rows=10000)

# Training and evaluation
model = NNModel(data_loader=loader, network_layers=network_layers, loss='mean_squared_error', print_summary=True)
model.load_model_data(test_size=0.2, is_classification_problem=False, normalize_output=False)
if new_model:
    model.train_model(output_file_name=model_name, save_file=True, epochs=5)
else:
    model.load_model(f"generated_models/{model_name}")
model.evaluate_model(model_name=model_name, log_evaluation=True, include_classification_metrics=False)
# size_to_view = 40
# y_pred = model.predict(model.x_test[0:size_to_view])
# for i in range(len(y_pred)):
#     print(f"{(model.y_test[i])} --> {(y_pred[i])}")


# Error bars and point clouds evaluation
save_figs = False
selected_permittivity = 3
observed_values = model.y_test[:, 1]
predicted_values = model.y_pred[:, 1]
points_cloud.plot_cloud(observed_values, predicted_values, "Observed thickness (mm)", "Predicted thickness (mm)", save_fig=save_figs, output_file_name="ThicknessCloud")
error_bars.generate_error_bars(observed_values, predicted_values, "Observed thickness (mm)", "Predicted thickness (mm)", save_fig=save_figs, output_file_name="ThicknessErrorBars")

observed_values = model.y_test[:, 0]
predicted_values = model.y_pred[:, 0]
error_bars.generate_error_bars(observed_values, predicted_values, "Observed permittivity", "Predicted permittivity", save_fig=save_figs, output_file_name="PermittivityErrorBars")

# Circle visualization
env = e.get_circle_thickness_distribution(size=200, smallest_thickness=1, step_size=1)
e.visualize_environment(env)
# populated_env_permittivity = e.fill_environment_with_reflectivity_data_2_outputs(env, data_loader=loader, model=model, is_multi_output=True, selected_permittivity=selected_permittivity)
# print(np.average(populated_env_permittivity))
populated_env_thickness = e.fill_environment_with_reflectivity_data_2_outputs(env, data_loader=loader, model=model, is_multi_output=True, is_thickness=True,
                                                                              selected_permittivity=selected_permittivity)
# e.compare_environments(env, populated_env_permittivity, save_fig=save_figs, output_file_name="Spill vs Permittivity view")
e.compare_two_environments(env, populated_env_thickness, save_fig=save_figs, output_file_name="Spill vs Thickness view")

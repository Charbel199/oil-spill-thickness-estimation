from data.data_loader import DataLoader
from model.nn_model import NNModel
from visualization import points_cloud, error_bars
from sklearn.metrics import mean_squared_error

file_name = 'thickness-4freqs-variance0.001'
network_layers = [
    ["Input", 4],
    ["Dense", 1, "relu"],
    ["Dense", 1, "relu"],
    ["Dense", 1, "relu"],
    ["Dense", 1, "linear"]
]
new_model = False

# Data
loader = DataLoader()
max_number_of_rows = 6000
loader.load_data_from_file(
    file_name=f"generated_data/{file_name}",
    file_format="{}-{}.txt",
    possible_output_values=[(1, 10, 1)],
    max_number_of_rows=max_number_of_rows)
print(loader.all_data_x.shape)
print(loader.all_data_y.shape)
mse_values = []
number_of_neurons = []
number_of_neurons_layer_1 = []
number_of_neurons_layer_2 = []
number_of_neurons_layer_3 = []
number_of_layers = []
for i in range(5, 9):
    for j in range(5, 9):
        for z in range(5, 9):

            nn_layers_text = ''

            network_layers[1][1] = i
            network_layers[2][1] = j
            network_layers[3][1] = z
            for layer in network_layers:
                nn_layers_text += str(layer[1]) + '-'

            print(f"Training model with layers: {network_layers}")
            model_name = f'compare-nn-1output-thickness-architecture-{nn_layers_text}4freqs-variance0.001-{max_number_of_rows}'
            # Training and evaluation
            model = NNModel(data_loader=loader, network_layers=network_layers, loss='mean_squared_error',
                            print_summary=True, metrics="mean_absolute_error")
            model.load_model_data(test_size=0.2, is_classification_problem=False, normalize_output=False)

            if not model.check_if_model_exists(f"generated_models/comparison/{model_name}.h5"):
                model.train_model(output_file_name=f"generated_models/comparison/{model_name}", save_file=True, epochs=5)
            else:
                model.load_model(f"generated_models/comparison/{model_name}")
            model.evaluate_model(model_name=model_name, log_evaluation=True, include_classification_metrics=False)

            size_to_view = 40
            y_pred = model.predict(model.x_test[0:size_to_view])
            for a in range(len(y_pred)):
                print(f"{(model.y_test[a])} --> {(y_pred[a])}")

            number_of_neurons.append(network_layers[1][1])
            number_of_neurons_layer_1.append(network_layers[1][1])
            number_of_neurons_layer_2.append(network_layers[2][1])
            number_of_neurons_layer_3.append(network_layers[3][1])
            mse_values.append(mean_squared_error(model.y_test, model.y_pred))
            number_of_layers.append(len(network_layers) - 2)
            # save_figs = False
            # observed_values = model.y_test[:, 0]
            # predicted_values = model.y_pred[:, 0]
            # points_cloud.plot_cloud(observed_values, predicted_values, "Observed thickness (mm)", "Predicted thickness (mm)", save_fig=save_figs, output_file_name="ThicknessCloud")
            # error_bars.generate_error_bars(observed_values, predicted_values, "Observed thickness (mm)", "Predicted thickness (mm)", save_fig=save_figs, output_file_name="ThicknessErrorBars")

import matplotlib.pyplot as plt
import numpy as np

# plt.grid()
# plt.plot(number_of_neurons, mse_values, '-x')
# plt.xlabel("Number of neurons")
# plt.ylabel("MSE")
# plt.show()


# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(number_of_neurons_layer_1, number_of_neurons_layer_2, mse_values,
#            linewidths=1, alpha=.7,
#            edgecolor='k',
#            s=200,
#            c=mse_values)
# ax.set_xlabel('Layer 1 Neurons')
# ax.set_ylabel('Layer 2 Neurons')
# ax.set_zlabel('MSE')
#
# ax.set_xticks([5, 6, 7, 8])
# ax.set_yticks([5, 6, 7, 8])
#
# ax.set_zticks([0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
# plt.show()
# for i in range(len(mse_values)):
#     print(f"For nn: 4-{number_of_neurons_layer_1[i]}-{number_of_neurons_layer_2[i]}-1 -> MSE: {mse_values[i]}")


for i in range(len(mse_values)):
    print(f"For nn: 4-{number_of_neurons_layer_1[i]}-{number_of_neurons_layer_2[i]}-{number_of_neurons_layer_3[i]}-1 -> MSE: {mse_values[i]}")

from data.data_loader import DataLoader
from model.nn_model import NNModel
from visualization import points_cloud, error_bars
from sklearn.metrics import mean_squared_error

file_name = 'thickness-4freqs-variance0.001'
network_layers = [
    ["Input", 4],
    ["Dense", 1, "relu"],
    # ["Dense", 1, "relu"],
    # ["Dense", 1, "relu"],
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
number_of_layers =[]
for i in range(2, 15):

    nn_layers_text = ''

    network_layers[1][1] = i
    # network_layers[2][1] = i
    # network_layers[3] [1] = i
    for layer in network_layers:
        nn_layers_text += str(layer[1]) + '-'

    print(f"Training model with layers: {network_layers}")
    model_name = f'compare-nn-1output-thickness-architecture-{nn_layers_text}4freqs-variance0.001-{max_number_of_rows}'
    # Training and evaluation
    model = NNModel(data_loader=loader, network_layers=network_layers, loss='mean_squared_error', print_summary=True, metrics="mean_absolute_error")
    model.load_model_data(test_size=0.2, is_classification_problem=False, normalize_output=False)

    if not model.check_if_model_exists(f"generated_models/comparison/{model_name}.h5"):
        model.train_model(output_file_name=model_name, save_file=True,  epochs=5)
    else:
        model.load_model(f"generated_models/comparison/{model_name}")
    model.evaluate_model(model_name=model_name, log_evaluation=True,  include_classification_metrics=False)

    size_to_view = 40
    y_pred = model.predict(model.x_test[0:size_to_view])
    for j in range(len(y_pred)):
        print(f"{(model.y_test[j])} --> {(y_pred[j])}")

    number_of_neurons.append(network_layers[1][1])
    mse_values.append(mean_squared_error(model.y_test, model.y_pred))
    number_of_layers.append(len(network_layers)-2)
    # save_figs = False
    # observed_values = model.y_test[:, 0]
    # predicted_values = model.y_pred[:, 0]
    # points_cloud.plot_cloud(observed_values, predicted_values, "Observed thickness (mm)", "Predicted thickness (mm)", save_fig=save_figs, output_file_name="ThicknessCloud")
    # error_bars.generate_error_bars(observed_values, predicted_values, "Observed thickness (mm)", "Predicted thickness (mm)", save_fig=save_figs, output_file_name="ThicknessErrorBars")

import matplotlib.pyplot as plt
import numpy as np
plt.grid()
plt.plot(number_of_neurons, mse_values, '-x')
plt.xlabel("Number of neurons")
plt.ylabel("MSE")
plt.show()


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# # Data for three-dimensional scattered points
# zdata = np.array(mse_values)
# xdata = np.array(number_of_neurons)
# ydata = np.array(number_of_layers)
# print(zdata)
# print(xdata)
# print(ydata)
# ax.scatter3D(xdata, ydata, zdata, linewidth=0.2, antialiased=True);
# ax.set_xlabel('Number of neurons')
# ax.set_ylabel('Number of layers')
# ax.set_zlabel('MSE')
# plt.show()
import numpy as np

from visualization.environment_oil_thickness_distribution import fill_environment_with_reflectivity_data, \
    visualize_environment
from visualization.environments import generate_circle_environment, generate_fractal_environment
from data.data_loader import DataLoader
from helper.numpy_helpers import *
from model.nn_model import NNModel

file_name = 'thickness-4freqs-variance0.001'

# Data
loader = DataLoader()
max_number_of_rows = 10000
loader.load_data_from_file(
    file_name=f"../generated_data/{file_name}",
    file_format="{}-{}.txt",
    possible_output_values=[(1, 10, 1)],
    max_number_of_rows=max_number_of_rows)
network_layers = [
    ["Input", 4],
    ["Dense", 5, "relu"],
    ["Dense", 6, "relu"],
    ["Dense", 6, "relu"],
    ["Dense", 1, "linear"]
]

# env = generate_fractal_environment(shape=(100, 100), res=(2, 2), octaves=1)
#
# visualize_environment(environment=env,
#                       save_fig=False,
#                       show_fig=True)
# reflect_env = fill_environment_with_reflectivity_data(environment=env, data_loader=loader)
#
# visualize_environment(environment=reflect_env[:, :, 0],
#                       save_fig=False,
#                       show_fig=True)
# visualize_environment(environment=reflect_env[:, :, 1],
#                       save_fig=False,
#                       show_fig=True)
# visualize_environment(environment=reflect_env[:, :, 2],
#                       save_fig=False,
#                       show_fig=True)
# visualize_environment(environment=reflect_env[:, :, 3],
#                       save_fig=False,
#                       show_fig=True)


for i in range(20):
    env = generate_fractal_environment(shape=(100, 100), res=(2, 2), octaves=1)
    reflect_env = fill_environment_with_reflectivity_data(environment=env, data_loader=loader)
    save_np(reflect_env, f'../generated_data/fractals/validation/x{i}')
    save_np(env, f'../generated_data/fractals/validation/y{i}')
    visualize_environment(environment=env,
                          output_file_name=f'../generated_data/fractals/validation/y{i}',
                          save_fig=True,
                          show_fig=False,
                          file_type='jpeg')





# model_name = 'compare-nn-1output-thickness-architecture-4-5-6-6-1-4freqs-variance0.001-6000'
# model = NNModel(data_loader=loader, network_layers=network_layers, loss='mean_squared_error', print_summary=True)
# model.load_model(f"../generated_models/comparison/{model_name}")
# env  = load_np('../generated_data/fractals/x1')
# output = np.zeros((env.shape[0],env.shape[1]))
#
# for i in range(len(env)):
#     print(f"Line {i}")
#
#     to_predict = env[i]
#     output[i,:] = np.squeeze(model.predict(to_predict))
# visualize_environment(environment=output,show_fig=True,save_fig=False)
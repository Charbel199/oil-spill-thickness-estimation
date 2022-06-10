import os.path

from visualization.environment_oil_thickness_distribution import fill_environment_with_reflectivity_data, \
    visualize_environment
from visualization.environments import generate_fractal_environment
from data.data_loader import DataLoader
from helper.numpy_helpers import *

# Parameters
# ==================================================================================================================
FILE_NAME = 'thickness-4freqs-variance0.02'
SHOW_REFLECTIVITIES_PLOTS = False
RES = (2, 2)
OCTAVES = 2
OUTPUT_SHAPE = (100, 100)
SMALLEST_THICKNESS = 1
LARGEST_THICKNESS = 10
NUMBER_OF_DATA_POINTS = 20
FOR_TRAINING = False
OUTPUT_FOLDER_PATH = f"assets/generated_data/variance_0.02/fractals/{'training' if FOR_TRAINING else 'validation'}"
# ==================================================================================================================


# Data
loader = DataLoader()
max_number_of_rows = 10000
loader.load_data_from_file(
    file_name=f"assets/generated_data/variance_0.02/{FILE_NAME}",
    file_format="{}-{}.txt",
    possible_output_values=[(1, 10, 1)],
    max_number_of_rows=max_number_of_rows)

if SHOW_REFLECTIVITIES_PLOTS:
    env = generate_fractal_environment(smallest_thickness=SMALLEST_THICKNESS, largest_thickness=LARGEST_THICKNESS,
                                       shape=OUTPUT_SHAPE, res=RES, octaves=OCTAVES)

    visualize_environment(environment=env,
                          save_fig=False,
                          show_fig=True)
    reflect_env = fill_environment_with_reflectivity_data(environment=env, data_loader=loader)

    visualize_environment(environment=reflect_env[:, :, 0],
                          save_fig=False,
                          show_fig=True)
    visualize_environment(environment=reflect_env[:, :, 1],
                          save_fig=False,
                          show_fig=True)
    visualize_environment(environment=reflect_env[:, :, 2],
                          save_fig=False,
                          show_fig=True)
    visualize_environment(environment=reflect_env[:, :, 3],
                          save_fig=False,
                          show_fig=True)

for i in range(NUMBER_OF_DATA_POINTS):
    env = generate_fractal_environment(smallest_thickness=SMALLEST_THICKNESS, largest_thickness=LARGEST_THICKNESS,
                                       shape=OUTPUT_SHAPE, res=RES, octaves=OCTAVES)
    reflect_env = fill_environment_with_reflectivity_data(environment=env, data_loader=loader)
    save_np(reflect_env, os.path.join(OUTPUT_FOLDER_PATH, f'x{i}'))
    save_np(env, os.path.join(OUTPUT_FOLDER_PATH, f'y{i}'))
    visualize_environment(environment=env,
                          output_file_name=os.path.join(OUTPUT_FOLDER_PATH, f'y{i}'),
                          save_fig=True,
                          show_fig=False,
                          file_type='jpeg')
    print(f"Saved image {i}")

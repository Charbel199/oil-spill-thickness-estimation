import os.path

from visualization.environment_oil_thickness_distribution import fill_environment_with_reflectivity_data, \
    visualize_environment
from visualization.environments import generate_fractal_environment
from data.data_loader import DataLoader
from helper.numpy_helpers import *

# Parameters
# ==================================================================================================================
FILE_NAME = 'thickness-4freqs-variance0.02'
DATA_PATH = f"assets/generated_data/variance_0.02_windspeed_1/{FILE_NAME}"
SHOW_REFLECTIVITIES_PLOTS = False
RES = (1, 1)
OCTAVES = 2
OUTPUT_SHAPE = (100, 100)
SMALLEST_THICKNESS = 0
LARGEST_THICKNESS = 10
STARTING_POINT = 150
NUMBER_OF_DATA_POINTS = 150
FOR_TRAINING = True
IS_CLASSIFICATION = True
CLASSIFICATION_ONLY = False
INVERTED = True
OUTPUT_FOLDER_PATH = f"assets/generated_data/variance_0.02_windspeed_1/fractals_with_0_cascaded/{'training' if FOR_TRAINING else 'validation'}"
# ==================================================================================================================


# Data
loader = DataLoader()
max_number_of_rows = 10000
loader.load_data_from_file(
    file_name=DATA_PATH,
    file_format="{}-{}.txt",
    possible_output_values=[(0, 10, 1)],
    max_number_of_rows=max_number_of_rows)

if SHOW_REFLECTIVITIES_PLOTS:
    env = generate_fractal_environment(smallest_thickness=SMALLEST_THICKNESS, largest_thickness=LARGEST_THICKNESS,
                                       shape=OUTPUT_SHAPE, res=RES, octaves=OCTAVES, inverted=INVERTED)
    if IS_CLASSIFICATION:
        env[env != 0] = 1
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

for i in range(STARTING_POINT, STARTING_POINT+NUMBER_OF_DATA_POINTS):
    env = generate_fractal_environment(smallest_thickness=SMALLEST_THICKNESS, largest_thickness=LARGEST_THICKNESS,
                                       shape=OUTPUT_SHAPE, res=RES, octaves=OCTAVES, inverted=INVERTED)
    reflect_env = fill_environment_with_reflectivity_data(environment=env, data_loader=loader)
    save_np(reflect_env, os.path.join(OUTPUT_FOLDER_PATH, f'x{i}'))
    if IS_CLASSIFICATION:
        classification_env = env.copy()
        classification_env[classification_env != 0] = 1
        save_np(classification_env, os.path.join(OUTPUT_FOLDER_PATH, f'yc{i}'))
        if CLASSIFICATION_ONLY:
            continue
    save_np(env, os.path.join(OUTPUT_FOLDER_PATH, f'ye{i}'))
    visualize_environment(environment=env,
                          output_file_name=os.path.join(OUTPUT_FOLDER_PATH, f'y{i}'),
                          save_fig=True,
                          show_fig=False,
                          file_type='jpeg')

    print(f"Saved image {i}")

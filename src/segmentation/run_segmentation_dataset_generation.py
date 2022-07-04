import os.path

from visualization.environment_oil_thickness_distribution import fill_environment_with_reflectivity_data, \
    visualize_environment
from visualization.environments import generate_fractal_environment
from data.data_loader import DataLoader
from helper.numpy_helpers import *
import glob

# Parameters
# ==================================================================================================================
FILE_NAME = 'thickness-4freqs-variance0.02'
DATA_PATH = f"assets/generated_data/variance_0.02/{FILE_NAME}"
SHOW_REFLECTIVITIES_PLOTS = False
RES = (1, 1)
OCTAVES = 2
OUTPUT_SHAPE = (100, 100)
SMALLEST_THICKNESS = 0
LARGEST_THICKNESS = 10
STARTING_POINT = 0
NUMBER_OF_DATA_POINTS = 150
FOR_TRAINING = True
IS_CLASSIFICATION = True
CLASSIFICATION_ONLY = False
INVERTED = False
TEXT_DIRECTORY = "assets/generated_map"
OUTPUT_FOLDER_PATH = f"assets/generated_data/variance_0.02/fluids_with_0_cascaded/{'training' if FOR_TRAINING else 'validation'}"
# ==================================================================================================================

loader = DataLoader()
max_number_of_rows = 10000
loader.load_data_from_file(
    file_name=DATA_PATH,
    file_format="{}-{}.txt",
    possible_output_values=[(0, 10, 1)],
    max_number_of_rows=max_number_of_rows)


def generate_fractals_segmentation_dataset():
    # Data
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

    for i in range(STARTING_POINT, STARTING_POINT + NUMBER_OF_DATA_POINTS):
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


def generate_fractals_segmentation_dataset_from_text():
    dataset_path = TEXT_DIRECTORY + "/*.csv"
    for i, file in enumerate(glob.glob(dataset_path)):
        env = np.genfromtxt(file,
                            delimiter=',')
        env = np.rint(env)
        env = env.astype('uint8')

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


if __name__ == "__main__":
    generate_fractals_segmentation_dataset_from_text()
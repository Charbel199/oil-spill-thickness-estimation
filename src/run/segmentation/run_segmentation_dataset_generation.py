import os.path

from visualization.environment_oil_thickness_distribution import fill_environment_with_reflectivity_data, \
    visualize_environment
from visualization.environments import generate_fractal_environment
from data.data_loader import DataLoader
from helper.numpy_helpers import *
import glob

# Parameters
# ==================================================================================================================

# INPUT: Data coming from MATLAB simulation (in txt format)
FILE_NAME = 'thickness-9freqs-variance0.02'
DATA_PATH = f"assets/generated_data/real_data_updated/vibrations/{FILE_NAME}"
# INPUT: Thickness distribution coming from Oil Spill Simulation (in csv format)
CSV_DIRECTORY = "assets/generated_map/val"
# OUTPUT: Output directory for the generated data
OUTPUT_FOLDER_PATH = f"assets/generated_data/real_data_updated/val_with_vibrations"

SHOW_REFLECTIVITIES_PLOTS = False
RES = (1, 1)
OCTAVES = 2
INVERTED = False

OUTPUT_SHAPE = (100, 100)
SMALLEST_THICKNESS = 0
LARGEST_THICKNESS = 10

STARTING_POINT = 0
NUMBER_OF_DATA_POINTS = 60

FOR_TRAINING = True
IS_CLASSIFICATION = True
CLASSIFICATION_ONLY = False



# Inputs: Parameters - CSV DIRECTORY - OUTPUT DIRECTORY - DATA PATH (Reflectivities path Ex: 10 txt file 0 -> 10 mm)
# Takes CSV thickness distribution, map reflectivities based on files generated from matlab and outputs inputs and ouputs
# Output: In the output directory -> x0 ye0 yc0 ... x1 ye1 yc1 ...
# ==================================================================================================================


loader = DataLoader()
possible_output_values = [(0, 10, 1)]
max_number_of_rows = 20000
loader.load_data_from_file(
    file_name=DATA_PATH,
    file_format="{}-{}.txt",
    possible_output_values=possible_output_values,
    max_number_of_rows=max_number_of_rows)


def generate_npy_segmentation_dataset_from_text():
    dataset_path = CSV_DIRECTORY + "/*.csv"
    for i, file in enumerate(glob.glob(dataset_path)):
        if i > NUMBER_OF_DATA_POINTS:
            break
        env = np.genfromtxt(file,
                            delimiter=',')
        env = np.rint(env)
        env = env.astype('uint8')

        reflect_env = fill_environment_with_reflectivity_data(environment=env, data_loader=loader,
                                                              possible_output_values=possible_output_values[0])
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
                              file_type='svg')

        print(f"Saved image {i}")


if __name__ == "__main__":
    generate_fractals_segmentation_dataset_from_text()

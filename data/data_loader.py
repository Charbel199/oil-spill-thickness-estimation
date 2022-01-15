import sys

import numpy as np


class DataLoader(object):
    def __init__(self):
        self.all_thicknesses_data: np.ndarray = np.array()
        self.all_thicknesses: np.ndarray = np.array()

    def load_data(self,
                  synthetic_data_file_name: str,
                  smallest_thickness: int = 0,
                  largest_thickness: int = 10,
                  step_size: int = 1,
                  max_number_of_rows: int = sys.maxsize,
                  real_data_file_name: str = '',
                  lower_bound_thickness_for_real_data: int = 0,
                  upper_bound_thickness_for_real_data: int = 0):

        # Getting data from files
        all_thicknesses_data = []
        for thickness in range(smallest_thickness, largest_thickness + 1, step_size):
            if lower_bound_thickness_for_real_data != 0 and upper_bound_thickness_for_real_data != 0 and \
                    (lower_bound_thickness_for_real_data <= thickness <= upper_bound_thickness_for_real_data):
                file = open(f"{real_data_file_name}{str(thickness)}.txt", "rt")
            else:
                file = open(f"{synthetic_data_file_name}{str(thickness)}.txt", "rt")

            temp_thickness_data = []
            for line in file:
                number_strings = line.split()  # Split the line on runs of whitespace
                try:
                    numbers = [float(n) for n in number_strings]  # Convert to integers
                except Exception:
                    continue
                temp_thickness_data.append(numbers)
                if len(temp_thickness_data) > max_number_of_rows:
                    break

            all_thicknesses_data.append(temp_thickness_data)  # Add the thickness data
            file.close()

        all_thicknesses_data = np.array(all_thicknesses_data)
        print(all_thicknesses_data.shape)
        self.all_thicknesses_data = all_thicknesses_data
        self.all_thicknesses = np.array(range(smallest_thickness, largest_thickness + 1, step_size))

    def save_data(self,
                  file_name: str):
        np.save(f'{file_name}_x_data.npy', self.all_thicknesses_data)
        np.save(f'{file_name}_y_data.npy', self.all_thicknesses)

    def load_data(self,
                  file_name: str):
        try:
            self.all_thicknesses_data = np.array(np.load(f'{file_name}_x_data.npy'))
            self.all_thicknesses = np.array(np.load(f'{file_name}_y_data.npy'))
        except OSError:
            print('Files not found')

import sys

import numpy as np


class DataLoader(object):
    def __init__(self):
        self.all_data_x: np.ndarray = np.array([])
        self.all_data_y: np.ndarray = np.array([])

    def load_data(self,
                  synthetic_data_file_name: str,
                  smallest_value: float = 0,
                  largest_value: float = 10,
                  step_size: float = 1,
                  max_number_of_rows: int = sys.maxsize,
                  real_data_file_name: str = '',
                  lower_bound_for_real_data: int = 0,
                  upper_bound_for_real_data: int = 0):

        # Getting data from files
        all_data_x = []
        all_data_y = []
        value = smallest_value
        values = []
        while value <= largest_value:
            value = round(value, 2)
            if int(value) == value:
                value = int(value)
            values.append(value)
            value += step_size
        for val in values:
            if lower_bound_for_real_data != 0 and upper_bound_for_real_data != 0 and \
                    (lower_bound_for_real_data <= val <= upper_bound_for_real_data):
                file = open(f"{real_data_file_name}{str(val)}.txt", "rt")
            else:
                file = open(f"{synthetic_data_file_name}{str(val)}.txt", "rt")

            counter = 0
            for line in file:
                number_strings = line.split()  # Split the line on runs of whitespace
                try:
                    numbers = [float(n) for n in number_strings]  # Convert to float
                except Exception:
                    continue
                counter += 1
                all_data_x.append(numbers)
                all_data_y.append(val)
                if counter > max_number_of_rows:
                    break

            file.close()

        all_data = np.array(all_data_x)
        print(all_data.shape)
        self.all_data_x = all_data
        self.all_data_y = np.array(all_data_y)

    def save_data(self,
                  file_name: str):
        np.save(f'{file_name}_x_data.npy', self.all_data_x)
        np.save(f'{file_name}_y_data.npy', self.all_data_y)

    def load_existing_data(self,
                           file_name: str):
        try:
            self.all_data_x = np.array(np.load(f'{file_name}_x_data.npy'))
            self.all_data_y = np.array(np.load(f'{file_name}_y_data.npy'))
        except OSError:
            print('Files not found')

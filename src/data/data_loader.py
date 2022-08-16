import sys
import numpy as np
from typing import List, Tuple
import itertools
from sklearn.utils import shuffle


class DataLoader(object):
    def __init__(self):
        self.all_data_x: np.ndarray = np.array([])
        self.all_data_y: np.ndarray = np.array([])

    def load_data_from_file(self,
                            file_name: str,
                            file_format: str,
                            possible_output_values: List[Tuple],
                            max_number_of_rows: int = sys.maxsize,
                            ):

        print("Loading data from files ...")
        # Getting data from files
        all_data_x = []
        all_data_y = []
        # Values list such as: [[1,2,3,4,5,6,7,8,9,10],[2.8,2.9,3.0,3.1,3.2]]
        outputs_list = []
        # Possible output values should be formatted as (smallest_value, largest_value, step_size)
        for possible_output_value in possible_output_values:

            smallest_value, largest_value, step_size = possible_output_value

            value = smallest_value
            # List of possible outputs
            values = []
            while value <= largest_value:
                if int(value) == value:
                    value = int(value)
                values.append(value)
                value += step_size
                value = round(value, 2)

            outputs_list.append(values)

        # All output combinations [(1,2.8),(1,2.9),(1,3.0),...,(2,3.1),...]
        all_output_combinations = list(itertools.product(*outputs_list))

        # Load data from files
        for val in all_output_combinations:
            file = open(file_format.format(file_name, *map(lambda x: str(x), val)), "rt")
            counter = 0
            for line in file:
                number_strings = line.split()  # Split the line on runs of whitespace
                try:
                    numbers = [float(n) for n in number_strings]  # Convert to float
                except Exception:
                    continue
                counter += 1
                all_data_x.append(numbers)
                all_data_y.append(list(val))
                if counter >= max_number_of_rows:
                    break

            file.close()

        self.all_data_x = np.array(all_data_x)
        self.all_data_y = np.array(all_data_y)
        self.all_data_x, self.all_data_y = shuffle(self.all_data_x, self.all_data_y, random_state=0)
        print("Loaded data from files")

    def save_data(self,
                  file_name: str):
        # Save np arrays
        print("Saving x and y data ...")
        np.save(f'{file_name}_x_data.npy', self.all_data_x)
        np.save(f'{file_name}_y_data.npy', self.all_data_y)
        print(f"Saved x and y data in {file_name}_<x||y>_data.npy")

    def load_data(self,
                  file_name: str):
        # Load np arrays
        print(f"Loading x and y data from {file_name}_<x||y>_data.npy ...")
        try:
            self.all_data_x = np.array(np.load(f'{file_name}_x_data.npy'))
            self.all_data_y = np.array(np.load(f'{file_name}_y_data.npy'))
            print("Loaded x and y data")
        except OSError:
            print('Files not found, could not load x and y data ...')

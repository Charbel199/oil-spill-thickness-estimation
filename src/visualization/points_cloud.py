from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


def plot_cloud(observed_values,
               predicted_values,
               x_label, y_label,
               save_fig=False,
               output_file_name="test"):
    raw_values: Dict[float, List] = {}
    mean_values: Dict[float, float] = {}
    std_values: Dict[float, float] = {}

    for index, o in enumerate(observed_values):
        if o in raw_values:
            raw_values[o].append(predicted_values[index])
        else:
            raw_values[o] = [predicted_values[index]]

    for key, values in raw_values.items():
        mean_values[key] = sum(values) / len(values)
        std_values[key] = np.std(np.array(values))

    plt.grid()
    plt.plot(observed_values, predicted_values, marker='x', markersize=1, linestyle='None')
    plt.plot(list(raw_values.keys()), list(raw_values.keys()), "r-")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    file_type = 'svg'
    if save_fig:
        plt.savefig(f'{output_file_name}.{file_type}', format=file_type, dpi=300)
    plt.show()

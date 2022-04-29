import numpy as np;
from typing import List, Dict
import matplotlib.pyplot as plt

np.random.seed(42)


def plot_error_bars(x, y, y_error,
                    x_label, y_label,
                    output_file_name: str = 'test',
                    file_type: str = 'svg',
                    save_fig: bool = False,
                    show_fig: bool = True):
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.errorbar(x, y, yerr=y_error,
                 capsize=2,
                 marker="x", markersize=4,
                 linestyle="none",
                 markeredgewidth=1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.grid()
    plt.xticks(np.linspace(min(x), max(x), len(x)))
    plt.yticks(np.linspace(min(x), max(x), len(x)))
    if save_fig:
        plt.savefig(f'{output_file_name}.{file_type}', format=file_type, dpi=300)
    if show_fig:
        plt.show()


def generate_error_bars(observed_values,
                        predicted_values,
                        x_label,
                        y_label,
                        save_fig=False,
                        output_file_name="test"):
    raw_values: Dict[float, List] = {}
    mean_values: Dict[float, float] = {}
    std_values: Dict[float, float] = {}

    # Add raw values, example: "1":[0.91,0.99,1.01,1.01] All predicted values for the observed values
    for index, o in enumerate(observed_values):
        if o in raw_values:
            raw_values[o].append(predicted_values[index])
        else:
            raw_values[o] = [predicted_values[index]]

    # Compute mean and std
    for key, values in raw_values.items():
        mean_values[key] = sum(values) / len(values)
        std_values[key] = np.std(np.array(values))

    plot_error_bars(
        list(raw_values.keys()),
        (mean_values.values()),
        list(std_values.values()),
        x_label, y_label,
        output_file_name=output_file_name,
        save_fig=save_fig)


if __name__ == "__main__":
    x = [1, 2, 3]
    y = [2, 3, 4]
    error = [1, 1, 1]
    plot_error_bars(x, y, error)

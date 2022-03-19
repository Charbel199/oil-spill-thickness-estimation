import numpy as np;

np.random.seed(42)
import matplotlib.pyplot as plt


def plot_error_bars(x, y, y_error,
                    output_file_name: str = 'test',
                    file_type: str = 'svg',
                    fig_size: int = 25,
                    font_size: int = 25,
                    save_fig: bool = False,
                    show_fig: bool = True):
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.errorbar(x, y, yerr=y_error,
                 capsize=2,
                 marker="x", markersize=4,
                 linestyle="none",
                 markeredgewidth=1)
    plt.xlabel("Observed permittivity")
    plt.ylabel("Predicted permittivity")

    plt.grid()
    plt.xticks(np.linspace(min(x), max(x), len(x)))
    plt.yticks(np.linspace(min(x), max(x), len(x)))
    if save_fig:
        plt.savefig(f'{output_file_name}.{file_type}', format=file_type, dpi=300)
    if show_fig:
        plt.show()


if __name__ == "__main__":
    x = [1, 2, 3]
    y = [2, 3, 4]
    error = [1, 1, 1]
    plot_error_bars(x, y, error)

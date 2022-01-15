import numpy as np
from matplotlib import pyplot as plt
from data.data_loader import DataLoader
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_circle_thickness_distribution(
        size: int = 500,
        radius_step_size: float = 4 / 100,
        smallest_thickness: int = 0,
        largest_thickness: int = 10,
        step_size: int = 1,
) -> np.ndarray:
    def fill_circle(circle_environment, circle_thickness, circle_radius, circle_center):
        for x in range(len(circle_environment)):
            for y in range(len(circle_environment[0])):
                # 2D Circle formula
                if ((x - circle_center) ** 2 + (y - circle_center) ** 2) < (circle_radius ** 2):
                    circle_environment[x][y] = circle_thickness
        return circle_environment

    environment = np.zeros(shape=(size, size))
    center = int(len(environment) / 2)
    radius = center
    for thickness in range(smallest_thickness, largest_thickness + 1, step_size):
        environment = fill_circle(environment, thickness, radius, center)
        radius -= int(radius_step_size * size)
    return environment


def fill_environment_with_reflectivity_data(
        environment: np.ndarray,
        data_loader: DataLoader
) -> np.ndarray:
    populated_environment = []
    for x in range(len(environment)):
        temp_populated_environment = []
        for y in range(len(environment[x])):
            thickness = environment[x][y]
            thickness_index = np.where(data_loader.all_thicknesses == thickness)[0]
            # Get thickness index
            random_data_point_index = random.randint(0, len(data_loader.all_thicknesses_data[thickness_index]))
            ref = data_loader.all_thicknesses_data[thickness_index][random_data_point_index]
            temp_populated_environment.append(ref)

        populated_environment.append(temp_populated_environment)

    populated_environment = np.array(populated_environment)
    return populated_environment


def visualize_environment(
        environment: np.ndarray,
        output_file_name: str = 'test',
        file_type: str = 'svg',
        fig_size: int = 25,
        font_size: int = 25,
        save_fig: bool = False,
        show_fig: bool = True
):
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': font_size}

    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.2)
    im = ax.imshow(environment, cmap='jet')
    fig.colorbar(im, cax=cax, orientation='vertical')
    if save_fig:
        plt.savefig(f'{output_file_name}.{file_type}', format=file_type)
    if show_fig:
        plt.show()


if __name__ == "__main__":
    env = get_circle_thickness_distribution()
    visualize_environment(env)

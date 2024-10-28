import numpy as np
from matplotlib import pyplot as plt
from data.data_loader import DataLoader
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
from visualization.environments import generate_circle_environment


def get_all_index(data_loader: DataLoader, possible_output_value=(0, 10, 1)):
    smallest_value, largest_value, step_size = possible_output_value
    thickness_index = {}
    for thickness in np.arange(smallest_value, largest_value + step_size, step_size):
        thickness_index[thickness] = np.where(data_loader.all_data_y == thickness)[0]
    return thickness_index


def fill_environment_with_reflectivity_data(
        environment: np.ndarray,
        data_loader: DataLoader,
        possible_output_values=(0, 10, 1)
) -> np.ndarray:
    thickness_index = get_all_index(data_loader, possible_output_values)

    def fill_point(x):
        return np.array(data_loader.all_data_x[thickness_index[x][random.randint(0, len(thickness_index[x]) - 1)]])

    g = np.vectorize(fill_point, otypes=[np.ndarray])
    populated_environment = np.array(g(environment).tolist())

    return populated_environment


def fill_environment_with_reflectivity_data_2_outputs(
        environment: np.ndarray,
        data_loader: DataLoader,
        model,
        permittivity_index=0,
        thickness_index=1,
        selected_permittivity=3.3,
        is_multi_output=False,
        is_thickness=False
) -> np.ndarray:
    populated_environment = []

    if is_multi_output:
        loaded_thicknesses = model.y_test[np.where(model.y_test[:, permittivity_index] == selected_permittivity)]
    else:
        loaded_thicknesses = model.y_test
    loaded_x = model.x_test[np.where(model.y_test[:, permittivity_index] == selected_permittivity)]
    for x in range(len(environment)):
        temp_populated_environment = []
        for y in range(len(environment[x])):
            thickness = environment[x][y]
            if (isinstance(thickness, float)):
                thickness = round(thickness, 2)
            possible_thickness_indices = np.where(loaded_thicknesses[:, thickness_index] == thickness)[0]
            # Get thickness index
            # print(f"thickness: {thickness}, thickness index: {possible_thickness_indices}, len: {len(loaded_x[possible_thickness_indices]) - 1}")
            random_data_point_index = random.randint(0, len(loaded_x[possible_thickness_indices]) - 1)
            test = loaded_x[possible_thickness_indices][random_data_point_index]
            temp_populated_environment.append(test)

        if is_thickness:
            predicted_values = model.predict(np.array(temp_populated_environment))[:, thickness_index]
        else:
            predicted_values = model.predict(np.array(temp_populated_environment))[:, permittivity_index]
        populated_environment.append(predicted_values)

    populated_environment = np.array(populated_environment)
    return populated_environment


def visualize_environment(
        environment: np.ndarray,
        output_file_name: str = 'run_test',
        file_type: str = 'svg',
        fig_size: int = 25,
        font_size: int = 25,
        save_fig: bool = False,
        show_fig: bool = True,
        cmap: str = 'jet'
):
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': font_size}

    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.2)
    im = ax.imshow(environment, cmap=cmap)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(3)
    if save_fig:
        plt.savefig(f'{output_file_name}.{file_type}', format=file_type)
    if show_fig:
        plt.show()
    else:
        plt.close(fig)


def compare_two_environments(
        environment: np.ndarray,
        populated_environment: np.ndarray,
        output_file_name: str = 'run_test',
        file_type: str = 'svg',
        fig_size_width: int = 25,
        fig_size_height: int = 20,
        font_size: int = 32,
        save_fig: bool = False,
        show_fig: bool = True
):
    # Setup
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(fig_size_width, fig_size_height))
    font = {'family': 'normal',
            'weight': 'regular',
            'size': font_size}

    plt.rc('font', **font)

    # Left side
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='7%', pad=0.4)
    im = axs[0].imshow(environment, cmap='jet')
    axs[0].grid()
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Right side
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='7%', pad=0.4)
    im = axs[1].imshow(populated_environment, cmap='jet')
    axs[1].grid()
    fig.colorbar(im, cax=cax, orientation='vertical')

    if save_fig:
        plt.savefig(f'{output_file_name}.{file_type}', format=file_type)
    if show_fig:
        plt.show()
    else:
        plt.close(fig)


def compare_three_environments(
        environment1: np.ndarray,
        environment2: np.ndarray,
        environment3: np.ndarray,
        output_file_name: str = 'run_test',
        file_type: str = 'svg',
        fig_size_width: int = 25,
        fig_size_height: int = 20,
        font_size: int = 32,
        save_fig: bool = False,
        show_fig: bool = True
):
    # Setup
    fig = plt.figure(figsize=(fig_size_width, fig_size_height))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)
    # fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(fig_size_width, fig_size_height))
    font = {'family': 'normal',
            'weight': 'regular',
            'size': font_size}

    plt.rc('font', **font)

    # Left side
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='7%', pad=0.4)
    im = ax1.imshow(environment1, cmap='jet')
    ax1.grid()
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Right side
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='7%', pad=0.4)
    im = ax2.imshow(environment2, cmap='jet')
    ax2.grid()
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Down side
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='7%', pad=0.4)
    im = ax3.imshow(environment3, cmap='jet')
    ax3.grid()
    fig.colorbar(im, cax=cax, orientation='vertical')

    if save_fig:
        plt.savefig(f'{output_file_name}.{file_type}', format=file_type)
    if show_fig:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    env = generate_circle_environment()
    visualize_environment(env)
    compare_two_environments(env, env)

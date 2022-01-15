import numpy as np
from matplotlib import pyplot as plt
from data.data_parser import DataParser
import random


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
        data_parser: DataParser
) -> np.ndarray:
    populated_environment = []
    for x in range(len(environment)):
        temp_populated_environment = []
        for y in range(len(environment[x])):
            thickness = environment[x][y]
            thickness_index = np.where(data_parser.all_thicknesses == thickness)[0]
            # Get thickness index
            random_data_point_index = random.randint(0, len(data_parser.all_thicknesses_data[thickness_index]))
            ref = data_parser.all_thicknesses_data[thickness_index][random_data_point_index]
            temp_populated_environment.append(ref)

        populated_environment.append(temp_populated_environment)

    populated_environment = np.array(populated_environment)
    return populated_environment


if __name__ == "__main__":
    env = get_circle_thickness_distribution()
    plt.imshow(env)
    plt.show()
import numpy as np
from matplotlib import pyplot as plt


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


if __name__ == "__main__":
    env = get_circle_thickness_distribution()
    plt.imshow(env)
    plt.show()

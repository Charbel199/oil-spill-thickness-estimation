import numpy as np;
from typing import List, Dict
import matplotlib.pyplot as plt
from itertools import chain
np.random.seed(42)
from helper.numpy_helpers import load_np

def plot_histograms(x1: List,
                    x2: List,
                    x3: List,
                    x4: List,
                    output_file_name: str = 'run_test',
                    file_type: str = 'svg',
                    fig_size_width: int = 25,
                    fig_size_height: int = 20,
                    font_size: int = 32,
                    save_fig: bool = False,
                    show_fig: bool = True):
    # Setup
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(fig_size_width, fig_size_height))
    # font = {'family': 'normal',
    #         'weight': 'regular',
    #         'size': font_size}
    #
    # plt.rc('font', **font)
    from collections import Counter


    plt.setp(axs, xticks=range(11), xticklabels=range(11))

    count = Counter(x1)
    for i in range(11):
        if i not in count.keys():
            count[i] = 0
    to_add = []
    for key in sorted(count):
        to_add.append(count[key])

    im1 = axs[0][0].bar(range(11), to_add,align='center')
    axs[0][0].grid()
    count = Counter(x2)
    for i in range(11):
        if i not in count.keys():
            count[i] = 0
    to_add = []
    for key in sorted(count):
        to_add.append(count[key])
    im2 = axs[0][1].bar(range(11), to_add,align='center')
    axs[0][1].grid()
    count = Counter(x3)
    for i in range(11):
        if i not in count.keys():
            count[i] = 0
    to_add = []
    for key in sorted(count):
        to_add.append(count[key])
    im3 = axs[1][0].bar(range(11), to_add,align='center')
    axs[1][0].grid()
    count = Counter(x4)
    for i in range(11):
        if i not in count.keys():
            count[i] = 0
    to_add = []
    for key in sorted(count):
        to_add.append(count[key])
    im4 = axs[1][1].bar(range(11), to_add,align='center')
    axs[1][1].grid()


    fig.text(0.5, 0.04, 'Estimated Thickness (mm)', ha='center', fontsize=font_size)
    fig.text(0.04, 0.5, 'Number of points', va='center', rotation='vertical',fontsize=font_size)

    axs = list(chain.from_iterable(axs))
    axs[0].set_title('SVR', fontsize=font_size)
    axs[1].set_title('ANN', fontsize=font_size)
    axs[2].set_title('Regular U-Net', fontsize=font_size)
    axs[3].set_title('Cascaded U-Net', fontsize=font_size)
    for a in axs:
        a.tick_params(axis='both', which='major', labelsize=font_size)
        a.patch.set_edgecolor('black')
        a.patch.set_linewidth('3')
        a.set_ylim([0, 600])

    if save_fig:
        plt.savefig(f'{output_file_name}.{file_type}', format=file_type)
    if show_fig:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    thickness = 3
    x4 = load_np(f'cascaded_unet_{thickness}mm').astype(int)
    x3 = load_np(f'unet_{thickness}mm').astype(int)
    x2 = load_np(f'ann_{thickness}mm').astype(int)
    x1 = load_np(f'svr_{thickness}mm').astype(int)
    # x = np.random.normal(170, 10, 250)
    plot_histograms(x1.tolist(),x2.tolist(),x3.tolist(),x4.tolist(), save_fig=True, output_file_name='histo_3mm',file_type='svg')


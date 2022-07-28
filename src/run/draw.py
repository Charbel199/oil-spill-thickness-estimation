import numpy as np
import matplotlib.pyplot as plt

def iouVSnumoffreq():
    font_size = 15
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': font_size}

    # plt.rc('font', **font)
    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    x = range(2, 13)
    y = [0.626, 0.655, 0.62, 0.69, 0.70, 0.71, 0.76, 0.773, 0.73, 0.757, 0.753]

    fig, ax = plt.subplots()
    ax.plot(x, y, 'o-')

    ax.set_xticks(np.arange(2, 13, 1))
    ax.set_yticks(np.arange(0.55, 0.9, 0.05))

    ax.set_xlabel("Number of frequencies")
    ax.set_ylabel("IoU metric")

    ax.grid()
    plt.show()
    fig.savefig('IOUvsNumOfFreqs.svg')

def heightVSWindSpeed():
    font_size = 15
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': font_size}

    # plt.rc('font', **font)
    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


    x = range(2, 17,2)
    y = [7e-4 ,7e-4, 8e-4, 1e-3 ,1.5e-3 ,1.6e-3 ,1.7e-3, 1.8e-3]
    y2 = [2.5e-2 ,1e-1, 2.3e-1, 4e-1 ,6.5e-1 ,8.6e-1 ,1.2, 1.8]

    fig, ax = plt.subplots()
    ax.plot(x, y2, 'o-', label='Gravity')
    ax.plot(x, y, 'x-', label='Capillary (Elf)')

    ax.set_yscale('log')
    ax.set_xticks(np.arange(2, 17,2))
    # ax.set_yticks(np.arange(0.55, 0.9, 0.05))

    ax.set_xlabel("Wind speed at 10 meters above sea (m/s)")
    ax.set_ylabel("RMS height (m)")
    ax.legend()
    fig.tight_layout()
    ax.grid()
    plt.show()

    fig.savefig('heightVSWindSpeed.svg',format='svg')

def iouVSwindspeed():
    font_size = 15
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': font_size}

    # plt.rc('font', **font)
    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    x = range(2, 17, 2)
    y = [0.772, 0.771, 0.76, 0.75, 0.59, 0.56, 0.527, 0.484]

    fig, ax = plt.subplots()
    ax.plot(x, y, 'o-')

    ax.set_xticks(np.arange(2, 17 ,2))
    ax.set_yticks(np.arange(0.45, 0.9, 0.05))

    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("IoU metric")

    ax.grid()
    plt.show()
    fig.savefig('IOUvsWindspeed.svg')

heightVSWindSpeed()
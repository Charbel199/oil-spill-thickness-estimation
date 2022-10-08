import numpy as np
import matplotlib.pyplot as plt


def iou_vs_num_of_freq_plot():
    font_size = 15
    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels

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


def height_vs_windspeed_plot():
    font_size = 15

    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels

    x = range(2, 17, 2)
    y = [7e-4, 7e-4, 8e-4, 1e-3, 1.5e-3, 1.6e-3, 1.7e-3, 1.8e-3]
    y2 = [2.5e-2, 1e-1, 2.3e-1, 4e-1, 6.5e-1, 8.6e-1, 1.2, 1.8]

    fig, ax = plt.subplots()
    ax.plot(x, y2, 'o-', label='Gravity')
    ax.plot(x, y, 'x-', label='Capillary (Elf)')

    ax.set_yscale('log')
    ax.set_xticks(np.arange(2, 17, 2))

    ax.set_xlabel("Wind speed at 10 meters above sea (m/s)")
    ax.set_ylabel("RMS height (m)")
    ax.legend()
    fig.tight_layout()
    ax.grid()
    plt.show()

    fig.savefig('heightVSWindSpeed.svg', format='svg')


def iou_vs_windspeed_plot():
    font_size = 15

    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels

    x = range(2, 17, 2)
    y = [0.772, 0.771, 0.76, 0.75, 0.59, 0.56, 0.527, 0.484]

    fig, ax = plt.subplots()
    ax.plot(x, y, 'o-')

    ax.set_xticks(np.arange(2, 17, 2))
    ax.set_yticks(np.arange(0.45, 0.9, 0.05))

    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("IoU metric")

    ax.grid()
    plt.show()
    fig.savefig('IOUvsWindspeed.svg')


def iou_vs_windspeed_per_label_plot():
    x = range(11)
    y1 = [0.9467461229391183, 0.766781381002132, 0.823618299818493, 0.7665253984133632, 0.7467722060806236,
          0.8211951534538136, 0.7939392956531452, 0.7932351580325937, 0.7875876448104826, 0.7896569302839361,
          0.9219788154433634]
    y2 = [0.9466800315290109, 0.7719117772637785, 0.8274364852465578, 0.7686673177642502, 0.7471195809773367,
          0.8183628579021907, 0.7890643741876476, 0.7917349035413993, 0.790432623953487, 0.7972062636316832,
          0.9295713226919095]
    y3 = [0.9273479197278065, 0.674352741390375, 0.7888802066720505, 0.8182861250227563, 0.7726733590798374,
          0.7955599281631849, 0.7911790027337945, 0.806637859525865, 0.7659073127138357, 0.7729008331619809,
          0.9260754403867046]
    y4 = [0.813129821685056903, 0.61098967927364553, 0.52687397903640834, 0.7475633566988273, 0.789973187641989,
          0.655983165817449, 0.552810869129107, 0.6350854895719081, 0.6437910900534256, 0.6192558929546557,
          0.8562335613811837]
    font_size = 12

    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels

    fig, ax = plt.subplots()
    ax.plot(x, y1, 'o-', label='2 m/s')
    ax.plot(x, y2, 'o--', label='4 m/s')
    ax.plot(x, y3, 'o-.', label='6 m/s')
    ax.plot(x, y4, 'o:', label='8 m/s')

    ax.set_xticks(range(11))
    ax.set_yticks(np.arange(0.5, 1, 0.05))

    ax.set_xlabel("Thickness (mm)")
    ax.set_ylabel("IoU")
    plt.legend(loc="lower right")
    ax.grid()
    plt.show()
    fig.savefig('IOUvsWindspeedPerLabel.svg')


def detection_probability_plot():
    x = range(1, 11)
    y1 = [0.3352375, 0.801625, 0.998, 0.9999875, 1, 0.99997500, 0.99966250, 0.99758750, 0.99493750, 0.997250]
    y2 = [0.936, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    font_size = 12

    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels

    fig, ax = plt.subplots()
    ax.plot(x, y1, 'x-', label='Joint PDF', color='orange')
    ax.plot(x, y2, 'x--', label='Cascaded U-Net')

    ax.set_xticks(range(1, 11))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    ax.set_xlabel("Thickness (mm)")
    ax.set_ylabel("Probability of detection")
    plt.legend(loc="lower right")
    ax.grid()
    plt.show()
    fig.savefig('probabilityOfDetection.svg')


def detection_and_estimation_vs_freq_plot():
    x = range(2, 13)
    y1 = [0.53, 0.70, 0.69, 0.73, 0.73, 0.71, 0.76, 0.81, 0.81, 0.83, 0.82]
    y2 = [0.7, 0.8, 0.93, 0.939, 0.919, 0.95, 0.89, 0.94, 0.95, 0.95, 0.94]

    font_size = 12

    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels

    fig, ax = plt.subplots()
    l1, = ax.plot(x, y1, 'x-', label='Oil Thickness Estimation')

    ax.set_xticks(range(2, 13))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    ax.set_xlabel("Number of frequencies")
    ax.set_ylabel("Thickness Estimation IoU")

    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    l2, = ax2.plot(x, y2, 'x--', label='Oil Detection', color='orange')
    ax2.set_ylabel("Detection accuracy", rotation=-90, labelpad=17)
    ax2.set_yticks(np.arange(0, 1.1, 0.1))

    plt.legend([l1, l2], ["Oil Thickness Estimation", "Oil Detection"], loc="lower right")
    ax.grid()
    plt.show()
    fig.savefig('detectionAndEstimationVsFreq.svg')


if __name__ == "__main__":
    detection_probability_plot()
    detection_and_estimation_vs_freq_plot()
    iou_vs_num_of_freq_plot()
    iou_vs_windspeed_per_label_plot()
    iou_vs_windspeed_per_label_plot()
    height_vs_windspeed_plot()

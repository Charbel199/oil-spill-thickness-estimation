import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
from visualization.environment_oil_thickness_distribution import visualize_environment
from model.base_semantic_segmentation_model import SemanticSegmentationModel
from metrics.pixel_wise_iou import pixel_wise_iou
from metrics.pixel_wise_recall import pixel_wise_recall
from metrics.pixel_wise_precision import pixel_wise_precision
from metrics.pixel_wise_dice import pixel_wise_dice
from metrics.pixel_wise_accuracy import pixel_wise_accuracy
from visualization.histograms import plot_histograms
from helper.numpy_helpers import save_np
from helper.general_helpers import avg_list
import glob
import shutil

# absolute path to search all text files inside a specific folder
path = '/home/charbel199/PycharmProjects/Oil-Spill-Thickness-Estimation/src/assets/generated_data/variance_0.02_all_windspeeds/fluids_cascaded_9freq/validation/*.npy'
files = glob.glob(path)

ye_files = [k for k in files if 'ye' in k]
print(len(ye_files))

path='/home/charbel199/projs/U-2-Net/test_data/data/pred/*.npy'
files = glob.glob(path)

y_pred_files = [k for k in files if 'x' in k]
print(len(y_pred_files))
iou = []
iou_per_class = []
recall = []
precision = []
accuracy = []
dice = []
index = 0

for ye_file in ye_files:
    last_part = ye_file.split('/')[-1]
    num = ''.join([n for n in last_part if n.isdigit()])
    print(num)
    yp_file = [file for file in y_pred_files if f"x{num}.npy" in file][0]

    print(f"yefile {ye_file}")
    print(f"ypfile {yp_file}")

    y_true = np.load(ye_file)
    y_pred = np.squeeze(np.load(yp_file))
    y_pred = np.rint(y_pred)
    y_pred = y_pred.astype('uint8')
    # visualize_environment(y_true)
    # visualize_environment(y_pred)

    print(y_pred)
    print(f"Sizes {y_pred.shape} / {y_true.shape} ")
    iou.append(pixel_wise_iou(y_true, y_pred))
    iou_per_class.append(pixel_wise_iou(y_true, y_pred, per_label=True))
    accuracy.append(pixel_wise_accuracy(y_true, y_pred))
    dice.append(pixel_wise_dice(y_true, y_pred))
    precision.append(pixel_wise_precision(y_true, y_pred))
    recall.append(pixel_wise_recall(y_true, y_pred))
    index += 1

print(f"Average iou coefficient: {sum(iou) / len(iou)}")
print(f"Average dice coefficient: {sum(dice) / len(dice)}")
print(f"Average precision coefficient: {sum(precision) / len(precision)}")
print(f"Average recall coefficient: {sum(recall) / len(recall)}")
print(f"Average accuracy coefficient: {sum(accuracy) / len(accuracy)}")
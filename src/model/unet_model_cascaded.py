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


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class SemanticSegmentationCascadedModel:
    def __init__(
            self, classifier, estimator
    ):
        self.classifier: SemanticSegmentationModel = classifier
        self.estimator: SemanticSegmentationModel = estimator
        self.batch_count = 0

    def train_fn(self, loader, opt_classifier, opt_estimator, opt_all, criterion_classifier, criterion_estimator,
                 summary_writer,
                 combined_loss=False, device='cpu'):
        # loop = tqdm(loader)

        for batch_idx, (data, targets) in enumerate(loader):
            self.batch_count += 1
            classification_targets, estimator_targets = targets
            data = data.to(device=device)
            classification_targets.to(device=device)
            estimator_targets.to(device=device)
            BATCH_SIZE = data.shape[0]

            classified_output = self.classifier(data)
            loss_classifier = criterion_classifier(classified_output.squeeze(), classification_targets)
            # print(f"Classifier loss {loss_classifier}")
            if summary_writer is not None:
                summary_writer.add_scalar("classifier loss", loss_classifier, self.batch_count)
            if not combined_loss:
                opt_classifier.zero_grad()
                loss_classifier.backward(retain_graph=True)
                opt_classifier.step()

            estimator_input = torch.cat((data, classified_output.detach()), dim=1)
            estimated_output = self.estimator(estimator_input)
            loss_estimator = criterion_estimator(estimated_output, estimator_targets)
            # print(f"Estimator loss {loss_estimator}")
            if summary_writer is not None:
                summary_writer.add_scalar("estimator loss", loss_estimator, self.batch_count)
            if not combined_loss:
                opt_estimator.zero_grad()
                loss_estimator.backward(retain_graph=True)
                opt_estimator.step()

            total_loss = 0.8 * loss_classifier + loss_estimator
            # print(f"Total loss {total_loss}")
            if summary_writer is not None:
                summary_writer.add_scalar("total loss", total_loss, self.batch_count)
            if combined_loss:
                opt_all.zero_grad()
                total_loss.backward(retain_graph=True)
                opt_all.step()
            summary_writer.flush()
            # print(f"Done batch {batch_idx}")

            # loop.set_postfix(loss=total_loss.item())

    def save_predictions_as_images(self, loader, folder, device="cuda"):

        self.classifier.eval()
        self.estimator.eval()

        with torch.no_grad():
            for idx, (x, y) in enumerate(loader):
                yc, ye = y
                x.to(device)
                yc.to(device)
                ye.to(device)
                classification = self.classifier(x)

                estimator_input = torch.cat((x, classification), dim=1)
                estimation = self.estimator(estimator_input)

                # Format classification results
                classification = self.classifier.process_prediction(classification)
                # Format estimation results
                estimation = self.estimator.process_prediction(estimation)

                index = idx * estimation.shape[0]

                for pred in estimation:
                    visualize_environment(environment=pred, save_fig=True, show_fig=False,
                                          output_file_name=f"{folder}/pred_estimation_{index}", file_type='png')
                    index += 1
                    print(f"Saved estimation environment {index}")
                index = idx * classification.shape[0]
                for pred in classification:
                    visualize_environment(environment=pred, save_fig=True, show_fig=False,
                                          output_file_name=f"{folder}/pred_classification_{index}", file_type='png')
                    index += 1
                    print(f"Saved classification environment {index}")

        self.classifier.train()
        self.estimator.train()

    def evaluate_metrics(self, loader, device='cuda'):
        self.classifier.eval()
        self.estimator.eval()

        with torch.no_grad():
            iou = []
            iou_per_class = []
            recall = []
            precision = []
            accuracy = []
            dice = []

            iou_classification = []
            recall_classification = []
            precision_classification = []
            accuracy_classification = []
            dice_classification = []

            for idx, (x, y) in enumerate(loader):
                yc, ye = y
                x.to(device)
                yc.to(device)
                ye.to(device)
                classification = self.classifier(x)

                estimator_input = torch.cat((x, classification), dim=1)
                estimation = self.estimator(estimator_input)

                # Format classification results
                classification = self.classifier.process_prediction(classification)
                # Format estimation results
                estimation = self.estimator.process_prediction(estimation)

                index = idx * estimation.shape[0]

                for i, pred in enumerate(estimation):
                    y_true = ye[i].numpy()
                    y_pred = pred.numpy()
                    iou.append(pixel_wise_iou(y_true, y_pred))
                    iou_per_class.append(pixel_wise_iou(y_true, y_pred, per_label=True))
                    accuracy.append(pixel_wise_accuracy(y_true, y_pred))
                    dice.append(pixel_wise_dice(y_true, y_pred))
                    precision.append(pixel_wise_precision(y_true, y_pred))
                    recall.append(pixel_wise_recall(y_true, y_pred))
                    index += 1

                index = idx * classification.shape[0]
                pred_detect = {
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0,
                    9: 0,
                    10: 0
                }
                total_detect = {
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0,
                    9: 0,
                    10: 0
                }
                for i, pred in enumerate(classification):
                    y_true = yc[i].numpy()
                    ye_true = ye[i].numpy()
                    y_pred = pred.numpy()
                    y_pred[y_pred >= 0.5] = 1
                    y_pred[y_pred < 0.5] = 0

                    for i, row in enumerate(y_pred):
                        for j, point in enumerate(row):
                            thickness = ye_true[i][j]
                            if thickness == 0:
                                continue
                            total_detect[thickness] += 1
                            if y_pred[i][j] == 1:
                                pred_detect[thickness] += 1

                    # visualize_environment(y_true, cmap='Greys')
                    # visualize_environment(y_pred, cmap='Greys', save_fig=True, output_file_name="./test",file_type='svg')
                    iou_classification.append(pixel_wise_iou(y_true, y_pred))
                    accuracy_classification.append(pixel_wise_accuracy(y_true, y_pred))
                    dice_classification.append(pixel_wise_dice(y_true, y_pred))
                    precision_classification.append(pixel_wise_precision(y_true, y_pred))
                    recall_classification.append(pixel_wise_recall(y_true, y_pred))
                    index += 1

        for key in list(total_detect.keys()):
            print(f"Avg for {key} is {(pred_detect[key] / total_detect[key]) * 100}")
        avg_iou_per_class = []
        for i in range(11):
            temp = []
            for iou in iou_per_class:
                if len(iou) < 11:
                    continue
                temp.append(iou[i])

            avg_iou_per_class.append(avg_list(temp))

        print(f"Average iou classification per label coefficient: {avg_iou_per_class}")

        print(f"Average iou classification coefficient: {sum(iou_classification) / len(iou_classification)}")
        print(f"Average dice classification coefficient: {sum(dice_classification) / len(dice_classification)}")
        print(
            f"Average precision classification coefficient: {sum(precision_classification) / len(precision_classification)}")
        print(f"Average recall classification coefficient: {sum(recall_classification) / len(recall_classification)}")
        print(
            f"Average accuracy classification coefficient: {sum(accuracy_classification) / len(accuracy_classification)}")

        print(f"Average iou coefficient: {sum(iou) / len(iou)}")
        print(f"Average dice coefficient: {sum(dice) / len(dice)}")
        print(f"Average precision coefficient: {sum(precision) / len(precision)}")
        print(f"Average recall coefficient: {sum(recall) / len(recall)}")
        print(f"Average accuracy coefficient: {sum(accuracy) / len(accuracy)}")

        self.classifier.train()
        self.estimator.train()

    def predict(self, x):
        x = x.float()
        # x = torch.moveaxis(x, -1, 0)
        # x = x[None,:]

        classification = self.classifier(x)
        # classification = self.classifier.process_prediction(classification)
        estimator_input = torch.cat((x, classification), dim=1)
        estimation = self.estimator(estimator_input)
        estimation = self.estimator.process_prediction(estimation)
        return estimation

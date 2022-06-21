import torch.nn as nn
import torch
from tqdm import tqdm
from visualization.environment_oil_thickness_distribution import visualize_environment
from model.base_semantic_segmentation_model import SemanticSegmentationModel
from metrics.iou import iou_coefficient


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


class SemanticSegmentationCascadedModel():
    def __init__(
            self, classifier, estimator
    ):
        self.classifier: SemanticSegmentationModel = classifier
        self.estimator: SemanticSegmentationModel = estimator

    def train_fn(self, loader, opt_classifier, opt_estimator, opt_all, criterion_classifier, criterion_estimator,
                 combined_loss=False, device='cpu'):
        # loop = tqdm(loader)

        for batch_idx, (data, classification_targets, estimator_targets) in enumerate(loader):
            BATCH_SIZE = data.shape[0]

            classified_output = self.classifier(data)
            loss_classifier = criterion_classifier(classified_output.squeeze(), classification_targets)
            print(f"Classifier loss {loss_classifier}")
            if not combined_loss:
                opt_classifier.zero_grad()
                loss_classifier.backward(retain_graph=True)
                opt_classifier.step()

            estimator_input = torch.cat((data, classified_output.detach()), dim=1)
            estimated_output = self.estimator(estimator_input)
            loss_estimator = criterion_estimator(estimated_output, estimator_targets)
            print(f"Estimator loss {loss_estimator}")
            if not combined_loss:
                opt_estimator.zero_grad()
                loss_estimator.backward(retain_graph=True)
                opt_estimator.step()

            total_loss = 0.8 * loss_classifier + loss_estimator
            print(f"Total loss {total_loss}")
            if combined_loss:
                opt_all.zero_grad()
                total_loss.backward(retain_graph=True)
                opt_all.step()

            print(f"Done batch {batch_idx}")

            # loop.set_postfix(loss=total_loss.item())

    def save_predictions_as_images(self, loader, folder, device="cuda"):

        self.classifier.eval()
        self.estimator.eval()

        with torch.no_grad():
            for idx, (x, yc, ye) in enumerate(loader):
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
                                          output_file_name=f"{folder}/pred_estimation_{index}", file_type='jpeg')
                    index += 1
                    print(f"Saved estimation environment {index}")
                index = idx * classification.shape[0]
                for pred in classification:
                    visualize_environment(environment=pred, save_fig=True, show_fig=False,
                                          output_file_name=f"{folder}/pred_classification_{index}", file_type='jpeg')
                    index += 1
                    print(f"Saved classification environment {index}")

        self.classifier.train()
        self.estimator.train()

    def evaluate_metrics(self, loader):
        self.classifier.eval()
        self.estimator.eval()

        with torch.no_grad():
            iou_coefficients = []

            for idx, (x, yc, ye) in enumerate(loader):
                classification = self.classifier(x)

                estimator_input = torch.cat((x, classification), dim=1)
                estimation = self.estimator(estimator_input)

                # Format classification results
                classification = self.classifier.process_prediction(classification)
                # Format estimation results
                estimation = self.estimator.process_prediction(estimation)

                index = idx * estimation.shape[0]

                for pred in estimation:
                    iou_coefficients.append(iou_coefficient(ye.numpy(), pred.numpy()))
                    index += 1

                index = idx * classification.shape[0]
                for pred in classification:
                    index += 1

        print(f"Average iou coefficient: {sum(iou_coefficients) / len(iou_coefficients)}")

        self.classifier.train()
        self.estimator.train()

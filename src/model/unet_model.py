import torch
from model.base_semantic_segmentation_model import SemanticSegmentationModel
from metrics.pixel_wise_iou import pixel_wise_iou
from metrics.pixel_wise_recall import pixel_wise_recall
from metrics.pixel_wise_precision import pixel_wise_precision
from metrics.pixel_wise_dice import pixel_wise_dice
from metrics.pixel_wise_accuracy import pixel_wise_accuracy
from torchsummary import summary
from helper.general_helpers import avg_list

class UNET(SemanticSegmentationModel):

    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], normalize_output=False, device="cuda"
    ):
        super().__init__(in_channels, out_channels, features, normalize_output, device)
       # summary(self, (10, 100,100))

    def process_prediction(self, predictions):
        predictions = torch.nn.functional.softmax(predictions, dim=1)
        predictions = torch.argmax(predictions, dim=1)
        return predictions

    def evaluate_metrics(self, loader, num_classes=11):
        self.eval()


        with torch.no_grad():
            iou = []
            iou_per_class = []
            recall = []
            precision = []
            accuracy = []
            dice = []

            for idx, (x, y) in enumerate(loader):
                predictions = self(x.to(self.device))
                predictions = self.process_prediction(predictions)

                index = idx * predictions.shape[0]

                for i, pred in enumerate(predictions):
                    y_true = y[i].numpy()
                    y_pred = pred.cpu().numpy()
                    iou_per_class.append(pixel_wise_iou(y_true, y_pred, per_label=True))
                    iou.append(pixel_wise_iou(y_true, y_pred))
                    accuracy.append(pixel_wise_accuracy(y_true, y_pred))
                    dice.append(pixel_wise_dice(y_true, y_pred))
                    precision.append(pixel_wise_precision(y_true, y_pred))
                    recall.append(pixel_wise_recall(y_true, y_pred))
                    index += 1

        avg_iou_per_class = []
        for i in range(num_classes):
            temp = []
            for iou in iou_per_class:
                if len(iou) < num_classes:
                    continue
                temp.append(iou[i])

            avg_iou_per_class.append(avg_list(temp))

        print(f"Average iou classification per label coefficient: {avg_iou_per_class}")
        print(f"Average iou coefficient: {sum(iou) / len(iou)}")
        print(f"Average dice coefficient: {sum(dice) / len(dice)}")
        print(f"Average precision coefficient: {sum(precision) / len(precision)}")
        print(f"Average recall coefficient: {sum(recall) / len(recall)}")
        print(f"Average accuracy coefficient: {sum(accuracy) / len(accuracy)}")

        self.train()


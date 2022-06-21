import torch
from model.base_semantic_segmentation_model import SemanticSegmentationModel
from metrics.iou import iou_coefficient


class UNET(SemanticSegmentationModel):

    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], normalize_output=False
    ):
        super().__init__(in_channels, out_channels, features, normalize_output)

    def process_prediction(self, predictions):
        predictions = torch.nn.functional.softmax(predictions, dim=1)
        predictions = torch.argmax(predictions, dim=1)
        return predictions

    def evaluate_metrics(self, loader):
        self.eval()


        with torch.no_grad():
            iou_coefficients = []

            for idx, (x, y) in enumerate(loader):
                predictions = self(x)
                predictions = self.process_prediction(predictions)

                index = idx * predictions.shape[0]

                for i, pred in enumerate(predictions):
                    iou_coefficients.append(iou_coefficient(y[i].numpy(), pred.numpy()))
                    index += 1

        print(f"Average iou coefficient: {sum(iou_coefficients) / len(iou_coefficients)}")

        self.train()


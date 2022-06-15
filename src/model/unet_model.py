import torch
from model.base_semantic_segmentation_model import SemanticSegmentationModel


class UNET(SemanticSegmentationModel):

    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], normalize_output=False
    ):
        super().__init__(in_channels, out_channels, features, normalize_output)

    def process_prediction(self, predictions):
        predictions = torch.nn.functional.softmax(predictions, dim=1)
        predictions = torch.argmax(predictions, dim=1)
        return predictions

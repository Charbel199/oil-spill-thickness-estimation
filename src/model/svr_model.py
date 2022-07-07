from data.data_loader import DataLoader
from model.base_model import Model
from sklearn.svm import SVR
import pickle
from metrics.pixel_wise_iou import pixel_wise_iou
from metrics.pixel_wise_recall import pixel_wise_recall
from metrics.pixel_wise_precision import pixel_wise_precision
from metrics.pixel_wise_dice import pixel_wise_dice
from metrics.pixel_wise_accuracy import pixel_wise_accuracy
import numpy as np
from visualization.environment_oil_thickness_distribution import visualize_environment


class SVRModel(Model):
    def __init__(self, data_loader: DataLoader, **kwargs):
        super().__init__(data_loader, **kwargs)
        self.model = self.create_svr()

    def train_model(self,
                    output_file_name: str,
                    output_file_extension: str = "sav",
                    save_file: bool = False,
                    batch_size=20,
                    epochs=10):
        # Train model
        self.model.fit(self.x_train, self.y_train)
        print('Done training...')

        # Saving file
        if save_file:
            self.save_model(f"{output_file_name}", extension=output_file_extension)

    def save_model(self, output_file_name: str, extension: str = "sav"):
        pickle.dump(self.model, open(f"{output_file_name}.{extension}", 'wb'))
        print(f'Saved model in {output_file_name}.{extension}')

    def load_model(self, file_name: str, extension: str = "sav"):
        self.model = pickle.load(open(f'{file_name}.{extension}', 'rb'))
        print(f'Loaded model from {file_name}.{extension}')

    def evaluation_signature(self) -> str:
        return f"SVR: C={self.C}, kernel={self.kernel}, epsilon={self.epsilon}"

    def parse_args(self, **kwargs):
        self.kernel = kwargs.get('kernel', "rbf")
        self.C = kwargs.get('C', 100)
        self.epsilon = kwargs.get('epsilon', 0.1)

    def create_svr(self):
        model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, verbose=True)
        return model

    def evaluate_metrics(self, x_all, y_all, folder):

        iou = []
        recall = []
        precision = []
        accuracy = []
        dice = []

        for j in range(len(x_all)):
            x = x_all[j]
            ye = y_all[j]

            def pred(x):
                a = x.reshape(1, -1)
                return self.predict(a)

            y_pred = np.apply_along_axis(pred, 2, x)
            y_pred = np.squeeze(y_pred)
            y_pred = np.rint(y_pred)
            y_pred[y_pred > 10] = 10
            y_pred[y_pred < 0] = 0

            y_true = ye
            iou.append(pixel_wise_iou(y_true, y_pred))
            accuracy.append(pixel_wise_accuracy(y_true, y_pred))
            dice.append(pixel_wise_dice(y_true, y_pred))
            precision.append(pixel_wise_precision(y_true, y_pred))
            recall.append(pixel_wise_recall(y_true, y_pred))
            visualize_environment(environment=y_pred, save_fig=True, show_fig=False,
                                  output_file_name=f"{folder}/pred_estimation_{j}", file_type='jpeg')
            print(f"Done image {j}")

        print(f"Average iou coefficient: {sum(iou) / len(iou)}")
        print(f"Average dice coefficient: {sum(dice) / len(dice)}")
        print(f"Average precision coefficient: {sum(precision) / len(precision)}")
        print(f"Average recall coefficient: {sum(recall) / len(recall)}")
        print(f"Average accuracy coefficient: {sum(accuracy) / len(accuracy)}")

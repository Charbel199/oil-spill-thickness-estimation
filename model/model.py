from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics as skmetrics
from data.data_loader import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plot
from sklearn import preprocessing
from abc import abstractmethod


class Model(object):
    def __init__(self,
                 data_loader: DataLoader):
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.data_loader = data_loader
        self.model = None

    def load_model_data(self,
                        test_size=0.2,
                        random_state=42,
                        is_classification_problem=True,
                        number_of_classes=10,
                        normalize_output=False,
                        normalize_input=False,
                        normalize_range=None
                        ):
        print("Loading model data ...")

        # Normalize input
        if normalize_input:
            if normalize_range is None:
                normalize_range = (0, 1)
            self.minmax_scale = preprocessing.MinMaxScaler(feature_range=normalize_range)
            self.minmax_scale.fit(self.data_loader.all_data_y)
            self.data_loader.all_data_x = self.minmax_scale.transform(self.data_loader.all_data_x)
        # Normalize output
        if normalize_output:
            if normalize_range is None:
                normalize_range = (0, 1)
            self.minmax_scale = preprocessing.MinMaxScaler(feature_range=normalize_range)
            self.minmax_scale.fit(self.data_loader.all_data_y)
            self.data_loader.all_data_y = self.minmax_scale.transform(self.data_loader.all_data_y)

        # Classification preprocessing
        if is_classification_problem:
            self.data_loader.all_data_y = self.preprocess_y_data(self.data_loader.all_data_y, number_of_classes)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data_loader.all_data_x,
                                                                                self.data_loader.all_data_y,
                                                                                test_size=test_size,
                                                                                random_state=random_state)
        print("Training data loaded")

    @abstractmethod
    def train_model(self,
                    output_file_name: str,
                    output_file_extension: str = "h5",
                    save_file: bool = False,
                    epochs=10):
        pass

    @abstractmethod
    def save_model(self, output_file_name: str, extension: str = "h5"):
        pass

    @abstractmethod
    def load_model(self, file_name: str, extension: str = "h5"):
        pass

    @staticmethod
    def preprocess_y_data(y_data: np.ndarray, number_of_classes=11):
        y_data_preprocessed = []

        for y in y_data:
            y_data_row = np.zeros(number_of_classes)
            y_data_row[y] = 1
            y_data_preprocessed.append(y_data_row)

        # y = 3 -> y = [0 0 0 1 0 0 0 0 0 0 0 0]
        return np.array(y_data_preprocessed)

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred

    def evaluate_model(self,
                       model_name: str,
                       include_regression_metrics=True,
                       include_classification_metrics=False,
                       largest_classification_value=10,
                       plot_classification_data=False,
                       log_evaluation=True):
        print("Evaluating model ...")
        evaluation = []

        self.model.evaluate(self.x_test, self.y_test, verbose=2)
        self.y_pred = self.model.predict(self.x_test)

        # Regression Metrics
        if include_regression_metrics:
            # R2 score
            r2 = r2_score(self.y_test, self.y_pred)
            # MSE
            mse = mean_squared_error(self.y_test, self.y_pred)

            evaluation.append(f'R2 score: {r2}\n')
            evaluation.append(f'MSE: {mse}\n')

        # Classification Metrics
        if include_classification_metrics:
            # In case of classification, get max index as class
            y_pred_classification = list(map(self._get_max_index, self.y_pred))
            y_test_classification = list(map(self._get_max_index, self.y_test))
            # Rounding to maximum classification value
            y_pred_classification = [round(pred) if pred <= largest_classification_value else largest_classification_value for pred in y_pred_classification]
            # Classification metrics
            classification_report = skmetrics.classification_report(y_test_classification, y_pred_classification)
            classification_confusion_matrix = confusion_matrix(y_test_classification, y_pred_classification)

            evaluation.append(f'Report: \n{classification_report}\n')
            evaluation.append(f'Confusion matrix: \n{classification_confusion_matrix}\n')

            if plot_classification_data:
                plot.scatter(y_test_classification, y_pred_classification, color='red')
                plot.xlabel('True Value')
                plot.ylabel('Predicted Value')
                plot.show()

        # Log evaluation
        if log_evaluation:
            file_object = open(f'evaluation_logs/{model_name}.txt', 'a')
            file_object.write(f"\n{model_name}\n {evaluation} \n=====================================\n")
            file_object.close()

        print(f"Evaluation:\n{evaluation}")

    @staticmethod
    def _get_max_index(np_array: np.ndarray):
        l = np_array.tolist()
        return l.index(max(l))

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import metrics
from data.data_loader import DataLoader
from sklearn.svm import SVR
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix


class SVRModel(object):
    def __init__(self,
                 data_loader: DataLoader,
                 kernel='rbf',
                 C=100,
                 epsilon=- 0.1):
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.data_loader = data_loader
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)

    def load_training_data(self,
                           test_size=0.2,
                           random_state=42  # Use same state to compare models
                           ):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data_loader.all_thicknesses_data,
                                                                                self.data_loader.all_thicknesses,
                                                                                test_size=test_size,
                                                                                random_state=random_state)

    def train_model(self,
                    output_file_name: str,
                    save_file: bool = False):
        self.model.fit(self.x_train, self.y_train)
        print('Done training...')
        if save_file:
            self.save_model(output_file_name)

    def save_model(self, output_file_name: str):
        pickle.dump(self.model, open(f"{output_file_name}.sav", 'wb'))
        print(f'Saved model in {output_file_name}')

    def load_model(self, file_name: str):
        self.model = pickle.load(open(f'{file_name}.sav', 'rb'))
        print(f'Loaded model from {file_name}')

    def evaluate_model(self,
                       largest_thickness=10):
        y_pred = self.model.predict(self.x_test)
        print(f'R2 score: {r2_score(self.y_test, y_pred)}')

        y_pred_classification = [round(pred) if pred <= largest_thickness else largest_thickness for pred in y_pred]
        # Classification metrics
        classification_report = metrics.classification_report(self.y_test, y_pred_classification)
        classification_confusion_matrix = confusion_matrix(self.y_test, y_pred_classification)
        print(f'Report: {classification_report}')
        print(f'Confusion matrix: {classification_confusion_matrix}')


if __name__ == "__main__":
    pass

from data.data_loader import DataLoader
from model.base_model import Model
import tensorflow as tf
from tensorflow import keras
import datetime
from typing import List
import os
from sklearn.metrics import r2_score
from metrics.pixel_wise_iou import pixel_wise_iou
from metrics.pixel_wise_recall import pixel_wise_recall
from metrics.pixel_wise_precision import pixel_wise_precision
from metrics.pixel_wise_dice import pixel_wise_dice
from metrics.pixel_wise_accuracy import pixel_wise_accuracy
import numpy as np
from visualization.environment_oil_thickness_distribution import visualize_environment


class NNModel(Model):
    def __init__(self, data_loader: DataLoader, network_layers: List, **kwargs):
        super().__init__(data_loader, **kwargs)
        self.network_layers = network_layers
        self.model = self.create_neural_network(network_layers)

    def train_model(self,
                    output_file_name: str,
                    output_file_extension: str = "h5",
                    save_file: bool = False,
                    model_path=None,
                    batch_size=20,
                    epochs=10):
        print('Started training model ...')
        # Tensorboard
        logdir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        print(self.x_train.shape)
        print(self.y_train.shape)
        # Train model
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                       callbacks=[tensorboard_callback])
        print('Done training...')

        # Saving file
        if save_file:
            generated_model_path = 'generated_models' if not model_path else f'generated_models/{model_path}'
            if not os.path.exists(generated_model_path):
                os.makedirs(generated_model_path)
            self.save_model(f"{output_file_name}", extension=output_file_extension)

    def train_model_kfold_cross_validation(self,
                                           output_file_name: str,
                                           batch_size=20,
                                           epochs=10):

        loss_per_fold = []
        mae_per_fold = []
        r2_per_fold = []
        fold_no = 1
        self.print_summary = False
        for train, test in self.kfold_indices:
            self.model = self.create_neural_network(self.network_layers)
            history = self.model.fit(self.data_loader.all_data_x[train], self.data_loader.all_data_y[train],
                                     batch_size=batch_size,
                                     epochs=epochs)

            # Generate generalization metrics
            scores = self.model.evaluate(self.data_loader.all_data_x[test], self.data_loader.all_data_y[test],
                                         verbose=0)
            y_pred = self.model.predict(self.data_loader.all_data_x[test])
            r2 = r2_score(self.data_loader.all_data_y[test], y_pred)

            loss_per_fold.append(scores[0])
            mae_per_fold.append(scores[1])
            r2_per_fold.append(r2)

            print(
                f'> Score for fold {fold_no}: {self.model.metrics_names[0]} of {scores[0]}; {self.model.metrics_names[1]} of {scores[1]}')
            print(
                f'> R2: {r2}; Loss per fold: {loss_per_fold}')

            # Increase fold number
            fold_no = fold_no + 1
        # == Provide average scores ==
        metrics = {
            "mse": np.mean(loss_per_fold),
            "std_dev": np.std(loss_per_fold),
            "r2": np.mean(r2_per_fold),
            "mae": np.mean(mae_per_fold),
            "loss_per_fold": loss_per_fold
        }

        print('------------------------------------------------------------------------')
        print('Score per fold')
        for i in range(0, len(loss_per_fold)):
            print('------------------------------------------------------------------------')
            print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]}')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'{metrics}')
        print('------------------------------------------------------------------------')
        return metrics

    def save_model(self, output_file_name: str, extension: str = "h5"):
        self.model.save(f"{output_file_name}.{extension}")
        print(f'Saved model in {output_file_name}.{extension}')

    def load_model(self, file_name: str, extension: str = "h5"):
        self.model = tf.keras.models.load_model(f'{file_name}.{extension}')
        print(f'Loaded model from {file_name}.{extension}')

    def evaluation_signature(self) -> str:
        return f"Neural Network: \n{self.network_layers}\noptimizer={self.optimizer}\nlearning rate={self.learning_rate}\nloss={self.loss}\nmetrics={self.metrics}"

    def parse_args(self, **kwargs):
        self.optimizer = kwargs.get('optimizer', "Adam")
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.loss = kwargs.get('loss', 'categorical_crossentropy')
        self.metrics = kwargs.get('metrics', ["accuracy"])
        self.print_summary = kwargs.get('print_summary', True)

    def create_neural_network(self, neural_network: List):
        print(f"Created network {neural_network}")
        # Generating nn layers
        nn_layers = []
        for layer in neural_network:
            if layer[0] == "Input":
                nn_layers.append(tf.keras.layers.Input(shape=(layer[1],)))
            elif layer[0] == "Dense":
                nn_layers.append(tf.keras.layers.Dense(layer[1], activation=layer[2]))
        model = tf.keras.models.Sequential(nn_layers)

        # Optimizer
        if self.optimizer == "Adam":
            opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            opt = self.optimizer

        model.compile(optimizer=opt,
                      loss=self.loss,
                      metrics=self.metrics)
        if self.print_summary:
            model.summary()
        return model

    def evaluate_metrics(self, x_all, y_all, folder):

        iou = []
        recall = []
        precision = []
        accuracy = []
        dice = []
        counter = 0
        for j in range(len(x_all)):
            x = x_all[j]
            ye = y_all[j]

            # def pred(x):
            #     a = x.reshape(1, -1)
            #     print("PREDDING")
            #     return self.predict(a)
            #
            # y_pred = np.apply_along_axis(pred, 2, x)
            x = np.array(x)
            x = np.reshape(x, (-1, 9))
            y_pred = (self.predict(x))
            y_pred = np.reshape(y_pred, (100, 100, -1))
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
                                  output_file_name=f"{folder}/pred_estimation_{j}", file_type='png')
            print(f"Done image {j}")

        print(f"Average iou coefficient: {sum(iou) / len(iou)}")
        print(f"Average dice coefficient: {sum(dice) / len(dice)}")
        print(f"Average precision coefficient: {sum(precision) / len(precision)}")
        print(f"Average recall coefficient: {sum(recall) / len(recall)}")
        print(f"Average accuracy coefficient: {sum(accuracy) / len(accuracy)}")

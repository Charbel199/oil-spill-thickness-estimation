from data.data_loader import DataLoader
from model.model import Model
import tensorflow as tf
from tensorflow import keras
import datetime
from typing import List


class NNModel(Model):
    def __init__(self, data_loader: DataLoader, network_layers: List, **kwargs):
        super().__init__(data_loader)
        self.model = self.create_neural_network(network_layers, **kwargs)

    def train_model(self,
                    output_file_name: str,
                    output_file_extension: str = "h5",
                    save_file: bool = False,
                    epochs=10):
        print('Started training model ...')
        # Tensorboard
        logdir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        # Train model
        self.model.fit(self.x_train, self.y_train, epochs=epochs, callbacks=[tensorboard_callback])
        print('Done training...')

        # Saving file
        if save_file:
            self.save_model(f"generated_models/{output_file_name}", extension=output_file_extension)

    def save_model(self, output_file_name: str, extension: str = "h5"):
        self.model.save(f"{output_file_name}.{extension}")
        print(f'Saved model in {output_file_name}.{extension}')

    def load_model(self, file_name: str, extension: str = "h5"):
        self.model = tf.keras.models.load_model(f'{file_name}.{extension}')
        print(f'Loaded model from {file_name}.{extension}')

    @staticmethod
    def create_neural_network(neural_network: List,
                              **kwargs):

        optimizer = kwargs.get('optimizer', "Adam")
        learning_rate = kwargs.get('learning_rate', 0.001)
        loss = kwargs.get('loss', 'categorical_crossentropy')
        metrics = kwargs.get('metrics', ["accuracy"])
        print_summary = kwargs.get('print_summary', True)

        # Generating nn layers
        nn_layers = []
        for layer in neural_network:
            if layer[0] == "Input":
                nn_layers.append(tf.keras.layers.Dense(layer[1], input_shape=(layer[1],)))
            elif layer[0] == "Dense":
                nn_layers.append(tf.keras.layers.Dense(layer[1], activation=layer[2]))
        model = tf.keras.models.Sequential(nn_layers)

        # Optimizer
        if optimizer == "Adam":
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            opt = optimizer

        model.compile(optimizer=opt,
                      loss=loss,
                      metrics=metrics)
        if print_summary:
            model.summary()
        return model

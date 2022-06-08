from data.data_loader import DataLoader
from model.model import Model
import tensorflow as tf
from tensorflow import keras
import datetime
from typing import List
import os


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
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, callbacks=[tensorboard_callback])
        print('Done training...')

        # Saving file
        if save_file:
            generated_model_path = 'generated_models' if not model_path else f'generated_models/{model_path}'
            if not os.path.exists(generated_model_path):
                os.makedirs(generated_model_path)
            self.save_model(f"{output_file_name}", extension=output_file_extension)

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

        # Generating nn layers
        nn_layers = []
        for layer in neural_network:
            if layer[0] == "Input":
                nn_layers.append(tf.keras.layers.Dense(layer[1], input_shape=(layer[1],)))
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

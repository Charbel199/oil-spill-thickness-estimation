from data.data_loader import DataLoader
from model.model import Model
from sklearn.svm import SVR
import pickle


class SVRModel(Model):
    def __init__(self, data_loader: DataLoader, **kwargs):
        super().__init__(data_loader)

        self.model = self.create_svr(**kwargs)

    def train_model(self,
                    output_file_name: str,
                    output_file_extension: str = "sav",
                    save_file: bool = False,
                    epochs=10):
        # Train model
        self.model.fit(self.x_train, self.y_train)
        print('Done training...')

        # Saving file
        if save_file:
            self.save_model(f"generated_models/{output_file_name}", extension=output_file_extension)

    def save_model(self, output_file_name: str, extension: str = "sav"):
        pickle.dump(self.model, open(f"{output_file_name}.{extension}", 'wb'))
        print(f'Saved model in {output_file_name}.{extension}')

    def load_model(self, file_name: str, extension: str = "sav"):
        self.model = pickle.load(open(f'{file_name}.{extension}', 'rb'))
        print(f'Loaded model from {file_name}.{extension}')

    @staticmethod
    def create_svr(**kwargs):
        kernel = kwargs.get('kernel', "rbf")
        C = kwargs.get('C', 100)
        epsilon = kwargs.get('epsilon', 0.1)

        model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        return model

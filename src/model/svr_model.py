from data.data_loader import DataLoader
from model.base_model import Model
from sklearn.svm import SVR
import pickle


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
            self.save_model(f"generated_models/{output_file_name}", extension=output_file_extension)

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

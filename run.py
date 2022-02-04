import sys

from data.data_loader import DataLoader
from model.svr_model import SVRModel

loader = DataLoader()
loader.load_data(synthetic_data_file_name="final",
                 smallest_value=1.9,
                 largest_value=3.3,
                 step_size=0.1)
print(loader.all_data_x.shape)
print(loader.all_data_y.shape)

new_model = False
model = SVRModel(data_loader=loader)
model.load_training_data()
if new_model:
    model.train_model(output_file_name="5mmthickness", save_file=True)
else:
    model.load_model('5mmthickness')

model.evaluate_model(largest_value=sys.maxsize)

y_pred = model.predict(model.x_test[0:8])
print(model.y_test[0:8])
print(y_pred)

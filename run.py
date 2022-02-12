import sys

from data.data_loader import DataLoader
from model.svr_model import SVRModel

file_name ='5mm-50observations-4freqs-variance0.02-'
model_name = '5mm-4freqs-variance0.001-2000'
new_model = False

loader = DataLoader()
loader.load_data(synthetic_data_file_name=f"generated_data/{file_name}",
                 smallest_value=1.9,
                 largest_value=3.3,
                 step_size=0.1,
                 max_number_of_rows=2000)


# file_name ='fromThicknessLowVariance'
# model_name = 'variance0.01-4000'
# new_model = True
#
# loader = DataLoader()
# loader.load_data(synthetic_data_file_name=file_name,
#                  smallest_value=0,
#                  largest_value=10,
#                  step_size=1,
#                  max_number_of_rows=4000)

print(loader.all_data_x.shape)
print(loader.all_data_y.shape)


model = SVRModel(data_loader=loader, epsilon=0.1, C=10)
model.load_training_data()
if new_model:
    model.train_model(output_file_name=model_name, save_file=True)
else:
    model.load_model(f"generated_models/{model_name}")

model.evaluate_model(file_name=model_name,largest_value=sys.maxsize, log_eval = True)

size_to_view = 40
y_pred = model.predict(model.x_test[0:size_to_view])
for i in range(len(y_pred)):
    print(f"{model.y_test[i]} --> {y_pred[i]}")


from data.data_loader import DataLoader
from model.nn_model import NNModel
import numpy as np
import matplotlib.pyplot as plt

# Parameters
# ==================================================================================================================
FILE_NAME = 'oldfreqsrealdatathickness7mm'
MODEL_NAME = 'WITHOUT0-nn-v3-2outputs-thickness-9freqs-variance0.001-20000'
NEW_MODEL = False
REAL_WORLD_THICKNESS = 7
THICKNESS_INDEX = 1
PERMITTIVITY_INDEX = 0
network_layers = [
    ["Input", 9],
    ["Dense", 12, "relu"],
    ["Dense", 16, "relu"],
    ["Dense", 12, "relu"],
    ["Dense", 2, "linear"]
]
# ==================================================================================================================


loader = DataLoader()
loader.load_data_from_file(
    file_name=f"generated_data/{FILE_NAME}",
    file_format="{}-{}.txt",
    possible_output_values=[(7, 7, 1)],
    max_number_of_rows=4000)

model = NNModel(data_loader=loader, network_layers=network_layers, loss='mean_squared_error', print_summary=True)
model.load_model_data(test_size=0.9, is_classification_problem=False, normalize_output=False)
if NEW_MODEL:
    model.train_model(output_file_name=MODEL_NAME, save_file=True, epochs=15)
else:
    model.load_model(f"generated_models/{MODEL_NAME}")

save_figs = True


# Generate predicted thickness and permittivity for each number of observation
number_of_measurements = 15
starting_value = 320

x_values = model.x_test[starting_value:number_of_measurements + starting_value]
measurements = list(range(1, number_of_measurements + 1, 1))
thicknesses = []
permittivities = []
for i in measurements:
    narray = np.array(x_values[0:i])
    mean = np.mean(narray, axis=0)
    prediction = model.predict(np.array(mean).reshape((-1, 9)))
    thicknesses.append(prediction[0][THICKNESS_INDEX])
    permittivities.append(prediction[0][PERMITTIVITY_INDEX])

# Generate multiple observation plot
fig, ax = plt.subplots(figsize=(10, 5))
ax2 = ax.twinx()
plt.grid()
plt.xticks(np.arange(0, 16, 1))
ax.plot(measurements, thicknesses, 'g-x', label="Thickness")
ax2.plot(measurements, permittivities, 'b-x', label="Permittivity")
ax.set_xlabel("Number of observations, N")
ax.set_ylabel("Estimated thickness (mm)")
ax2.set_ylabel("Estimated permittivity")
output_file_name = "ExperimentalThicknesstest"
file_type = "svg"
fig.legend(loc="upper right")
plt.savefig(f'{output_file_name}.{file_type}', format=file_type)
plt.show()

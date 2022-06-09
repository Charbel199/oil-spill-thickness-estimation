import pandas as pd

from data.data_loader import DataLoader
from model.nn_model import NNModel
import itertools
import os.path

# Parameters
# ==================================================================================================================
DATA_FILE_NAME = 'thickness-4freqs-variance0.001'
NEW_MODEL = False
NUMBER_OF_HIDDEN_LAYERS = 2
MINIMUM_NUMBER_OF_NEURONS_PER_LAYER = 2
MAXIMUM_NUMBER_OF_NEURONS_PER_LAYER = 15
INPUT_FEATURES = 4
OUTPUT_FEATURES = 1
RESULTS_FILE_PATH = 'assets/comparing_nn_topologies.csv'
base_network_layers = [
    ["Input", INPUT_FEATURES],
    ["Dense", OUTPUT_FEATURES, "linear"]
]
# ==================================================================================================================

# Data
loader = DataLoader()
max_number_of_rows = 10000
loader.load_data_from_file(
    file_name=f"assets/generated_data/{DATA_FILE_NAME}",
    file_format="{}-{}.txt",
    possible_output_values=[(1, 10, 1)],
    max_number_of_rows=max_number_of_rows)

# Generate all possible topologies
possible_number_of_neurons_per_layer = range(MINIMUM_NUMBER_OF_NEURONS_PER_LAYER,
                                             MAXIMUM_NUMBER_OF_NEURONS_PER_LAYER + 1)
possible_hidden_layers_architectures = list(
    itertools.product(*[ele for ele in [possible_number_of_neurons_per_layer] for _ in range(NUMBER_OF_HIDDEN_LAYERS)]))

if os.path.exists(RESULTS_FILE_PATH):
    results = pd.read_csv(RESULTS_FILE_PATH)
else:
    results = pd.DataFrame(columns=['layers', 'mse', 'std_dev', "r2", "mae", "loss_per_fold"])

for hidden_layers_architecture in possible_hidden_layers_architectures:
    network_layers = base_network_layers[1:].copy()

    network_layers_text = f'{INPUT_FEATURES}-'
    for layer_size in hidden_layers_architecture:
        network_layers.insert(-1, ["Dense", layer_size, "relu"])
        network_layers_text += f"{layer_size}-"
    network_layers_text += f"{OUTPUT_FEATURES}"

    network_layers.insert(0, base_network_layers[0])

    if network_layers_text in set(results['layers']):
        print(f"Architecture {network_layers_text} already evaluated")
        continue

    print(f"NN architecture: {network_layers}")
    print(f"Training model with layers: {network_layers_text}")
    model_name = f'assets/comparing_nn_topologies/nn-{network_layers_text}-4freqs-variance0.001-{max_number_of_rows}'
    # Training and evaluation
    model = NNModel(data_loader=loader, network_layers=network_layers, loss='mean_squared_error',
                    print_summary=True, metrics="mean_absolute_error")
    model.load_model_data(test_size=0.2, is_classification_problem=False, normalize_output=False, num_of_folds=10)

    # if not model.check_if_model_exists(f"/{model_name}.h5"):
    metrics = model.train_model_kfold_cross_validation(
        batch_size=64,
        output_file_name=f"assets/generated_models/comparison/{model_name}",
        epochs=3)
    metrics["layers"] = network_layers_text

    results = results.append(metrics, ignore_index=True)
    results.to_csv(f'{RESULTS_FILE_PATH}')
    print(results)

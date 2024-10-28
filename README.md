If you are using this code in your own project, please cite at least one of our relevant papers:

```
@article{hammoud2024date,
  title={DATE on the Edge: On-Site Oil Spill Detection and Thickness Estimation using Drone-based Radar Backscattering},
  author={Hammoud, Bilal and Maroun, Charbel Bou and Wehn, Norbert},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```
```
@inproceedings{hammoud2023oil,
  title={Oil Spill Detection in Calm Ocean Conditions: A U-Net Model Novel Solution},
  author={Hammoud, Bilal and Maroun, Charbel Bou and Moursi, Mohamed and Wehn, Norbert},
  booktitle={IGARSS 2023-2023 IEEE International Geoscience and Remote Sensing Symposium},
  pages={4658--4661},
  year={2023},
  organization={IEEE}
}
```
```
@inproceedings{maroun2022machine,
  title={Machine Learning Using Support Vector Regression in Radar Remote Sensing for Oil-Spill Thickness Estimation},
  author={Maroun, Charbel Bou and Daou, Georges and Hammoud, Bassel and Hammoud, Bilal},
  booktitle={2021 18th European Radar Conference (EuRAD)},
  pages={221--224},
  year={2022},
  organization={IEEE}
}
```


# Oil Spill Thickness Estimation

This repository provides a proof of concept for estimating oil spill thickness using machine learning models trained on radar data from a UAV. The models detect the presence of oil and estimate its thickness, ranging from 1 to 10 mm, which is essential for assessing oil spill scenarios accurately.

## Machine learning models

We tested 3 different approaches for oil spill detection and thickness estimation:<br>
&emsp; **SVR**<br>
&emsp; **Neural Networks**<br>
&emsp; **[U-Net Model](src/model/unet_model.py)**<br>

**Key Insights:**

- SVR and ANN models are effective for estimating oil thickness in scenarios where the thickness distribution is relatively uniform. These point-based models have low computational complexity, making them suitable for onboard monitoring.
- The ANN model provides additional estimates for oil thickness and oil permittivity, enhancing detection accuracy.
- U-Net Model significantly outperforms SVR and ANN by leveraging spatial correlations across the spill area, making it ideal for more complex spill distributions. However, it requires a complete scan of the environment before processing, making it more computationally intensive.

The following table summarizes key performance metrics for each model:

| Model          | IoU          | Dice                   | Precision     | Recall |
|----------------|--------------|------------------------|---------------|--------|
| SVR            | 0.29 | 0.43                   | 0.43  | 0.51   |
| ANN            | 0.36		| 0.49 	                 | 0.49  | 0.56   |
| U-Net          | 0.75			| 0.86 	                 | 0.85  | 0.87   |


## Data

Due to limited access to real-world oil spill data—especially for radars operating in wide-band frequencies—we created an [oil spill simulator](https://github.com/Charbel199/Oil-Spill-Simulation) that generates realistic oil spill distributions. The simulation incorporates essential real-world characteristics, providing data for training and validating our models.

<img src="docs/RealFormatted.svg" width="45%">
<img src="docs/SimulatedFormatted.svg" width="45%">

Our simulation pipeline also models radar reflectivities through Monte Carlo simulations, incorporating additive white Gaussian noise and roughness loss, adjusted for various radar frequencies. This process is facilitated by the **[data generator](src/data/data_generator/generate_reflectivities_from_thickness/data_generator.m)** module , which uses the **[relative dielectric constant](src/data/data_generator/generate_reflectivities_from_thickness/module4_2.m)**, **[noise](src/data/data_generator/generate_reflectivities_from_thickness/noise.m)**
and **[export to file](src/data/data_generator/generate_reflectivities_from_thickness/export_to_file.m)** modules to produce data for model training.

Finally, we use the generated reflectivity text files along with the simulated thickness matrices in [run_segmentation_dataset_generation](src/run/segmentation/run_segmentation_dataset_generation.py) to populate each environment with realistic reflectivities, producing a comprehensive dataset that represents oil spills of varying thickness in a radar context.


## Dataset

The complete dataset used in this project can be accessed [here](link).

## Run

To train the U-Net model, execute the launch_unet_training.sh script located in the src directory:

``` bash
chmod +x ./src/launch_unet_training.sh
./src/launch_unet_training.sh
```

This script initiates model training with default parameters. Adjustments can be made in the script to customize the training process.
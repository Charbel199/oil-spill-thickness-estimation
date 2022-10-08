If you are using this code in your own project, please cite at least one of our relevant papers:

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
```
@inproceedings{hammoud2022artificial,
  title={Artificial Neural Networks-Based Radar Remote Sensing to Estimate Geographical Information during Oil-Spills},
  author={Hammoud, Bilal and Maroun, Charbel Bou and Ney, Jonas and Wehn, Norbert},
  booktitle={2022 European Signal Processing Conference (EUSIPCO). Piscataway, NJ, USA: IEEE},
  year={2022}
}
```

# Oil Spill Thickness Estimation

A proof of concept for using machine learning with radar data from a UAV hovering over a body of water to detect the
presence of oil
and estimate its thickness. For proper assessment of an oil spill scenario, thickness measurements need to range from 1
to 10 mm.

## Machine learning models

We tested 4 different approaches for oil spill detection and thickness estimation:<br>
&emsp; **[SVR](src/model/svr_model.py)**<br>
&emsp; **[Neural Networks](src/model/nn_model.py)**<br>
&emsp; **[U-Net Model](src/model/unet_model.py)**<br>
&emsp; **[Cascaded U-Net Model](src/model/unet_model_cascaded.py)**<br>

In summary:

- The SVR and ANN models are point-based models, they perform well when the oil thickness is uniform over the
  environment.
- We used the ANN model to estimate the oil thickness and the oil permittivity.
- The U-Net models perform way better since they take into account the spatial correlation within the spill.
- SVR and ANN have lower complexity compared to U-Net models making onboard monitoring and processing of oil spill data
  feasible.
- For the U-Net based approach, the environment has to be fully scanned and then processed.

Here are some metrics for the 4 machine learning models:

| Model          | IoU          | Dice                   | Precision     | Recall |
|----------------|--------------|------------------------|---------------|--------|
| SVR            | 0.29 | 0.43                   | 0.43  | 0.51   |
| ANN            | 0.36		| 0.49 	                 | 0.49  | 0.56   |
| U-Net          |0.75			| 0.86 	                 | 0.85  | 0.87   |
| Cascaded U-Net | 0.82			|   0.89 	|  0.90 	|  0.90  |

Regarding the cascaded U-Net model, here is a diagram showcasing its architecture:

![cascaded](docs/Cascaded_UNET_Model_Archi.png)


## Data

Due to the scarcity of oil spill data, especially for radars operating in wide-band
ranges, we developed an [oil spill simulator](https://github.com/Charbel199/Oil-Spill-Simulation) to model the spill. The developed model reflects
essential properties of the real world and provides us with realistic oil spill distributions (More info in the github repo).

<img src="docs/RealFormatted.svg" width="45%">
<img src="docs/SimulatedFormatted.svg" width="45%">

The second part takes care of populating the simulated environment with radar
reflectivities using a Monte-Carlo simulation by introducing the additive white Gaussian
noise and roughness loss to the computed reflectivities based on input frequencies.
To generate the needed data, the **[data generator](src/data/data_generator/generate_reflectivities_from_thickness/data_generator.m)** module is used.
It utilizes the **[relative dielectric constant](src/data/data_generator/generate_reflectivities_from_thickness/module4_2.m)** module along with the **[noise](src/data/data_generator/generate_reflectivities_from_thickness/noise.m)**
and the **[export to file](src/data/data_generator/generate_reflectivities_from_thickness/export_to_file.m)** modules.



## Run

All .py files used to train and evaluate the models are stored in the [run](src/run) directory.
The [segmentation](src/run/segmentation) directory contains all run files related to the segmentation models.
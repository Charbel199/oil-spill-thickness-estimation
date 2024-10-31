If you are using this code or the linked dataset in your own project, please cite the following paper:

```
@article{hammoud2024date,
  title={DATE on the Edge: On-Site Oil Spill Detection and Thickness Estimation using Drone-based Radar Backscattering},
  author={Hammoud, Bilal and Maroun, Charbel Bou and Wehn, Norbert},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2024},
  publisher={IEEE}
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


## Dataset

The complete dataset used in this project can be accessed [here](https://drive.google.com/file/d/1P35IgPGuhHE5zpNwdiYnA2W2Su4BN0Nz/view?usp=sharing).

## Run

To train the U-Net model, execute the launch_unet_training.sh script located in the src directory:

``` bash
chmod +x ./src/launch_unet_training.sh
./src/launch_unet_training.sh
```

This script initiates model training with default parameters. Adjustments can be made in the script to customize the training process.
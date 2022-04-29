# Oil Spill Thickness Estimation
A proof of concept for using machine learning with radar data from a UAV hovering over a body of water to
estimate the thickness of an oil spill.

For proper assessment of an oil spill scenario, thickness measurements need to range from 1 to 10 mm.

## Machine learning models
For now, two main machine learning models are being tested:<br>
&emsp; **[SVR](model/svr_model.py)**<br>
&emsp; **[Neural Networks](model/nn_model.py)**<br>

SVRs are quite simple but very inefficient when it comes to many input features and a large amount of training data.
<br>
Therefore, we worked on studying the most performant neural network architecture and corresponding hyperparameters.
<br>
**NOTE:** We should keep in mind the complexity of the model since the end goal would be to load it on an FPGA that 
would be installed on a drone.

## Data
We are generating training data from MATLAB simulations.
To generate the needed data, the **[data generator](data/data_generator/generate_reflectivities_from_thickness/data_generator.m)** module is used.
It utilizes the **[relative dielectric constant](data/data_generator/generate_reflectivities_from_thickness/module4_2.m)** module along with the **[noise](data/data_generator/generate_reflectivities_from_thickness/noise.m)** 
and the **[export to file](data/data_generator/generate_reflectivities_from_thickness/export_to_file.m)** modules.




import tensorflow as tf

extension = 'h5'
file_name = 'ann_highvariance_windspeed8_with_0'
model = tf.keras.models.load_model(f'{file_name}.{extension}')
print(f'Loaded model from {file_name}.{extension}')

print((model.model.get_layer("dense").weights))
print((model.model.get_layer("dense_1").weights))
print((model.model.get_layer("dense_2").weights))
print((model.model.get_layer("dense_3").weights))

exit()

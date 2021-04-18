import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential()
model.add(layers.ConvLSTM2D(32, (3, 3), activation='relu', batch_size=32, input_shape=(1, 28, 28, 1), padding='same', strides=2,
                            stateful=True))
model.add(layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same'))
model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
model.compile(optimizer='adam', loss=['mse'])
model.save('teste.tf')
model = tf.keras.models.load_model('teste.tf')
print(model)
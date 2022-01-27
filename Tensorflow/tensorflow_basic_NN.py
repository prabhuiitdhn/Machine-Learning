import tensorflow as tf
import numpy as np
#importing tensorflow

#difinfing the model layer
model = tf.keras.Sequential(
    [tf.keras.layers.Dense(units=1, input_shape=[1])]
)
# input_shape: 1; one neuron
#creating a sequential layer: It is used for writing the sequential layer
#Dense layer is used for creating fully connected layer
model.compile(
    optimizer ='sgd',
    loss = 'mean_squared_error'
)
# model.compile is for getting the optimiser, loss and all the neccessary information

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype = float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype= float)
# this is the data explanation, the relationship needs to find using the NN

model.summary()
model.fit(xs, ys, epochs=500)
#model.fit is for training the model

#predicting the value using train model of 500 epochs on the data xs, ys

print(model.predict([10.0]))








#fashion-MNIST data
'''
steps:
1. load the data
2. define the model architecture
pass the data
3. train the model using .fit
'''


import tensorflow as tf


fashion_mnist = tf.keras.datasets.fashion_mnist
#defining the data; Loading the data
(train_images, train_labels), (test_images, test_labels) = \
    fashion_mnist.load_data()

#loading the data and separatting train_image, train_labels as well as test_images, test_labels

#Defining the architecture
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    #expecting the data to be size of 28 * 28 and it flatten the 28 * 28 by simple linear array of 784
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

#prepare the model to train
model.compile(
    optimizer=tf.train.AdamOptimizer,
    loss='sparse_categorical_crossentropy'
)

#training the model using train_images, and train_labels, with epochs5
model.fit(train_images,
          train_labels,
          epochs=5)
#using the trained model; Try to evaluates in testing data

model.evaluate(test_images, test_labels)








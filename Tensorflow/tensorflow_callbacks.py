#using the callbacks we can stop the traing and much more things we can
import tensorflow as tf

#callback is being written so that If model has to stop the training, stops when loss comes < 0.4
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss')<0.4):
            print("Loss is low and cancelling the training")
            self.model.stop_training=True

#load the fashion_mnist dataset

callbacks = myCallback()

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# to train the data in better way we can normalise the data
train_images = train_images/255.0
test_images = test_images/255.0
#define the architecture

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

#preparing the model to train

model.compile(
    optimizer='sgd',
    loss = 'sparse_categorical_crossentropy'
)

model.fit(train_images, train_labels, epochs=5, callbacks=[callbacks])


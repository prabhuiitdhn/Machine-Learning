# using cnn concepts in tensorflow
'''
steps:
1. Load the data
2. define the model architecture
3. prepare the model to train
4. Train the model
'''
import tensorflow as tf

# loading the data
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_image, train_label), (test_images, test_labels) = fashion_mnist.load_data()

# define the model architecture

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64,
                           kernel_size=(3, 3),
                           strides=(1, 1),
                           padding='valid',
                           data_format=None,
                           dilation_rate=(1, 1),
                           activation='relu',
                           use_bias=True,
                           kernel_initializer='glorot_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=None,
                           bias_regularizer=None,
                           activity_regularizer=None,
                           kernel_constraint=None,
                           bias_constraint=None,
                           input_shape=(28, 28, 1)
                           ),
    # total number of 64 filter of size (3,3) will be created with the stride of (1,1)
    # to convert the grey scale image, (28, 28, 1) will be colored image
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                 strides=None,
                                 padding='valid',
                                 data_format=None),

    # take the maximum value of 2 X 2 pixels of images;  and compress the original(28 X 28) by (14 X 14)
    #
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# model summary helps to give the model summary


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=None,
    loss_weights=None,
    sample_weight_mode=None,
    weighted_metrics=None,
    target_tensors=None,
    distribute=None)

# gives the model summary
model.summary()

# It starts the training the model
model.fit(x=train_image,
          y=train_label,
          batch_size=None,
          epochs=5,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False)

# after model trained, evaluate the model on test_data

train_loss = model.evaluate(x=test_images,
                            y=test_labels,
                            batch_size=None,
                            verbose=1,
                            sample_weight=None,
                            steps=None,
                            max_queue_size=10,
                            workers=1,
                            use_multiprocessing=False)

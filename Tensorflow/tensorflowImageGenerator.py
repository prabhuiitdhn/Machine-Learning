'''
Imagegenerator: tensorflow api which generated the automatically label for feeded image based on the directory
If the image will be in this directory; tensorflow api automatically generates the labels to the images as per the image directory available
images:
    Train:
        class1
        class2
    validation:
        class1
        class2
'''

import tensorflow as tf


train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
               featurewise_center=False,
               samplewise_center=False,
               featurewise_std_normalization=False,
               samplewise_std_normalization=False,
               zca_whitening=False,
               zca_epsilon=1e-6,
               rotation_range=0,
               width_shift_range=0.,
               height_shift_range=0.,
               brightness_range=None,
               shear_range=0.,
               zoom_range=0.,
               channel_shift_range=0.,
               fill_mode='nearest',
               cval=0.,
               horizontal_flip=False,
               vertical_flip=False,
               rescale=1./255,
               preprocessing_function=None,
               data_format=None,
               validation_split=0.0)
#this Imagedatagenerator is preprocssing the image from the image directory which is being created as per tensorflow directory structure
#it automatically resizes the image if images is not sized.
train_generator = train_data_gen.flow_from_directory(
                            directory = train_director_where_data_is_being_saved,
                            target_size=(256, 256),
                            color_mode='rgb', #defin wether the color is colored image or greyscale image
                            classes=None, #define the classes how many classes you have
                            class_mode='binary', #please check 'categorical' is for multi-class
                            batch_size=32,
                            shuffle=True, #do I need t shuffle the image
                            seed=None, #Please ccheck
                            save_to_dir=None, #which directory needs to save the image
                            save_prefix='',
                            save_format='png', #what would be the format of creating the image
                            follow_links=False,
                            subset=None,
                            interpolation='nearest')



#defining the architectire

'''
Multiple layer of convolution which reflects higher complexity and sizes of image

'''







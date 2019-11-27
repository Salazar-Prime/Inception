# Project-IE590: Deep Learning for Machine Vision
# Keras Implementation of InceptionResNetV2
# Author: Varun Aggarwal


# ensures back compatibility
from tensorflow.keras import backend as K

# for reading and preprocessing data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# inceptionv4 model from keras with pretrained weights
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import tensorflow as tf
# inception_resnet_v2.InceptionResNetV2

DATASET_PATH  = './weed_data'
IMAGE_SIZE    = (300, 300)
NUM_CLASSES   = 4
BATCH_SIZE    = 8
NUM_EPOCHS    = 20
WEIGHTS_FINAL = 'model-inception_resnet_v2-final.h5'

# specify data augmentation parameters for training images
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.4,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)


valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/val',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

# show class indices
print('****************')
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
print('****************')

# config = tf.compat.v1.ConfigProto()
# config.inter_op_parallelism_threads = 6
# config.intra_op_parallelism_threads = 6
# tf.compat.v1.Session(config=config)


# inceptionv4 - model setup
model = InceptionResNetV2(include_top=False,
			weights='imagenet',
                        input_tensor=None,
                        input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))


# inceptionv3 - model setup
# model = InceptionV3(include_top=False,
#                         weights='imagenet',
#                         input_tensor=None,
#                         input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))

# adding final FC layer at the end of model
x = model.output
x = Flatten()(x)
x = Dropout(0.5)(x)
output_layer = Dense( NUM_CLASSES,
                      activation='softmax',
                      name='softmax')(x)

model = Model(inputs=model.input,
              outputs=output_layer)


# ensure all layers are trainable
for layer in model.layers:
    layer.trainable = True

# setting up optimizer for model
model.compile(optimizer=Adam(lr=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# human readble model summary
# print(model.summary())



# train the model
hist = model.fit_generator(train_batches,
                    steps_per_epoch = train_batches.samples // BATCH_SIZE,
                    validation_data = valid_batches,
                    validation_steps = valid_batches.samples // BATCH_SIZE,
                    epochs = NUM_EPOCHS)

# save trained weights
model.save(WEIGHTS_FINAL)


# Implementation as explained in: https://arxiv.org/pdf/1602.07261v1.pdf
# Code Source: Keras InceptionResetV2 (https://jkjung-avt.github.io/keras-inceptionresnetv2/)

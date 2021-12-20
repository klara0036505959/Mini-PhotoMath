import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import cv2
import numpy as np
from matplotlib import pyplot as plt
import itertools
import pandas as pd
import PIL
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

directory = 'custom_dataset_prepared'

batch_size = 2
img_height = 28
img_width = 28
image_size = (img_height, img_width)
epochs = 20

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    label_mode='categorical',
    validation_split=0.1,
    subset="training",
    seed=1337,
    color_mode="grayscale",
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    validation_split=0.1,
    subset="validation",
    label_mode='categorical',
    seed=1337,
    color_mode="grayscale",
    image_size=image_size,
    batch_size=batch_size,
)


class_names = train_ds.class_names
print(class_names)
train_ds = train_ds.shuffle(buffer_size=160, reshuffle_each_iteration=True)


train_ds = train_ds.map(lambda image,label:(tf.cast(image,'float32'),label))
val_ds = val_ds.map(lambda image,label:(tf.cast(image,'float32'),label))

rescale = Rescaling(scale=1.0/255)
train_ds = train_ds.map(lambda image,label:(rescale(image),label))
val_ds = val_ds.map(lambda image,label:(rescale(image),label))



from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, LeakyReLU
)

num_classes = len(class_names)
print(class_names)

model = Sequential([
    Conv2D(16, kernel_size=(3,3), input_shape=(img_width, img_height, 1)),
    BatchNormalization(),
    LeakyReLU(alpha = 0.3),
    Dropout(0.2),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, kernel_size=(3,3), input_shape=(img_width, img_height, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation=tf.nn.relu),
    Dropout(0.3),
    Dense(num_classes,activation=tf.nn.softmax)
])


adam1 = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=adam1, metrics=['accuracy'] )

model.summary()


#for i in range(epochs):
  #checkpoint = ModelCheckpoint('model1.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
  #callbacks_list = [checkpoint]

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=60)

model.save('my_model_new.h5')



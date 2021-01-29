import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pickle

from keras.optimizers import SGD
from keras.models import Sequential, save_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D

# Set image information
channels = 1
height = 1000
width = 754

model = Sequential()
# Add a Conv2D layer with 32 nodes to the model
model.add(Conv2D(32, (3, 3), input_shape=(1000, 754, 1)))
# Add the reLU activation function to the model
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',  # sparse_categorical_crossentropy
              # Adam(lr=.0001) SGD variation with learning rate
              optimizer='adam',
              metrics=['accuracy'])

# Image data generator to import iamges from data folder
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)


# Flowing images from folders sorting by labels, and generates batches of images
train_it = datagen.flow_from_directory(
    "data/train/", batch_size=16, target_size=(height, width), shuffle=True, class_mode='categorical')
val_it = datagen.flow_from_directory(
    "data/validate/", batch_size=16, target_size=(height, width), shuffle=True, class_mode='categorical')

history = model.fit(
    train_it,
    epochs=2,
    batch_size=16,
    validation_data=val_it,
    shuffle=True,
    steps_per_epoch=2000 // 16,
    validation_steps=800 // 16)


model.save("my_model", save_format='h5')

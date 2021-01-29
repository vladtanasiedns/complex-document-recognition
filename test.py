import tensorflow as tf
from tensorflow import keras
import numpy as np
from os import path
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import mobilenet_v2

mymodel = "my_model"

model = keras.models.load_model(mymodel)

channels = 1
height = 1000
width = 754

datagen = ImageDataGenerator()
test_it = datagen.flow_from_directory(
    "data/test", target_size=(height, width), class_mode='categorical')

test_imgs, test_labels = next(test_it)

predictions = model.predict_generator(test_imgs, steps=10, verbose=2)
print(predictions)

from keras.models import load_model
import cv2
import numpy as np
import keras
from keras.preprocessing import image

model = load_model('my_model')

# First try


def prepare(file):
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (1000, 754))
    return new_array.reshape(3, 1000, 754, 1)


# Second try
img = image.load_img(
    "/home/user1/Desktop/Office/image-process/test/0000113760.tif",
    target_size=(1000, 754))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)


prediction = model.predict(img, batch_size=1)
print(prediction)

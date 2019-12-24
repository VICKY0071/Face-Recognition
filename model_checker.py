import cv2
import tensorflow
import numpy as np

persons = ['sanjana', 'manju', 'yash', 'vikas']


def prepare(filename):
    size = 70
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    image = image/255
    new_image = cv2.resize(image, (size, size))
    return new_image.reshape(-1, size, size, 1)

model = tensorflow.keras.models.load_model('face_recognition_model.model')

prediction = model.predict([prepare('face_image.jpg')])
index = max(prediction[0])
array = [i for i in prediction[0]]

print(f'The photo is of : {persons[array.index(index)]}')
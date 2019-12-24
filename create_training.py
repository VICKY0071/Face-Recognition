import matplotlib.pyplot as plt
import cv2
import sklearn
import pickle
import os
import numpy as np

datasets = 'C:\\Users\\Vikas Thapliyal\\Desktop\\face recognition\\datasets'
persons = ['sanjana', 'manju', 'yash', 'vikas']
training_data = []
size = 70

for person in persons:
    class_num = persons.index(person)
    path = os.path.join(datasets, person)
    for img in os.listdir(path):
        image = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        new_image = cv2.resize(image, (size, size))
        training_data.append([new_image, class_num])

from random import shuffle

shuffle(training_data)

X, y= [], []

for feature,label in training_data:
    X.append(feature)
    y.append(label)

X = np.array(X).reshape(-1, size,size, 1)
y = np.array(y)

pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()
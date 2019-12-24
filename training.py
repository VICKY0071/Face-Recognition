import matplotlib.pyplot as plt
import pickle
import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Flatten, MaxPooling2D, Conv2D, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
import time

X = pickle.load(open('X.pickle', 'rb'))
y = pickle.load(open('y.pickle', 'rb'))

X = X/255

model = Sequential()

Name = f'face-recognintion{int(time.time())}'
tensorboard = TensorBoard(log_dir = f'logs\\{Name}')

model.add(Conv2D(32, (3, 3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Dense(16))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(X, y, validation_split = 0.2, batch_size = 1, epochs = 3 ,callbacks = [tensorboard])

model.save('face_recognition_model.model')
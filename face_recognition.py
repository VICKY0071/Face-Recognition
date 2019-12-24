import cv2
import numpy as np
from urllib.request import urlopen
import tensorflow
import gtts
from gtts import gTTS

video = cv2.VideoCapture(0)
video.set(3, 320)
video.set(4, 260)

from model_checker import prepare

model = tensorflow.keras.models.load_model('face_recognition_model.model')

url = 'http://192.168.43.15:8080/shot.jpg'

face_cascade = cv2.CascadeClassifier('C:\\Users\\Vikas Thapliyal\\Desktop\\face recognition\\cascades\\haarcascade_frontalface_alt2.xml')
side_cascade = cv2.CascadeClassifier('C:\\Users\\Vikas Thapliyal\\Desktop\\face recognition\\cascades\\haarcascade_profileface.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

persons = ['sanjana', 'manju', 'yash', 'vikas']

while True:
#    image = urlopen(url)
 #   image_np = np.array(bytearray(image.read()), dtype = np.uint8)
  #  frame = cv2.imdecode(image_np, -1)

    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.5, 3)

    side_faces = side_cascade.detectMultiScale(gray, 1.5, 3)

    for x, y, w, h in faces:
        cv2.imwrite('face_image.jpg', frame[y:y+h+30, x:x+w+30])
        prediction = model.predict([prepare('face_image.jpg')])
        index = max(prediction[0])
        array = [i for i in prediction[0]]
        text = persons[array.index(index)]
        cv2.rectangle(frame, (x-30, y-30), (x+w+30, y+h+30), (255, 0, 0), 3)
        if index > 0.700:
            speech = gTTS(text = text, lang = 'en', slow = True, lang_check = True)
            cv2.putText(frame, text, (x-30, y-30), font, 2, (0, 0, 255), 3)
            
    for x, y, w, h in side_faces:
        cv2.imwrite('face_image.jpg', frame[y:y+h+30, x:x+w+30])
        prediction = model.predict([prepare('face_image.jpg')])
        index = max(prediction[0])
        array = [i for i in prediction[0]]
        text = persons[array.index(index)]
        cv2.rectangle(frame, (x-30, y-30), (x+w+30, y+h+30), (255, 0, 0), 3)
        if index > 0.700:
            cv2.putText(frame, text, (x-30, y-30), font, 2, (0, 0, 255), 3)
        
    cv2.imshow('window', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()

import cv2

video = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

while True:

	ret, frame = video.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

 	## start the detection module functions...!!!

 	
#cv2.rectangle(frame, (x, y), ((x+w), (y+h)), (255, 0, 0), 5)


	cv2.imshow('window', frame)

	key = cv2.waitKey(1)
	if key == 27:
		break
video.release()

cv2.destroyAllWindows()
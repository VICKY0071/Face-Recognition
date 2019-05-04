import cv2

video = cv2.VideoCapture(0)

##adding cascades for better face detections
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
face_cascade1 = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt_tree.xml')
face_cascade2 = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
face_cascade3 = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')

##font decleration
font = cv2.FONT_HERSHEY_SIMPLEX

##adding cascade for cat face detections
cat_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalcatface_extended.xml')

##adding cascades for smile detections
##smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

##adding cascades for eye detections
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

while True:

	ret, frame = video.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.5, 5)
	faces1 = face_cascade1.detectMultiScale(gray, 1.5, 5)
	faces2 = face_cascade2.detectMultiScale(gray, 1.5, 5)
	faces3 = face_cascade3.detectMultiScale(gray, 1.5, 5)

	cat = cat_cascade.detectMultiScale(gray, 1.5, 3)

	for (cx, cy, cw, ch) in cat:
		cv2.rectangle(frame, (cx, cy), ((cx+cw), (cy+ch)), (0, 255, 0), 5)

	
	profiles = profile_cascade.detectMultiScale(gray, 1.5, 5)
	for (x,y,w,h) in faces1:
		cv2.rectangle(frame, (x, y), ((x+w), (y+h)), (255, 0, 0), 5)
	
	for (x,y,w,h) in faces2:
		cv2.rectangle(frame, (x, y), ((x+w), (y+h)), (255, 0, 0), 5)
	
	for (x,y,w,h) in faces3:
		cv2.rectangle(frame, (x, y), ((x+w), (y+h)), (255, 0, 0), 5)

		cv2.putText(frame,'FACE', (x, y), font, 1, (0, 0, 0,), 2, cv2.LINE_AA)

	for (fx,fy,fw,fh) in profiles:

		cv2.rectangle(frame, (fx, fy), ((fx+fw), (fy+fh)), (255, 0, 0), 5)

		cv2.putText(frame, "SIDE FACE", (fx, fy), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

	for (x,y,w,h) in faces:
		roi_gray = gray[x:x+w, y:y+h]
		roi_color = frame[x:x+w, y:y+h]
		##smile = smile_cascade.detectMultiScale(roi_gray, 1.5, 5)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

		
		##for (sx, sy, sw, sh) in smile:

			##cv2.rectangle(frame, (sx, sy),((sx+sw), (sy+sh)), (255, 0, 0), 5)

		eyes = eye_cascade.detectMultiScale(roi_gray, 1.5, 5)

		for (ex,ey,ew,eh) in eyes:

			cv2.rectangle(frame, (ex, ey), ((ex + ew), (ey + eh)), (0, 0, 255), 5)


	cv2.imshow('window', frame)

	key = cv2.waitKey(1)
	if key == ord('q'):
		break

video.release()

cv2.destroyAllWindows()
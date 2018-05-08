import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')

if face_cascade.empty():
	raise IOError('Unable to load the face cascade classifier xml file')


scaling_factor = 0.5

frame = cv2.imread('../images/prateek.jpg')
frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in face_rects:
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

#cv2.imshow('Face Detector', frame)
cv2.imwrite('./face_detector.jpg',frame)

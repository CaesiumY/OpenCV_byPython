import cv2
import numpy as np

mouth_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_mcs_mouth.xml')

if mouth_cascade.empty():
	raise IOError('Unable to load the mouth cascade classifier xml file')

frame = cv2.imread('../images/prateek.jpg')
ret = cv2.imread('../images/prateek.jpg')
ds_factor = 0.5

frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
for (x,y,w,h) in mouth_rects:
    y = int(y - 0.15*h)
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
    break

#cv2.imshow('Mouth Detector', frame)
cv2.imwrite('moute_detector.jpg',frame)


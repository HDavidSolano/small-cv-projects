import cv2
import numpy as np

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

cap = cv2.VideoCapture(0)

while True:                 # here you get within each frame within the video
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)

    ma_string = 'variance '+str(fm)
    cv2.putText(gray,ma_string, (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv2.imshow('gray',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows



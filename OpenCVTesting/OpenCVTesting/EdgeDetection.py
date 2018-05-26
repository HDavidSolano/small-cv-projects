import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    soblex = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize = 5)
    sobley = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize = 5)
    kernel = np.ones((1,1),np.uint8)
    edges = cv2.Canny(frame,70,70)
    cv2.imshow('original',frame)
    cv2.imshow('laplacian',laplacian)
    cv2.imshow('edges',edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
cap.release()
cv2.destroyAllWindows

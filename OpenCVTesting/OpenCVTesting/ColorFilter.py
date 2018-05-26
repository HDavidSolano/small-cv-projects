
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) #For range, since RGB is a color value, but it doe snot give you range, whereas the hue is color, value is houw much of that color, and V is intensity
    lower_red = np.array([140,150,0]) #We actualy setted up to show EVERYTHING
    upper_red = np.array([180,255,255]) #max is 180
    mask = cv2.inRange(hsv, lower_red, upper_red) #To filter out reds from image

    res = cv2.bitwise_and(frame,frame,mask = mask) # wehere there is something in the frame, and where the mask has 255 value

    kernel = np.ones((15,15), np.float32)/225 #Average of 15 by 15 pixels

    smoothed = cv2.filter2D(res,-1,kernel)

    median = cv2.medianBlur(res,15)

    blur = cv2.GaussianBlur(res,(15,15),0)

    bilateral = cv2.bilateralFilter(res,15,75,75)

    #cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)
    #cv2.imshow('smoothed',smoothed)
    cv2.imshow('res',res)
    #cv2.imshow('blur',blur)
    cv2.imshow('median',median)
    cv2.imshow('bilateral',bilateral)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
cap.release()
cv2.destroyAllWindows
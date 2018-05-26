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

    kernel = np.ones((5,5),np.uint8)

    erosion = cv2.erode(mask,kernel,iterations = 1)  #removes pixels that have neighboring voids

    dilation = cv2.dilate(mask,kernel,iterations = 1) #Opposite

    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel) #Removes false positives
    closing = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel) #Removes false negatives


    cv2.imshow('res',res)
    cv2.imshow('erosion',erosion)
    cv2.imshow('dilation',dilation)
    cv2.imshow('opening',opening)
    cv2.imshow('closing',closing)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
cap.release()
cv2.destroyAllWindows

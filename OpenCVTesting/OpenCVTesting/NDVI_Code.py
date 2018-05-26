import numpy as np
import cv2

img = cv2.imread('E:\\Flight_2_13_2015\\NDVI\\IMG_1817.JPG', cv2.IMREAD_COLOR)
b,g,r = cv2.split(img)
stretch = 1                       #this one stretchs the bands in case they are too dim
min_sat = 0                       #used to define the minimum value for NDVI
max_sat = 0.5                     #used to define the maximum value for NDVI
NIR = np.asfarray(r)
BLU = np.asfarray(b)
NDVI = ((NIR-BLU)/(NIR+BLU))
NDVI[NDVI < min_sat] = min_sat          #filters minimum values
NDVI[NDVI > max_sat] = max_sat          #filters maximum values
NDVI = NDVI - min_sat
NDVI = stretch*(255/(max_sat-min_sat))*NDVI     #transforms the rest of the information into a 255-based system (byte)
NDVI_i = NDVI.astype('uint8')   #Needed to actually show the picture
cv2.imwrite('resultgray.png',NDVI_i) #Saving the image after operations have been conducted
cv2.imshow('image',NDVI_i)
cv2.waitKey(0)
cv2.destroyAllWindows()
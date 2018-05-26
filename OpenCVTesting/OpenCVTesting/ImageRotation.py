import cv2
import numpy as np
import imutils

img = cv2.imread('F:\\Flights\\Flight_2_05_2015\\IMG_1336.JPG') #Vamos a tratar de rotar la imagen
img_redux =  cv2.resize(img, (0,0), fx=0.25, fy=0.25) 
for angle in np.arange(0,360,1):            #Rotando la imagen de manera tradicional
    rotated = imutils.rotate(img_redux, angle)
    Ey , Ex , layers =  rotated.shape
    ma_string = 'img rotated traditional tamano '+ str(Ey)+' '+str(Ex)
    cv2.putText(rotated,ma_string, (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv2.imshow('normalsete',rotated)
    cv2.waitKey(0)
for angle in np.arange(0,360,1):            #Rotando la imagen de manera preservativa de angulo
    rotated = imutils.rotate_bound(img_redux, angle)
    Ey , Ex , layers =  rotated.shape
    ma_string = 'img rotated traditional tamano '+ str(Ey)+' '+str(Ex)
    cv2.putText(rotated,ma_string, (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv2.imshow('modificado',rotated)
    cv2.waitKey(0)


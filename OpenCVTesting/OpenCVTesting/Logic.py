import cv2
import numpy as np
img = cv2.imread('3D-Matplotlib.png', cv2.IMREAD_COLOR)
img2 = cv2.imread('mainlogo.png', cv2.IMREAD_COLOR)

rows,cols,other = img2.shape # extract size of the image
roi = img[0:rows,0:cols]

img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret,mask = cv2.threshold(img2gray,220,255,cv2.THRESH_BINARY_INV) #If pixel value is more than 220, it is converted to white, and if it is below it gets black (inverted this case due to THRESH_BINARY_INV)
#cv2.imshow('mask',mask)
mask_inv = cv2.bitwise_not(mask) #elementwise not, inverting what we did bifore
#cv2.imshow('mask inv',mask_inv)
img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv) # leaves in the image (since the same image is compared) the ones with 255 on the mask
cv2.imshow('img_bg',img1_bg )
img2_fg = cv2.bitwise_and(img2, img2, mask = mask)

dst = cv2.add(img1_bg, img2_fg)

img[0:rows,0:cols] = dst

cv2.imshow('result',img)


#cv2.imshow('mask',mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
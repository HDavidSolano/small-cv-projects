import cv2
import numpy as np
#The idea is to detect an image from the template
img1 = cv2.imread('base_image_for_matching.jpg')
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
temp = cv2.imread('template_for_matching.jpg',0)

w,h = temp.shape[::-1]
res = cv2.matchTemplate(img1_gray, temp, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img1,pt,(pt[0]+w, pt[1]+h),(0,255,255),2)

cv2.imshow('image',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
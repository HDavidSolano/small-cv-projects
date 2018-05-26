import cv2
import numpy as np
import matplotlib.pyplot as plt

#img = cv2.imread('opencv-feature-matching-image.jpg')
#reference = cv2.imread('opencv-feature-matching-template.jpg')

img = cv2.imread('F:\\Flights\\Hatillo\\Hatillo\\geotagged_Offset\\DSC04002_geotag.JPG')
reference = cv2.imread('F:\\Flights\\Hatillo\\Hatillo\\geotagged_Offset\\DSC04000_geotag.JPG')

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(reference,None)
kp2, des2 = orb.detectAndCompute(img,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING , crossCheck = True)

matches = bf.match(des1,des2)

matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(reference,kp1,img,kp2,matches[:20],None,flags=2)
plt.imshow(img3)
plt.show()
#cv2.imshow('result',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
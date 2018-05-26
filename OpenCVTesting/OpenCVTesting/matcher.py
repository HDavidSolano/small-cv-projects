import cv2
import numpy as np
import matplotlib.pyplot as plt
import math as m
import imutils as im

#4001 4002 work very well
#img = cv2.imread('C:\\Users\\David\\Desktop\\imagenes vuelo flir jpg\\20171125_143657.jpg')
#reference = cv2.imread('C:\\Users\\David\\Desktop\\imagenes vuelo flir jpg\\20171125_143659.jpg')
#img = cv2.imread('F:\\Flights\\Flight_2_05_2015\\IMG_1336.JPG')
#reference = cv2.imread('F:\\Flights\\Flight_2_05_2015\\IMG_1337.JPG')

img = cv2.imread('F:\\Flights\\Hatillo\\Hatillo\\geotagged_Offset\\DSC04052_geotag.JPG')
reference = cv2.imread('F:\\Flights\\Hatillo\\Hatillo\\geotagged_Offset\\DSC04053_geotag.JPG')

Ey , Ex , layers =  img.shape

Ey = Ey - 1
Ex = Ex - 1

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(reference,None)
kp2, des2 = orb.detectAndCompute(img,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING , crossCheck = True)

matches = bf.match(des1,des2)

matches = sorted(matches, key = lambda x:x.distance)

# LOOKING ANGLE OFFSET BETWEEN IMAGES

limit_matches = 20 #Max number of matches to use
delta_angle = np.zeros(shape=(1,limit_matches))
j = 0
for a_match in matches:
    img_index = a_match.queryIdx
    reference_index = a_match.trainIdx
    delta_angle[0,j] = (kp1[img_index].angle- kp2[reference_index].angle)
    j = j + 1
    if j >= limit_matches:
        break
angle = -np.median(delta_angle)

img = im.rotate(img, angle)

kp2, des2 = orb.detectAndCompute(img,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING , crossCheck = True)

matches = bf.match(des1,des2)

matches = sorted(matches, key = lambda x:x.distance)

point_displacements = []
point_locations = []

x_cum = 0
y_cum = 0

for a_match in matches:
    img_index = a_match.queryIdx
    reference_index = a_match.trainIdx
    img_loc = kp1[img_index].pt
    ref_loc = kp2[reference_index].pt
    locations = [img_loc[0],img_loc[1],ref_loc[0],ref_loc[1]]
    displacements = [ref_loc[0]-img_loc[0],ref_loc[1]-img_loc[1]]
    x_cum = x_cum + ref_loc[0]-img_loc[0]
    y_cum = y_cum + ref_loc[1]-img_loc[1]
    point_locations.append(locations)
    point_displacements.append(displacements)
    if len(point_displacements) >= limit_matches:
        break

x_cum = float(x_cum/limit_matches)     #Denotes the average displacement of the second picture with respect to the first
y_cum = float(y_cum/limit_matches)     #Denotes the average displacement of the first picture with respect to the second
x_index = int(m.fabs(x_cum))
y_index = int(m.fabs(y_cum))
if x_cum < 0 and y_cum < 0:     #Searches for the specific displacement case to re-combine
    img_roi = img[0:(Ey-y_index),0:(Ex-x_index)]
    reference_roi = reference[(0+y_index):Ey,(0+x_index):Ex]
if x_cum > 0 and y_cum < 0:
    img_roi = img[0:(Ey-y_index),(0+x_index):Ex]
    reference_roi = reference[(0+y_index):Ey,0:(Ex-x_index)]
if x_cum < 0 and y_cum > 0:
    img_roi = img[(0+y_index):Ey,0:(Ex-x_index)]
    reference_roi = reference[0:(Ey-y_index),(0+x_index):Ex]
if x_cum > 0 and y_cum > 0:
    img_roi = img[(0+y_index):Ey,(0+x_index):Ex]
    reference_roi = reference[0:(Ey-y_index),0:(Ex-x_index)]
    
img3 = cv2.drawMatches(reference,kp1,img,kp2,matches[:20],None,flags=2)
img3_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img_average = cv2.addWeighted(img_roi,0.5,reference_roi,0.5,0)
img_average_rgb = cv2.cvtColor(img_average, cv2.COLOR_BGR2RGB)
plt.figure(1)
ax = plt.subplot(211)
ax.set_title("Puntos de Emparejamiento")
ax.imshow(img3_rgb)
res = np.concatenate((img_roi,reference_roi),axis = 1)
res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
bx = plt.subplot(212)
bx.set_title("Seccion que Concuerda")
bx.imshow(res_rgb)
cx = plt.figure(2)
cx.suptitle("Combinacion de Fotos", fontsize=16)
plt.imshow(img_average_rgb)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

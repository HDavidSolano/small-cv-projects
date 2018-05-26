import numpy as np
from time import sleep
from CamRunnable import camVideoStream
import math as m
import cv2

frame_lim = 200   # Number of points to store
limit_matches = 20 # Max number of matches to use per frame
angle_meas_normal = np.zeros(shape=(frame_lim,limit_matches+1)) #Lists of measurements to show statistics from
angle_meas_CLAHE = np.zeros(shape=(frame_lim,limit_matches+1))

curr_frame = 0

orb_norm = cv2.ORB_create() #Setting up independent orbs, just for clarity
orb_CLAHE = cv2.ORB_create() 
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) # Setting up the histogram equalizer
cam_holder = camVideoStream(0,30,640,480)
cam_holder.start()
while curr_frame <= frame_lim:
    sleep(0.05)
    frame, t = cam_holder.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if curr_frame == 0: # capture a reference image
        kp1_norm, des1_norm = orb_norm.detectAndCompute(gray,None) # reference of unaltered frame
        cl_gray = clahe.apply(gray)
        kp1_CLAHE, des1_CLAHE = orb_norm.detectAndCompute(cl_gray,None) # reference of unaltered frame
    else:
        kp2_norm, des2_norm = orb_norm.detectAndCompute(gray,None)
        laplacian_normal = cv2.Laplacian(gray, cv2.CV_64F).var()
        cl_gray = clahe.apply(gray)
        kp2_CLAHE, des2_CLAHE = orb_CLAHE.detectAndCompute(cl_gray,None) # reference of unaltered frame
        laplacian_CLAHE = cv2.Laplacian(cl_gray, cv2.CV_64F).var()
        if kp2_norm and kp2_CLAHE: # if both frames were tractable
            bf = cv2.BFMatcher(cv2.NORM_HAMMING , crossCheck = True)
            matches_norm = bf.match(des1_norm,des2_norm)
            matches_norm = sorted(matches_norm, key = lambda x:x.distance)
            matches_CLAHE = bf.match(des1_CLAHE,des2_CLAHE)
            matches_CLAHE = sorted(matches_CLAHE, key = lambda x:x.distance)
            delta_angle_norm = np.zeros(shape=(1,limit_matches))
            delta_angle_CLAHE = np.zeros(shape=(1,limit_matches))
            j = 0
            for a_match in matches_norm:
                img_index = a_match.queryIdx
                reference_index = a_match.trainIdx
                angle = (kp1_norm[img_index].angle- kp2_norm[reference_index].angle)
                if abs(angle) > 180: #Some crazy angles require this check
                    if angle > 180:
                        angle = angle - 360
                    else:
                        angle = angle + 360
                angle_meas_normal[curr_frame-1,j] = angle
                j = j + 1
                if j >= limit_matches:
                    break
            angle_meas_normal[curr_frame-1,j] = laplacian_normal
            j = 0
            for a_match in matches_CLAHE:
                img_index = a_match.queryIdx
                reference_index = a_match.trainIdx
                angle = (kp1_CLAHE[img_index].angle- kp2_CLAHE[reference_index].angle)
                if abs(angle) > 180: #Some crazy angles require this check
                    if angle > 180:
                        angle = angle - 360
                    else:
                        angle = angle + 360
                angle_meas_CLAHE[curr_frame-1,j] = angle
                j = j + 1
                if j >= limit_matches:
                    break
            angle_meas_CLAHE[curr_frame-1,j] = laplacian_CLAHE
    curr_frame = curr_frame + 1
cam_holder.stop()
np.savetxt("norm.csv", angle_meas_normal, delimiter=",")
np.savetxt("CLAHE.csv", angle_meas_CLAHE, delimiter=",")
cv2.destroyAllWindows()

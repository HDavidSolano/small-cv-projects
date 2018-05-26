import numpy as np
from CamRunnable import camVideoStream
import math as m
from time import sleep
import cv2

"""frame_lim = 2000       # Number of independent frames to analyze
curr_frame = 0
previous_time = -1
curr_frame = 0
cam_holder = camVideoStream(0,30,640,480)
cam_holder.start()

min_area = 2500
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=50)"""

def get_movement(fgbg,current_frame,min_area): #Field substractor is handled OUTSIDE the function since it has tu endure time
    fgmask = fgbg.apply(current_frame)
    thresh = cv2.threshold(fgmask, 120, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(fgmask, None, iterations=2)     # THis function makes thresholded blobs bigger!!!
    (image, contours, hierarchy) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Second important function!!!! Very useful!!!
    X = []
    Y = []
    W = []
    H = []
    A = []
    Cont_char = []
    returned_movement = False
    for c in contours:
		# if the contour is too small, ignore it
            if cv2.contourArea(c) < min_area:
                continue
            cont_area = cv2.contourArea(c) #Store contour area
		# Compute bounding box of each detected movement
            (x, y, w, h) = cv2.boundingRect(c)
            Cont_char.append([x,y,w,h,cont_area])
        # Sort contours from smallest to largest
    if Cont_char:
        Cont_char = sorted(Cont_char,key=lambda area:area[4])
        X,Y,W,H,A = Cont_char[-1]
        returned_movement = True
    return X,Y,W,H,A, returned_movement
"""
while curr_frame <= frame_lim:
    frame, time = cam_holder.read()
    if time != previous_time:
        X,Y,W,H,A, returned_movement = get_movement(fgbg,frame,min_area)
        if returned_movement:
            cv2.rectangle(frame, (X, Y), (X + W, Y + H), (0, 255, 0), 2)
        # show the frame and record if the user presses a key
        cv2.imshow("Security Feed", frame)
        previous_time = time
        curr_frame = curr_frame + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        sleep(0.017)
cam_holder.stop()
cv2.destroyAllWindows()"""

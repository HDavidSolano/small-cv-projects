import numpy as np
from CamRunnable import camVideoStream
import math as m
from time import sleep
import cv2

frame_lim = 2000       # Number of independent frames to analyze
curr_frame = 0
previous_time = -1
curr_frame = 0
cam_holder = camVideoStream(0,30,640,480)
cam_holder.start()

# initialize the first frame in the video stream
firstFrame = None
min_area = 2500
frame_refresh = 3
while curr_frame <= frame_lim:
    a_frame, time = cam_holder.read()
    if time != previous_time:
        gray = cv2.cvtColor(a_frame,cv2.COLOR_BGR2GRAY) 
        gray = cv2.GaussianBlur(gray, (21, 21), 0)  # Lets blur the frame
        if (firstFrame is None) or (curr_frame%frame_refresh) == 0:                      # Grabbing the first frame
            firstFrame = gray
            text = "Unoccupied"
            curr_frame = curr_frame + 1
            continue
        # compute the absolute difference between the current frame and
	    # first frame
        frameDelta = cv2.absdiff(firstFrame, gray)   # Important FIRST FUNCTION TO LEARN!!!!
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
	    # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=5)     # THis function makes thresholded blobs bigger!!!
        (image, contours, hierarchy) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Second important function!!!! Very useful!!!

        # loop over the contours
        for c in contours:
		# if the contour is too small, ignore it
            if cv2.contourArea(c) < min_area:
                continue
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(a_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Occupied"
        cv2.putText(a_frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # show the frame and record if the user presses a key
        cv2.imshow("Security Feed", a_frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        previous_time = time
        curr_frame = curr_frame + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    else:
        sleep(0.017)
cam_holder.stop()
cv2.destroyAllWindows()
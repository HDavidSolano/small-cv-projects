import numpy as np
from CamRunnable import camVideoStream
import math as m
from time import sleep
from FrameAnalyzer import FrameAnalyzer
import cv2

frame_lim = 2000   # Number of points to store
limit_matches = 5 # Max number of matches to use per frame
candidates = 4 # number of candidates to consider to select a reference frame
frame_buffer = 20 #number of frames to store at a given time before a reference must be changed
curr_frame = 0

cam_holder = camVideoStream(0,30,640,480)
camera1 = FrameAnalyzer(frame_buffer,limit_matches,candidates)
cam_holder.start()

'''pic, stime = cam_holder.read()   #the the starting point
cv2.imshow('reference',pic)
cv2.waitKey(0)'''
previous_time = -1
while curr_frame <= frame_lim:
    a_frame, time = cam_holder.read()
    if time != previous_time:
        camera1.adaptFrame(a_frame,time)
        previous_time = time
    else:
        sleep(0.017)
    curr_frame = curr_frame + 1
cam_holder.stop()
cv2.destroyAllWindows()
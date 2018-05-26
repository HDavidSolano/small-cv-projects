import datetime
from time import sleep
import cv2
from CamRunnable import camVideoStream


cam_holder = camVideoStream(0,30,1280,720)

cam_holder.start()
sleep(2)
ma_frame = cam_holder.read()

cv2.imshow('maimage',ma_frame)
print(str(cam_holder.frameNum))
sleep(1)

ma_frame = cam_holder.read()

cv2.imshow('maimage2',ma_frame)
cam_holder.stop()
print(str(cam_holder.frameNum))

cam = cv2.VideoCapture(0)
frame_q = 0
ma_start = datetime.datetime.now()
while frame_q <= cam_holder.frameNum:
    ret, frame = cam.read()
    frame_q += 1
ma_end = datetime.datetime.now()
total = (ma_end-ma_start).total_seconds()
print(str(total))
cv2.waitKey()
cv2.destroyAllWindows()

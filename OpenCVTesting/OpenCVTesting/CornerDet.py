import cv2
import numpy as np

cap = cv2.VideoCapture(0)
# If i want to load a video file I can do:  cap = cv2.VideoCapture('the_video.avi') and count frames with max_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
while True:                 # here you get within each frame within the video
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.goodFeaturesToTrack(gray,100,0.08,10)
    corners = np.int0(corners)

    for corner in corners:
        x,y = corner.ravel()
        cv2.circle(frame,(x,y),3,255,-1)
    cv2.imshow('corners',frame)
    if ret == False:        #exit if there ar eno more frames to grab
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
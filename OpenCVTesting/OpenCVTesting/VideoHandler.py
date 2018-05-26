import cv2
import numpy as np

cap = cv2.VideoCapture(0)
# If i want to load a video file I can do:  cap = cv2.VideoCapture('the_video.avi') and count frames with max_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc,30.0,(640,480))
while True:                 # here you get within each frame within the video
    ret, frame = cap.read()
    #cv2.imwrite('afoto.png',frame) si queremos sacar una fotico
    #break
    if ret == False:        #exit if there ar eno more frames to grab
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    out.write(frame)
    cv2.imshow('gray',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows

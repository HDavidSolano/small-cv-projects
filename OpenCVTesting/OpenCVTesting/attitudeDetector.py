import cv2
import imutils
import numpy as np

limit_matches = 20 #Max number of matches to use
frame_reference_recapture = 30 # after how many frames we want to re-capture a reference image to make calcuations from
cap = cv2.VideoCapture(0)
frame_q = 0 # Con esto vamos a contar los frames
orb = cv2.ORB_create()
carry_angle = 0 #An angle saved to preserve initial attitude
angle = 0 #The relative position of a reference with respect a new frame
test_CLAHE = 1 #Enable CLAHE before frame
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) # Setting up the histogram equalizer
while True:                 # here you get within each frame within the video
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if test_CLAHE:
        gray = clahe.apply(gray)
    if frame_q%frame_reference_recapture == 0: #Every now and then capture a new reference image
        kp1, des1 = orb.detectAndCompute(gray,None)
        carry_angle = angle # Preserve reference frame
    else:
        kp2, des2 = orb.detectAndCompute(gray,None)
        if kp2:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING , crossCheck = True)
            matches = bf.match(des1,des2)
            matches = sorted(matches, key = lambda x:x.distance)
            delta_angle = np.zeros(shape=(1,limit_matches))
            j = 0
            for a_match in matches:
                img_index = a_match.queryIdx
                reference_index = a_match.trainIdx
                angle = (kp1[img_index].angle- kp2[reference_index].angle)
                if abs(angle) > 180: #Some crazy angles require this check
                    if angle > 180:
                        angle = angle - 360
                    else:
                        angle = angle + 360
                delta_angle[0,j] = angle
                j = j + 1
                if j >= limit_matches:
                    break
            angle = -np.median(delta_angle) + carry_angle
    ma_string = 'rot '+ str(angle)
    cv2.putText(gray,ma_string, (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv2.imshow('gray',gray)
    frame_q = frame_q + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows


import cv2
import numpy as np
import matplotlib.pyplot as plt

#img = cv2.imread('afoto.png')
#mask = np.zeros(img.shape[:2],np.uint8)
scale = 2
cap = cv2.VideoCapture(0)
bgd_model = np.zeros((1,65),np.float64)
fgd_model = np.zeros((1,65),np.float64)
rect = (int(50/scale),int(50/scale),int(620/scale),int(460/scale))
while True:
    ret, frame = cap.read()
    height , width , layers =  frame.shape
    new_h=int(height/2)
    new_w=int(width/2)
    frame = cv2.resize(frame, (new_w, new_h)) 
    mask = np.zeros(frame.shape[:2],np.uint8)
    cv2.grabCut(frame,mask,rect,bgd_model,fgd_model,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = frame*mask2[:,:,np.newaxis]
    cv2.imshow('feed',img)
    #plt.imshow(img)
    #plt.colorbar()
    #plt.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()




import numpy as np
from CamRunnable import camVideoStream
import math as m
from time import sleep
import cv2
import Plotter

frame_lim = 2000       # Number of independent frames to analyze
group_scale = 100       # How many key points will be tracked
use_CLAHE = False      # Whether or not normalize for brightness
buffer_comparator = [] # Where I will store the information
buffer_size = 3        # Number of frames to be considered to attempt a velocity estimation
animate = True
point_velocities = np.zeros(shape=(group_scale,2))   # Column matrix containing all velocities being monitored at any given time
point_distances = np.zeros(shape=(group_scale,2))    # Column matrix containing all velocities being monitored at any given time
orb = cv2.ORB_create() # Oriented FAST and rotated BRIEF characterizer to use within the system # nfeatures = 500,scaleFactor = 1.2,nlevels = 8,edgeThreshold = 31,firstLevel = 0,WTA_K = 2,scoreType = ORB.HARRIS_SCORE,patchSize = 31,fastThreshold = 20
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) # Setting up the histogram equalizer (these numbers work well for my web camera)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)      # Matching object being initialized beforehand
previous_time = -1
curr_frame = 0
cam_holder = cv2.VideoCapture('C:\\Users\\David\\Desktop\\maVideo.mp4')
#cam_holder = camVideoStream(0,30,640,480)
#cam_holder.start()

record_count = 0

division_bin = 50      # To differentiate points
max_uniaxial_bins = 10 # This number squared will divide the clusters of data
threshold_min = 20
threshold_max = division_bin*max_uniaxial_bins+threshold_min
time = 0
if animate: my_plot = Plotter.plotter(200,200,50,50,2,False)
def compare_velocities(buffer,point_velocities,point_distances,image):
    velocity_organizer = np.zeros(shape=(2*max_uniaxial_bins+1,2*max_uniaxial_bins+1,2))  # Refresh arrays used in velocity isolation
    position_organizer = np.zeros(shape=(2*max_uniaxial_bins+1,2*max_uniaxial_bins+1,2))
    frequency_organizer = np.zeros(shape=(2*max_uniaxial_bins+1,2*max_uniaxial_bins+1))
    extreme_points = np.zeros(shape=(2*max_uniaxial_bins+1,2*max_uniaxial_bins+1,4))
    captured_base = np.zeros(shape=(2*max_uniaxial_bins+1,2*max_uniaxial_bins+1))         # Check to accumulate the boundaries of the clusters

    T = threshold_min # some made up velocity threshold (40 pixels per second)
    
    stream = []
    minVx = []
    maxVx = []
    minVy = []
    maxVy = []
    position = []
    for a_reference1 in buffer:
        for a_reference2 in buffer:
            if a_reference2[2] > a_reference1[2]:   # Only compute matches IF the second reference has happened LATER than the first, 
                 matches = bf.match(a_reference1[1],a_reference2[1]) # giving us an upper triangular analysis of the buffers available: 12, 13, 14, 23, 24, 34 for a buffer of size 4, for example    
                 dt = a_reference2[2]-a_reference1[2]                # we need dt as well
                 matches = sorted(matches, key = lambda x:x.distance)
                 i = 0
                 for a_match in matches:
                    img_index = a_match.queryIdx
                    reference_index = a_match.trainIdx
                    img_loc = a_reference1[0][img_index].pt
                    ref_loc = a_reference2[0][reference_index].pt
                    locations = [img_loc[0],img_loc[1],ref_loc[0],ref_loc[1]]
                    displacements = [ref_loc[0]-img_loc[0],ref_loc[1]-img_loc[1]]
                    V_mag = m.sqrt((displacements[0]/dt)**2+(displacements[1]/dt)**2)
                    if m.fabs(displacements[0]/dt) < threshold_max and  m.fabs(displacements[1]/dt) < threshold_max:
                        if animate and V_mag > T: cv2.rectangle(image,(int(ref_loc[0]),int(ref_loc[1])),(int(ref_loc[0])+2,int(ref_loc[1])+2),(255,255,255),2)
                        if m.fabs(displacements[0]/dt) > threshold_min:
                            point_velocities[i][0] = (displacements[0]/dt)
                        else:
                            point_velocities[i][0] = 0
                        if m.fabs(displacements[1]/dt) > threshold_min:
                            point_velocities[i][1] = (displacements[1]/dt)
                        else:
                            point_velocities[i][1] = 0
                    x_bin_pos = int(np.sign(point_velocities[i][0])*m.ceil(m.fabs((point_velocities[i][0]-threshold_min*np.sign(point_velocities[i][0]))/division_bin)))+max_uniaxial_bins
                    y_bin_pos = int(np.sign(point_velocities[i][1])*m.ceil(m.fabs((point_velocities[i][1]-threshold_min*np.sign(point_velocities[i][1]))/division_bin)))+max_uniaxial_bins
                    if captured_base[x_bin_pos][y_bin_pos] == 0: # Captures extreme points of the cluster (first one is a default) [xmin xmax ymin ymax] per histogram slot
                        extreme_points[x_bin_pos][y_bin_pos][0] = int(ref_loc[0])
                        extreme_points[x_bin_pos][y_bin_pos][1] = int(ref_loc[0])
                        extreme_points[x_bin_pos][y_bin_pos][2] = int(ref_loc[1])
                        extreme_points[x_bin_pos][y_bin_pos][3] = int(ref_loc[1])
                        captured_base[x_bin_pos][y_bin_pos] = 1
                    else:
                        if int(ref_loc[0]) > extreme_points[x_bin_pos][y_bin_pos][1]:
                            extreme_points[x_bin_pos][y_bin_pos][1] = int(ref_loc[0])
                        if int(ref_loc[0]) < extreme_points[x_bin_pos][y_bin_pos][0]:
                            extreme_points[x_bin_pos][y_bin_pos][0] = int(ref_loc[0])
                        if int(ref_loc[1]) > extreme_points[x_bin_pos][y_bin_pos][3]:
                            extreme_points[x_bin_pos][y_bin_pos][3] = int(ref_loc[1])
                        if int(ref_loc[1]) < extreme_points[x_bin_pos][y_bin_pos][2]:
                            extreme_points[x_bin_pos][y_bin_pos][2] = int(ref_loc[1])
                    velocity_organizer[x_bin_pos][y_bin_pos][0] += (point_velocities[i][0]) # Here classification magic happens!!
                    velocity_organizer[x_bin_pos][y_bin_pos][1] += (point_velocities[i][1]) # Here classification magic happens!!
                    position_organizer[x_bin_pos][y_bin_pos][0] += int(ref_loc[0]) # Here classification magic happens!!
                    position_organizer[x_bin_pos][y_bin_pos][1] += int(ref_loc[1]) # Here classification magic happens!!
                    frequency_organizer[x_bin_pos][y_bin_pos] += 1
                    point_distances[i][0] = int(ref_loc[0])
                    point_distances[i][1] = int(ref_loc[1])
                    i+=1
                    if i >= group_scale:
                        minVx.append(np.min(point_velocities[:,0]))
                        maxVx.append(np.max(point_velocities[:,0]))
                        minVy.append(np.min(point_velocities[:,1]))
                        maxVy.append(np.max(point_velocities[:,1]))
                        break
                 stream.append(np.copy(point_velocities))
                 position.append(np.copy(point_distances))
    for x in range(len(frequency_organizer)):
        for y in range(len(frequency_organizer[0])):
            if frequency_organizer[x][y] > 0:
                velocity_organizer[x][y][0] = velocity_organizer[x][y][0]/(frequency_organizer[x][y])
                velocity_organizer[x][y][1] = velocity_organizer[x][y][1]/(frequency_organizer[x][y])
                position_organizer[x][y][0] = position_organizer[x][y][0]/(frequency_organizer[x][y])
                position_organizer[x][y][1] = position_organizer[x][y][1]/(frequency_organizer[x][y])
            if frequency_organizer[x][y] > 0 and (x != max_uniaxial_bins and y != max_uniaxial_bins):
                if animate:
                    x_pos = int(position_organizer[x][y][0])
                    y_pos = int(position_organizer[x][y][1])
                    V = m.sqrt((velocity_organizer[x][y][0])**2+(velocity_organizer[x][y][1])**2)
                    cv2.rectangle(image,(x_pos,y_pos),(x_pos+2,y_pos+2),(0,0,0),2)
                    cv2.rectangle(image,(int(extreme_points[x][y][0]),int(extreme_points[x][y][2])),(int(extreme_points[x][y][1]),int(extreme_points[x][y][3])),(238,130,238),2)
                    cv2.arrowedLine(image,(x_pos,y_pos),(x_pos+int(velocity_organizer[x][y][0]/10),y_pos+int(velocity_organizer[x][y][1]/10)),(0,255,0),thickness=1)
                    cv2.putText(image,str('%.3f'%V),(x_pos,y_pos), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
    if animate: my_plot.plot_information(image,stream,[-500,500],[-500,500])
    return stream, position
while curr_frame <= frame_lim:
    grabbed, a_frame = cam_holder.read()
    if grabbed == False:
        break
    a_frame = cv2.resize(a_frame, (640, 480)) 
    time += 1/30
    #a_frame, time = cam_holder.read()
    if time != previous_time:
        gray_frame = cv2.cvtColor(a_frame,cv2.COLOR_BGR2GRAY) 
        if use_CLAHE:
            gray_cl = clahe.apply(gray_frame)
            kp, des = orb.detectAndCompute(gray_cl,None)
        else:
            kp, des = orb.detectAndCompute(gray_frame,None)
        if kp:      # Proceed if detection was successful
            if record_count < buffer_size:    # If we have not captured enough frames to compute velocities
                buffer_comparator.append([kp,des,time])  # Save whatever you got in the current frame to be used
                record_count += 1                   # Let the program know that you have recorded a frame
            else:                                   # We now got enough frames to always be 'full'
                buffer_comparator = buffer_comparator[1:] # Erase first element from the list and move entire list of objects back
                buffer_comparator.append([kp,des,time])  # Save whatever you got in the current frame to be used in the LAST position
                stream, pos = compare_velocities(buffer_comparator,point_velocities,point_distances,a_frame)
                #compute_movement(stream,pos)
                cv2.imshow('original',a_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        previous_time = time
        curr_frame = curr_frame + 1
    else:
        sleep(0.017)
cam_holder.stop()
cv2.destroyAllWindows()

import numpy as np
from CamRunnable import camVideoStream
import math as m
from time import sleep
import cv2
import Plotter
from MovementBG_Ext_MOG import get_movement
import matplotlib.pyplot as plt

min_area = 500 # To weed out dud movements
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=50) # The extractor to use here

frame_lim = 500       # Number of independent frames to analyze % 30 for 5, 60 for 4 and for 3
group_scale = 5     # How many key points will be tracked
use_CLAHE = True     # Whether or not normalize for brightness
buffer_comparator = [] # Where I will store the information
movement_rectangles = []
buffer_size = 3        # Number of frames to be considered to attempt a velocity estimation
animate = True
point_velocities = np.zeros(shape=(group_scale,2))   # Column matrix containing all velocities being monitored at any given time
point_distances = np.zeros(shape=(group_scale,2))    # Column matrix containing all velocities being monitored at any given time
orb = cv2.ORB_create() # Oriented FAST and rotated BRIEF characterizer to use within the system # nfeatures = 500,scaleFactor = 1.2,nlevels = 8,edgeThreshold = 31,firstLevel = 0,WTA_K = 2,scoreType = ORB.HARRIS_SCORE,patchSize = 31,fastThreshold = 20
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) # Setting up the histogram equalizer (these numbers work well for my web camera)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)      # Matching object being initialized beforehand
previous_time = -1
curr_frame = 0

# Settings for webcan c310
Height = 480
Width = 640
#Fov = 43.6 # Camera's field of view in degrees
#Settings for cellphone camera
#Height = 1080
#Width = 1920
Fov = 60.87 # Camera's field of view in degrees
'''cam_holder = camVideoStream(0,30,Width,Height)
cam_holder.start()'''
cam_holder = cv2.VideoCapture('C:\\Users\\David\\Desktop\\DynExp\\NewVideos\\T11.mp4')
record_count = 0

division_bin = 2      # To differentiate points
max_uniaxial_bins = 2 # This number squared will divide the clusters of data
threshold_min = 0.05
threshold_max = division_bin*max_uniaxial_bins+threshold_min

Camera_Distance = 1.6 # Distance from camera to track in meters


if animate: my_plot = Plotter.plotter(200,200,50,50,2,False)
def compare_velocities(buffer,point_velocities,point_distances,image,movement_rectangles,FOV,Z,W,H):
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
    classified_key_points = []
    R_1_Counter = 0
    for a_reference1 in buffer:
        R_2_Counter = 0
        for a_reference2 in buffer:
            if a_reference2[2] > a_reference1[2]:   # Only compute matches IF the second reference has happened LATER than the first, 
                 matches = bf.match(a_reference1[1],a_reference2[1]) # giving us an upper triangular analysis of the buffers available: 12, 13, 14, 23, 24, 34 for a buffer of size 4, for example    
                 dt = a_reference2[2]-a_reference1[2]                # we need dt as well
                 matches = sorted(matches, key = lambda x:x.distance)
                 i = 0
                 for a_match in matches:
                    img_index = a_match.queryIdx
                    reference_index = a_match.trainIdx
                    frame_t1 = (movement_rectangles[R_1_Counter][0]+a_reference1[0][img_index].pt[0],movement_rectangles[R_1_Counter][1]+a_reference1[0][img_index].pt[1]) # Return position in pixel coordinates of first frame
                    frame_t2 = (movement_rectangles[R_2_Counter][0]+a_reference2[0][reference_index].pt[0],movement_rectangles[R_2_Counter][1]+a_reference2[0][reference_index].pt[1]) # Return position in pixel coordinates of second frame
                    
                    # Here I will apply the transformation from pixel coordinates to image-center coordinates
                    x_21 = frame_t1[0]-W/2
                    y_21 = -frame_t1[1]+H/2

                    x_22 = frame_t2[0]-W/2
                    y_22 = -frame_t2[1]+H/2

                    # Then I proceed to transform from image centered to camera coordinates:
                    X_c1 = Z*m.tan(FOV*x_21/W)
                    Y_c1 = Z*m.tan(FOV*y_21/W)

                    X_c2 = Z*m.tan(FOV*x_22/W)
                    Y_c2 = Z*m.tan(FOV*y_22/W)

                    # Finally, I will calculate locations, displacements and velocities in my new frame
                    locations = [X_c1,Y_c1,X_c2,Y_c2]
                    displacements = [X_c2-X_c1,Y_c2-Y_c1]

                    V_mag = m.sqrt((displacements[0]/dt)**2+(displacements[1]/dt)**2)
                    if m.fabs(displacements[0]/dt) < threshold_max and  m.fabs(displacements[1]/dt) < threshold_max:
                        if animate and V_mag > T: cv2.rectangle(image,(int(frame_t2[0]),int(frame_t2[1])),(int(frame_t2[0])+2,int(frame_t2[1])+2),(255,255,255),2)
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
                        extreme_points[x_bin_pos][y_bin_pos][0] = int(frame_t2[0])
                        extreme_points[x_bin_pos][y_bin_pos][1] = int(frame_t2[0])
                        extreme_points[x_bin_pos][y_bin_pos][2] = int(frame_t2[1])
                        extreme_points[x_bin_pos][y_bin_pos][3] = int(frame_t2[1])
                        captured_base[x_bin_pos][y_bin_pos] = 1
                    else:
                        if int(frame_t2[0]) > extreme_points[x_bin_pos][y_bin_pos][1]:
                            extreme_points[x_bin_pos][y_bin_pos][1] = int(frame_t2[0])
                        if int(frame_t2[0]) < extreme_points[x_bin_pos][y_bin_pos][0]:
                            extreme_points[x_bin_pos][y_bin_pos][0] = int(frame_t2[0])
                        if int(frame_t2[1]) > extreme_points[x_bin_pos][y_bin_pos][3]:
                            extreme_points[x_bin_pos][y_bin_pos][3] = int(frame_t2[1])
                        if int(frame_t2[1]) < extreme_points[x_bin_pos][y_bin_pos][2]:
                            extreme_points[x_bin_pos][y_bin_pos][2] = int(frame_t2[1])
                    velocity_organizer[x_bin_pos][y_bin_pos][0] += (point_velocities[i][0]) # Here classification magic happens!!
                    velocity_organizer[x_bin_pos][y_bin_pos][1] += (point_velocities[i][1]) # Here classification magic happens!!
                    position_organizer[x_bin_pos][y_bin_pos][0] += int(frame_t2[0]) # Here classification magic happens!!
                    position_organizer[x_bin_pos][y_bin_pos][1] += int(frame_t2[1]) # Here classification magic happens!!
                    frequency_organizer[x_bin_pos][y_bin_pos] += 1
                    point_distances[i][0] = int(frame_t2[0])
                    point_distances[i][1] = int(frame_t2[1])
                    i+=1
                    if i >= group_scale:
                        minVx.append(np.min(point_velocities[:,0]))
                        maxVx.append(np.max(point_velocities[:,0]))
                        minVy.append(np.min(point_velocities[:,1]))
                        maxVy.append(np.max(point_velocities[:,1]))
                        break
                 stream.append(np.copy(point_velocities))
                 position.append(np.copy(point_distances))
            R_2_Counter += 1
        R_1_Counter += 1
    for x in range(len(frequency_organizer)):
        for y in range(len(frequency_organizer[0])):
            if frequency_organizer[x][y] > 0:
                velocity_organizer[x][y][0] = velocity_organizer[x][y][0]/(frequency_organizer[x][y])
                velocity_organizer[x][y][1] = velocity_organizer[x][y][1]/(frequency_organizer[x][y])
                position_organizer[x][y][0] = position_organizer[x][y][0]/(frequency_organizer[x][y])
                position_organizer[x][y][1] = position_organizer[x][y][1]/(frequency_organizer[x][y])
            if frequency_organizer[x][y] > 0 and (x != max_uniaxial_bins and y != max_uniaxial_bins):
                classified_key_points.append([position_organizer[x][y][0],position_organizer[x][y][1],velocity_organizer[x][y][0],velocity_organizer[x][y][1],buffer[-1][2]])
                if animate:
                    x_pos = int(position_organizer[x][y][0])
                    y_pos = int(position_organizer[x][y][1])
                    V = m.sqrt((velocity_organizer[x][y][0])**2+(velocity_organizer[x][y][1])**2)
                    cv2.rectangle(image,(x_pos,y_pos),(x_pos+2,y_pos+2),(0,0,0),2)
                    cv2.rectangle(image,(int(extreme_points[x][y][0]),int(extreme_points[x][y][2])),(int(extreme_points[x][y][1]),int(extreme_points[x][y][3])),(238,130,238),2)
                    cv2.arrowedLine(image,(x_pos,y_pos),(x_pos+int(velocity_organizer[x][y][0]/10),y_pos+int(velocity_organizer[x][y][1]/10)),(0,255,0),thickness=1)
                    cv2.putText(image,str('%.3f'%V),(x_pos,y_pos), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
    if animate: my_plot.plot_information(image,stream,[-2,2],[-2,2])
    classified_key_points = np.array(classified_key_points)
    return stream, position, classified_key_points

time = 0
time_prev = 0
grabbed = True
Fov = Fov*m.pi/180
Vx_kp = []
Vx_raw = []
pos_raw = []
time_series = []
while curr_frame <= frame_lim:
    grabbed, a_frame = cam_holder.read()
    time_new = cam_holder.get(cv2.CAP_PROP_POS_MSEC)/1000
    if grabbed == False:
        break
    a_frame = cv2.resize(a_frame, (640, 480)) 

    time += (time_new-time_prev)
    #a_frame, time = cam_holder.read()
    if time != previous_time:
        X,Y,W,H,A, returned_movement = get_movement(fgbg,a_frame,min_area)      # Here we will see if the extracted frame has any relevant movement on it
        if returned_movement:                                                   # Only continue if there is anything relevant happening (To save computational time)
            movement_frame = [X,Y,W,H]                                          # Create an array with the relevant movement frame info
            sub_frame = a_frame[Y:Y+H, X:X+W]                                   # Down select movement area for analysis
            gray_frame = cv2.cvtColor(sub_frame,cv2.COLOR_BGR2GRAY) 
            if use_CLAHE:
                gray_cl = clahe.apply(gray_frame)
                kp, des = orb.detectAndCompute(gray_cl,None)
            else:
                kp, des = orb.detectAndCompute(gray_frame,None)
            if kp:      # Proceed if detection was successful
                if record_count < buffer_size:    # If we have not captured enough frames to compute velocities
                    buffer_comparator.append([kp,des,time])  # Save whatever you got in the current frame to be used
                    movement_rectangles.append(movement_frame) # Save new rectangle
                    record_count += 1                   # Let the program know that you have recorded a frame
                else:                                   # We now got enough frames to always be 'full'
                    buffer_comparator = buffer_comparator[1:] # Erase first element from the list and move entire list of objects back
                    movement_rectangles = movement_rectangles[1:]
                    buffer_comparator.append([kp,des,time])  # Save whatever you got in the current frame to be used in the LAST position
                    movement_rectangles.append(movement_frame) # Save new rectangle for the LAST position
                    stream, pos, key_points = compare_velocities(buffer_comparator,point_velocities,point_distances,a_frame,movement_rectangles,Fov,Camera_Distance,Width,Height)
                    #Vx_kp.append(np.mean(key_points[:,2]))
                    #Vx_raw.append(np.mean(stream[0][:,0]))
                    Vx_raw.append((np.mean(stream[0][:,0])+np.mean(stream[1][:,0])+np.mean(stream[2][:,0]))/3)
                    pos_raw.append((np.mean(pos[0][:,0])+np.mean(pos[1][:,0])+np.mean(pos[2][:,0]))/3)
                    time_series.append(time)
                    
                    #compute_movement(stream,pos)
                    cv2.rectangle(a_frame, (X, Y), (X + W, Y + H), (0, 255, 0), 2) # To see the captured area
                    cv2.imshow('original',a_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            previous_time = time
            time_prev = time_new
            #sleep(0.017)
            curr_frame = curr_frame + 1
    else:
        sleep(0.017)
csvfilex = 'C:\\Users\\David\\source\\repos\\CVTutorial21\\CVTutorial21\\outX.csv'
csvfilev = 'C:\\Users\\David\\source\\repos\\CVTutorial21\\CVTutorial21\\outV.csv'
np.savetxt(csvfilex, pos_raw, delimiter=",")
np.savetxt(csvfilev, Vx_raw, delimiter=",")
fig, ax = plt.subplots()
ax.set_title('Perceived Velocity as car runs')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Velocity (m/s)')
ax.plot(time_series,Vx_raw,'--', linewidth=2)
ax.grid(True)
plt.show()  

#cam_holder.stop()
cv2.destroyAllWindows()
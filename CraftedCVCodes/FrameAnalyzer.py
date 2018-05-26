import cv2
import numpy as np

class FrameAnalyzer:
    def __init__(self,frame_buffer,matches,candidate_frames):
        self.buffer_size = frame_buffer-1 #Last index of buffer
        self.matches = matches
        self.timeFrames = np.zeros(shape=(1,frame_buffer))
        self.angleFrames = np.zeros(shape=(matches,frame_buffer))
        self.velocitiesFrames = np.zeros(shape=(matches,frame_buffer))
        self.angleAbs = np.zeros(shape=(matches,frame_buffer))
        self.angleHist = np.zeros(shape=(1,frame_buffer))
        self.geometry = [None]*frame_buffer   #goes from zero to frame_buffer - 1
        self.laplacianFrames = np.zeros(shape=(1,frame_buffer))
        self.candidates = candidate_frames
        self.carry_angle = 0
        self.buffer_position = 0
        self.estimated_angle = 0
        self.orb = cv2.ORB_create() 
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) # Setting up the histogram equalizer
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    def adaptFrame(self,frame,fr_time):
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
        gray_cl = self.clahe.apply(gray_frame)
        kp, des = self.orb.detectAndCompute(gray_cl,None)
        if kp:
            laplacian = cv2.Laplacian(gray_cl, cv2.CV_64F).var()
            if self.buffer_position == 0 or self.buffer_position == self.buffer_size: #if reference updated
                self.compute_reference(frame,fr_time,kp,des,laplacian)
            else: #Normal computation starts here
                self.geometry[self.buffer_position] = (kp, des)
                self.timeFrames[0,self.buffer_position] = fr_time
                self.laplacianFrames[0,self.buffer_position] = laplacian
                #Computing contiguous movement
                matches_immediate = self.bf.match(self.geometry[self.buffer_position-1][1],self.geometry[self.buffer_position][1]) #evaluate current-previous
                matches_immediate = sorted(matches_immediate, key = lambda x:x.distance)
                j = 0
                for a_match in matches_immediate:
                    img_index = a_match.queryIdx
                    reference_index = a_match.trainIdx
                    angle = (self.geometry[self.buffer_position][0][reference_index].angle-self.geometry[self.buffer_position-1][0][img_index].angle)
                    if abs(angle) > 180: #Some crazy angles require this check
                        if angle > 180:
                            angle = angle - 360
                        else:
                            angle = angle + 360
                    self.angleFrames[j,self.buffer_position] = angle #Record relative angle change
                    self.velocitiesFrames[j,self.buffer_position] = angle/(self.timeFrames[0,self.buffer_position]-self.timeFrames[0,self.buffer_position-1]) #Record angular velocity change
                    j = j + 1
                    if j >= self.matches:
                        break
                #Computing Absolute Movement
                matches_absolute = self.bf.match(self.geometry[0][1],self.geometry[self.buffer_position][1]) #evaluate current-previous
                matches_absolute = sorted(matches_absolute, key = lambda x:x.distance)
                j = 0
                for a_match in matches_absolute:
                    img_index = a_match.queryIdx
                    reference_index = a_match.trainIdx
                    angle = (self.geometry[self.buffer_position][0][reference_index].angle-self.geometry[0][0][img_index].angle)
                    if abs(angle) > 180: #Some crazy angles require this check
                        if angle > 180:
                            angle = angle - 360
                        else:
                            angle = angle + 360
                    self.angleAbs[j,self.buffer_position] = angle #Record relative angle change
                    j = j + 1
                    if j >= self.matches:
                        break
                #Here comes the leverage of previous frame movement and current frame movement to make a compromised angular change:
                if self.buffer_position-1 != 0:
                    New_angle_1 = np.median(self.angleAbs[:,self.buffer_position])+self.carry_angle
                    New_angle_2 = np.median(self.angleFrames[:,self.buffer_position])+np.median(self.angleAbs[:,self.buffer_position-1])+self.carry_angle
                    self.estimated_angle = (New_angle_1+New_angle_2)/2
                else:
                    self.estimated_angle = np.median(self.angleAbs[:,self.buffer_position])+self.carry_angle
                print(str(self.estimated_angle))    
                self.angleHist[0,self.buffer_position] = self.estimated_angle
                self.buffer_position = self.buffer_position + 1
        else:
            print("Bad Frame")
    def compute_reference(self,frame,time,keypoints,descriptors,laplacian):
        if self.carry_angle == 0 and self.buffer_position == 0: #Is this the first frame? 
            self.timeFrames[0,self.buffer_position] = time        #Record reference time
            #No need to specify angles or velocities
            self.geometry[0]= (keypoints,descriptors)       #Record reference geometry on position zero of buffer
            self.buffer_position = self.buffer_position + 1     #move buffer one step
        else:    
            if self.buffer_position < self.candidates: #Checks if the number of recorded references (plus the new frame) is less than of candidate frames
                start_index = 1 
            else:
                start_index = self.buffer_position - (self.candidates-1) #Consider the k candidates (plus the new one that will come)
            print("Changing Reference")
            self.geometry[self.buffer_position] = (keypoints,descriptors) 
            self.timeFrames[0,self.buffer_position] = time
            self.laplacianFrames[0,self.buffer_position] = laplacian
            matches_immediate = self.bf.match(self.geometry[self.buffer_position-1][1],self.geometry[self.buffer_position][1]) #evaluate current-previous
            matches_immediate = sorted(matches_immediate, key = lambda x:x.distance)
            j = 0
            for a_match in matches_immediate:
                img_index = a_match.queryIdx
                reference_index = a_match.trainIdx
                angle = (self.geometry[self.buffer_position][0][reference_index].angle-self.geometry[self.buffer_position-1][0][img_index].angle)
                if abs(angle) > 180: #Some crazy angles require this check
                    if angle > 180:
                        angle = angle - 360
                    else:
                        angle = angle + 360
                self.angleFrames[j,self.buffer_position] = angle #Record relative angle change
                self.velocitiesFrames[j,self.buffer_position] = angle/(self.timeFrames[0,self.buffer_position]-self.timeFrames[0,self.buffer_position-1]) #Record angular velocity change
                j = j + 1
                if j >= self.matches:
                    break
            #Computing Absolute Movement
            matches_absolute = self.bf.match(self.geometry[0][1],self.geometry[self.buffer_position][1]) #evaluate current-previous
            matches_absolute = sorted(matches_absolute, key = lambda x:x.distance)
            j = 0
            for a_match in matches_absolute:
                img_index = a_match.queryIdx
                reference_index = a_match.trainIdx
                angle = (self.geometry[self.buffer_position][0][reference_index].angle-self.geometry[0][0][img_index].angle)
                if abs(angle) > 180: #Some crazy angles require this check
                    if angle > 180:
                        angle = angle - 360
                    else:
                        angle = angle + 360
                self.angleAbs[j,self.buffer_position] = angle #Record relative angle change
                j = j + 1
                if j >= self.matches:
                    break
            New_angle_1 = np.median(self.angleAbs[:,self.buffer_position])+self.carry_angle
            New_angle_2 = np.median(self.angleFrames[:,self.buffer_position])+np.median(self.angleAbs[:,self.buffer_position-1])+self.carry_angle
            self.angleHist[0,self.buffer_position] = (New_angle_1+New_angle_2)/2

            vel_variance = self.velocitiesFrames[:,start_index:self.buffer_position].var(0)
            score = np.divide(self.laplacianFrames[0,start_index:self.buffer_position],vel_variance) #IMPROVE UPON TESTING!!!!
            best = np.argmax(score) #Select the frame most likely to be 'good' amognst the frames
            # update buffer baseline
            self.geometry[0] = self.geometry[best+start_index]
            self.timeFrames[0,0] = self.timeFrames[0,best+start_index]
            self.carry_angle = self.angleHist[0,best+start_index] #update carry angle
            self.angleFrames = np.zeros(shape=(self.matches,self.buffer_size+1))
            self.velocitiesFrames = np.zeros(shape=(self.matches,self.buffer_size+1))
            self.angleAbs = np.zeros(shape=(self.matches,self.buffer_size+1))
            self.laplacianFrames = np.zeros(shape=(1,self.buffer_size+1))
            self.buffer_position = 1 # restart buffer after reference
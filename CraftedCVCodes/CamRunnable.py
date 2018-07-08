#This module will contain all that is needed to operate the cameras. It will evolve as more cameras are supported.

from threading import Thread
import cv2
import datetime
from time import sleep
class camVideoStream:
    def __init__(self,src,fps,width,height):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.frameNum = 0
        self.timeStart = datetime.datetime.now()
        self.frameTime = 0
        sleep(0.1)
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()
            self.frameTime = (datetime.datetime.now() - self.timeStart).total_seconds()
            self.frameNum += 1
    def read(self):
        return self.frame, self.frameTime
    
    def read_bytes(self):
        ret, jpeg = cv2.imencode('.jpg', self.frame)
        return jpeg.tobytes()
    
    def stop(self):
        self.stopped = True
        self.stream.release()
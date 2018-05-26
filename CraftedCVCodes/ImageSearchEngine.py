import cv2
import numpy as np
import _pickle as cPickle
import argparse
import glob

#I will be making an image classifier out of this code, by using 3D histogram information as my image descriptor
#More powerful descriptors can be used. However, the point of this tutorial is to go through the process of creating the engine
# The engine has 4 Parts: 1.Descriptor definition 2. Indexing of dataset 3. Similarity metric definition 4. Search

# 1. Descriptor definition. This is my historgram maker
class RGBHistogram: 
    def __init__(self,bins):
        self.bins = bins # Will accept [#, #, #] as input, as the historgram is 3D in nature
    def genHist(self,image): #This method generates smashed 3D histogram of an image
        hist = cv2.calcHist([image],[0,1,2],None,self.bins,[0,256,0,256,0,256])
        hist = cv2.normalize(hist)
        return hist.flatten() # Segment which takes the 3 channels of the histogram and shmashes them together
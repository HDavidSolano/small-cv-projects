import numpy as np
import cv2
import matplotlib.pyplot as plt
#Este archivo explora histogramas 1D y 2D para su uso general
image = cv2.imread('F:\\Flights\\Hatillo\\Hatillo\\geotagged_Offset\\DSC04059_geotag.JPG')

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
bw_hist = cv2.calcHist([gray],[0],None,[256],[0,256])
channels = cv2.split(image)

plt.figure(1)
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(bw_hist)
plt.xlim([0, 256])

plt.figure(2)
colors = ('b','g','r')
features = []
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
for (channel,acolor) in zip(channels,colors):
    histogram_band = cv2.calcHist([channel],[0],None,[256],[0,256])
    features.extend(histogram_band)
    plt.plot(histogram_band, color = acolor)
    plt.xlim([0, 256])

mix_histogram = cv2.calcHist([channels[0],channels[1]],[0,1],None,[32,32],[0,256,0,256])
plt.figure(3)
p = plt.imshow(mix_histogram,interpolation="nearest")
plt.title("Dual histogram")
plt.xlabel("Blue")
plt.ylabel("Green")
plt.colorbar(p)

plt.show()
image_thumb = cv2.resize(image, (0,0),fx = 0.25, fy = 0.25)
cv2.imshow('image',image_thumb )

key = cv2.waitKey(0)
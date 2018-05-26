import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('E:\\Flight_2_09_2015\\FightVsual\\IMG_0033.JPG')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Los colores me la pelan

hist,bins = np.histogram(img.flatten(),256,[0,256]) # Calculo de histograma de la imagen en grises con 256 bins

cdf = hist.cumsum()                                 # Calculo del cdf del histograma creado anteriormente
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.figure(1)
h1 = plt.subplot(311)
h1.set_title("Hist Oirginal")
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')


cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]

hist,bins = np.histogram(img2.flatten(),256,[0,256]) # Calculo de histograma de la imagen transformada con 256 bins
cdf = hist.cumsum()                                  # Calculo del cdf del histograma creado anteriormente
cdf_normalized = cdf * hist.max()/ cdf.max()
h2 = plt.subplot(312)
h2.set_title("Hist Igualado")
plt.plot(cdf_normalized, color = 'b')
plt.hist(img2.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')


clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
img3  = clahe.apply(img)

hist,bins = np.histogram(img3.flatten(),256,[0,256]) # Calculo de histograma de la imagen transformada mas cuidadosamente con 256 bins
cdf = hist.cumsum()                                  # Calculo del cdf del histograma creado anteriormente
cdf_normalized = cdf * hist.max()/ cdf.max()
h3 = plt.subplot(313)
h3.set_title("Hist CLHE")
plt.plot(cdf_normalized, color = 'b')
plt.hist(img3.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')


plt.figure(2)
ax = plt.subplot(311)
ax.set_title("original")
plt.imshow(img, cmap='gray')
bx = plt.subplot(312)
bx.set_title("Normalizado")
plt.imshow(img2, cmap='gray')
cx = plt.subplot(313)
cx.set_title("Super Normalizado", fontsize=16)
plt.imshow(img3, cmap='gray')
plt.show()
img3 = cv2.resize(img3, (0,0), fx=0.25, fy=0.25) 
cv2.imshow('CLAHE',img3)
img = cv2.resize(img, (0,0), fx=0.25, fy=0.25) 
cv2.imshow('normal',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

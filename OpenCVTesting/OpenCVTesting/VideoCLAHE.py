import cv2
from time import sleep
import numpy as np
import plotter

buffer_size = 100
laplac_norm = np.zeros(shape=(buffer_size,2))
laplac_even = np.zeros(shape=(buffer_size,2))
laplac_CLAHE = np.zeros(shape=(buffer_size,2))
cap = cv2.VideoCapture(0)
frame_q = 0 # Con esto vamos a contar los frames

laplace_capture_skip = 2  #Cada cuantos frames vamos a capturar el laplacian para estudiar
plot_counter = 0
ma_plot = plotter.plotter(200,200,50,50,2)
while True:
    ret, frame = cap.read()
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB) # Convirtiendo la imagen de color a LAB, donde el brillo es separable
    l, a, b = cv2.split(lab_frame) # valos a usar la 'l' como nuestra imagen de blanco y negro
    
    laplacian_normal = cv2.Laplacian(l, cv2.CV_64F).var()
    #--------------- Usar emparejamiento histograma GENERAL----------------------------
    hist,bins = np.histogram(l.flatten(),256,[0,256]) # Calculo de histograma de la imagen en grises con 256 bins
    cdf = hist.cumsum()                                 # Calculo del cdf del histograma creado anteriormente
    cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    even_l = cdf[l]
    frame_even = cv2.merge((even_l,a,b))
    frame_even_f = cv2.cvtColor(frame_even, cv2.COLOR_LAB2BGR)
    laplacian_even = cv2.Laplacian(even_l, cv2.CV_64F).var()
    #--------------- Usar emparejamiento histograma CLAHE----------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    frame_CLAHE = cv2.merge((cl,a,b))
    frame_CLAHE_f = cv2.cvtColor(frame_CLAHE, cv2.COLOR_LAB2BGR)
    laplacian_CLAHE = cv2.Laplacian(cl, cv2.CV_64F).var()
    if frame_q%laplace_capture_skip == 0: # Every now and then capture a new reference image
        laplac_norm[plot_counter] = [frame_q,laplacian_normal]
        laplac_even[plot_counter] = [frame_q,laplacian_even]
        laplac_CLAHE[plot_counter] = [frame_q,laplacian_CLAHE]
        bound_frame = [np.min(laplac_norm[:,0]),np.max(laplac_norm[:,0])+1]
        bound_norm_y = [np.min([np.min(laplac_norm[:,1]),np.min(laplac_even[:,1]),np.min(laplac_CLAHE[:,1])]),np.max([np.max(laplac_norm[:,1]),np.max(laplac_even[:,1]),np.max(laplac_CLAHE[:,1])])]
        composite = [laplac_norm,laplac_even,laplac_CLAHE]
        
        if plot_counter == 99:
            plot_counter = 0
        else:
            plot_counter += 1
    ma_plot.plot_information(frame,composite,bound_frame,bound_norm_y)
    ma_string = 'variance '+str(laplacian_normal)
    cv2.putText(frame,ma_string, (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    ma_string = 'variance '+str(laplacian_even)
    cv2.putText(frame_even_f,ma_string, (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    ma_string = 'variance '+str(laplacian_CLAHE)
    cv2.putText(frame_CLAHE_f,ma_string, (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

    cv2.imshow('original',frame)
    cv2.imshow('histograma igualado',frame_even_f)
    cv2.imshow('histograma CLAHE',frame_CLAHE_f)
    frame_q = frame_q + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows
"""fig, ax = plt.subplots()
ax.plot(laplac_norm, color = 'b',label='normal')
ax.plot(laplac_even, color = 'r',label='even')
ax.plot(laplac_CLAHE, color = 'g',label='CLAHE')
plt.xlabel('frame #')
plt.ylabel('Laplacian Variance')
legend = ax.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.show()"""


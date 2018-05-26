import sys
import os
import csv
import math as m
from datetime import datetime
import piexif
from PIL import Image
from os.path import expanduser
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog


def decToDMS(the_location):  #Cambia los grados decimales a minutos y segundos
    numerical_loc = m.fabs(float(the_location)) #Recibe texto como valor
    DMS = []
    DMS.append(m.floor(numerical_loc)) #We read the degrees
    DMS.append(m.floor(60*(numerical_loc-m.floor(numerical_loc)))) #We read minutes
    DMS.append(60*(60*(numerical_loc-m.floor(numerical_loc))-m.floor(60*(numerical_loc-m.floor(numerical_loc)))))
    return DMS
def CAM_Writer(name,path,cam_times,gaps,indexes,base,header,Attitudes,Altitudes,Camera_Labels,if_reative,base_alt,if_gimbal): #Esta function va a escribir el archivo
    download_dir = os.path.join(path, name+'.log') #path+'\\'+name+'.log' #where you want the file to be downloaded to 
    csv = open(download_dir, "w") 
    location_constructor = [] # Lo que va a escribir ese jugoso location csv file que Agisoft le encanta, con atitudes y todo (delay incluido)
    # Para el log de mission planner, se necesita un header
    for a_row in header:
        for every_string in a_row:
            if every_string != a_row[-1]:
                csv.write(every_string+',')
            else:
                csv.write(every_string)
        csv.write('\n')    
    # Aca comienza lo bacano
    i = 0
    for an_index in indexes:
        gap = m.fabs(cam_times[i]-int(base[an_index][1]))
        if m.fabs(gap-gaps[i]) < 0.001:
            if base[an_index][0] == 'GPS' or base[an_index][0] == ' GPS':
                base[an_index][0] = 'CAM'
                del base[an_index][2]
                del base[an_index][4]
                del base[an_index][4]
                base[an_index][7] = base[an_index][6]
                base[an_index].append('55') # Adiciona otra dimension a la fila GPS, debido a que esta se queda corta una
            if if_reative == 1: # Si se tiene la altura del sitio de depsegue, esta se usa + la del barometro para generar una altura muy buena
                use_altitude = Altitudes[i]+base_alt
                base[an_index][6] = str(use_altitude) # Escribo al CAM a TODAS las altitudes la altitud calculada
                base[an_index][7] = str(use_altitude)
                base[an_index][8] = str(use_altitude)
            else: #Si no, pues se usa la altitud GPS (la cual estaba en la columna 7 de la fila GPS
                base[an_index][7] = base[an_index][6]
                base[an_index][8] = base[an_index][6]
            if if_gimbal == 1:
                location_constructor.append([Camera_Labels[i],base[an_index][4],base[an_index][5],base[an_index][6],str(Attitudes[i][2]),'0.0','0'])
            else:
                location_constructor.append([Camera_Labels[i],base[an_index][4],base[an_index][5],base[an_index][6],str(Attitudes[i][2]),str(Attitudes[i][1]),str(Attitudes[i][0])])
            if if_gimbal == 1:
                base[an_index][9]  = '0'
                base[an_index][10] = '0.0'
                base[an_index][11] = str(Attitudes[i][2])
            else:
                base[an_index][9]  = str(Attitudes[i][0])
                base[an_index][10] = str(Attitudes[i][1])
                base[an_index][11] = str(Attitudes[i][2])
            for every_string in base[an_index]:
                if every_string != base[an_index][-1]:
                    csv.write(every_string+',')
                else:
                    csv.write(every_string)
            csv.write('\n')
        else:
            print('Error on index'+str(an_index))
        i = i + 1
    download_dir = os.path.join(path, name+'_location.csv') #path+'\\'+name+'.log' #where you want the file to be downloaded to 
    csv = open(download_dir, "w") 
    for a_line in location_constructor:
        for every_string in a_line:
            if every_string != a_line[-1]:
                csv.write(every_string+',')
            else:
                csv.write(every_string)
        csv.write('\n')
def Image_Writer(name,img_ref,path,cam_times,gaps,indexes,base,altitudes,usar_barometro,altura_base): #Esta funcion geotaguea las fotos. NO USA ACTITUDES!!!!! SOLO ALTITUD
    newpath = os.path.join(path,name)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    exif_dict = piexif.load(img_ref) # aca es donde cargamos la imagen para usar de referencia para replicar el EXIF
    ifd = "GPS" # Buscamos aquella etiqueta que nos conduzca al GPS
    for tag in exif_dict[ifd]: # primero velos los indices en donde el exif almacena la informacion de interes
        if piexif.TAGS[ifd][tag]["name"] == 'GPSLatitude':
            tag_latitude = tag
        if piexif.TAGS[ifd][tag]["name"] == 'GPSLongitude':
            tag_longitude = tag
        if piexif.TAGS[ifd][tag]["name"] == 'GPSAltitude':
            tag_altitude = tag
        print(tag, piexif.TAGS[ifd][tag]["name"], exif_dict[ifd][tag])
        
    directory = os.listdir(path) # Aca cargamos todas las imagenes
    img_paths = []
    img_labels = []
    for files in directory:
        if files.endswith (('jpg','JPG','png','PNG','tiff','TIFF')): # Va a soportar hasta su hermana
            file_path = os.path.join(path, files)
            #print(file_path)
            #img = Image.open(file_path)
            img_paths.append(file_path)
            img_labels.append(files)
    #Aca comienza a calculiar
    i = 0
    for an_index in indexes:
        gap = m.fabs(cam_times[i]-int(base[an_index][1]))
        if m.fabs(gap-gaps[i]) < 0.001:
            if base[an_index][0] == 'GPS' or base[an_index][0] == ' GPS':
                base[an_index][0] = 'CAM'
                del base[an_index][2]
                del base[an_index][4]
                del base[an_index][4]
                base[an_index][7] = base[an_index][6]
                base[an_index].append('55')
            my_lat = decToDMS(base[an_index][4])
            my_long = decToDMS(base[an_index][5])
            if usar_barometro == 1: #si queremos usar la altura relativa
                my_alt = str(altura_base+altitudes[i]) #Aca sobreescribimos las alturas utilizadas. Olvidense de las actitudes
            else:
                my_alt = base[an_index][6]
            out_lat = []        #Escribiendo las tuplas de GPS
            out_long = []
            out_lat.append(int(my_lat[0]*exif_dict[ifd][tag_latitude][0][1]))
            out_lat.append(int(my_lat[1]*exif_dict[ifd][tag_latitude][1][1]))
            out_lat.append(int(my_lat[2]*exif_dict[ifd][tag_latitude][2][1]))
            tup_lat = ((out_lat[0],exif_dict[ifd][tag_latitude][0][1]),(out_lat[1],exif_dict[ifd][tag_latitude][1][1]),(out_lat[2],exif_dict[ifd][tag_latitude][2][1]))
            exif_dict[ifd][tag_latitude] = tup_lat
            out_long.append(int(my_long[0]*exif_dict[ifd][tag_longitude][0][1]))
            out_long.append(int(my_long[1]*exif_dict[ifd][tag_longitude][1][1]))
            out_long.append(int(my_long[2]*exif_dict[ifd][tag_longitude][2][1]))
            tup_long = ((out_long[0],exif_dict[ifd][tag_longitude][0][1]),(out_long[1],exif_dict[ifd][tag_longitude][1][1]),(out_long[2],exif_dict[ifd][tag_longitude][2][1]))
            exif_dict[ifd][tag_longitude] = tup_long
            out_alt = int(float(my_alt)*exif_dict[ifd][tag_altitude][1])
            tup_alt = (out_alt,exif_dict[ifd][tag_altitude][1])
            exif_dict[ifd][tag_altitude] = tup_alt
            
            img = Image.open(img_paths[i])
            exif_bytes = piexif.dump(exif_dict)
            img.save(newpath+'\\'+img_labels[i], "jpeg", exif=exif_bytes, quality=95)
        else:
            print('Error on index'+str(an_index))
        i = i + 1

def geotagger(dircont,log_file,write_img_Offset,write_img_Absolute,Trigger_Offset,Reference_Image,DoRelative,Base_Altitude):
    DoGimbal = 0
    print('Direccion entrada es '+dircont)
    print('Nombre del log: '+log_file)
    print('Delay establecido es '+ str(Trigger_Offset)+'segundos')
    cam_date = datetime(2010, 1, 4, 0, 0) # En caso tal la camara este muy desfasada en tiempo
    cam_reference = 1 # 1 usa el primer mensaje CAM, 2 usa el ultimo (no)
    CamTimes = []
    directory = os.listdir(dircont) # Busca la lista de imagenes
    img_labels = [] # Los nombres de las camaras son usadas por la lista esa que usa agisoft
    for files in directory:
        if files.endswith (('jpg','JPG','png','PNG','tiff','TIFF')): # En papel, soporta hasta tu mae
            file_path = os.path.join(dircont, files)
            print(file_path)
            img = Image.open(file_path)
            img_labels.append(files)
            exif_data = img._getexif()
            dateTaken = exif_data[36867] # Pueden creer que ese numero guarro guarda el tiempo tomado??? Brutal
            photo_date = datetime.strptime(dateTaken, '%Y:%m:%d %H:%M:%S')
            delta_time = (photo_date-cam_date).total_seconds()
            CamTimes.append(delta_time)
    time_index = 0
    index_of_param = 0
    plotear = 0
    full_reader = []
    CAM_reader = []
    BARO_reader = [] # Para registrar la altura relativa
    ATT_reader = [] # Para registrar las actitudes del aeronave (super util para aviones)
    GPS_reader = [] # Para usar solo GPS en caso de que hayan delays (mucho mejor)
    header = []
    with open(log_file, newline='\n') as csvfile: #Reciclado del ploteador de log que le hice a Juanma
        myreader = csv.reader(csvfile, delimiter=',')
        for row in myreader:
            if 'FMT' == row[0] or ' FMT' == row[0] or 'PARAM' == row[0] or ' PARAM' == row[0] or 'CMD' == row[0] or 'MSG' == row[0]:
                header.append(row)
            if 'GPS' == row[0] or ' GPS' == row[0] or 'CAM' == row[0] or ' CAM' == row[0]:
                full_reader.append(row)
            if 'CAM' == row[0] or ' CAM' == row[0]:
                CAM_reader.append(row)
            if 'GPS' == row[0] or ' GPS' == row[0]:
                GPS_reader.append(row)
            if 'ATT' == row[0] or ' ATT' == row[0]:
                ATT_reader.append(row)
            if 'BARO' == row[0] or ' BARO' == row[0]:
                BARO_reader.append(row)
    #Desde esta linea comienza lo fuerte
    trimmed_CAM = [] #Para uso de la lista absoluta
    if cam_reference == 1: #Si el primer CAM es nuestra referencia de tiempo en el log:
        delta_time = CamTimes[0]*1000000-int(CAM_reader[0][1]) # Captura deiferencia en microsegundos del primer mensaje
        for a_cam_time in CamTimes:
            a_cam_time = a_cam_time*1000000 - delta_time + Trigger_Offset*1000000
            trimmed_CAM.append(a_cam_time)
        #print(trimmed_CAM)
    if cam_reference == 2: #SI se usa el ultimo mensaje CAM para el tiempo (no, tengo que validar este, porque saca tiempos negativos)
        delta_time = CamTimes[-1]*1000000-int(CAM_reader[-1][1]) 
        for a_cam_time in CamTimes:
            a_cam_time = a_cam_time*1000000 - delta_time + Trigger_Offset*1000000
            trimmed_CAM.append(a_cam_time)
        #print(trimmed_CAM)
    #La parte anterior prepara algo que ocurre luego de la primera seccion.
    #La seccion que sigue es usar los CAM directamente o indirectamente para acomodarle el offset
    print('Total CAM encontrados: '+str(len(CAM_reader))+' Total fotos: '+str(len(CamTimes)))
    if len(CAM_reader) == len(CamTimes): # Calcula los offsets de la manera facil
        print('Los mensajes coinciden. Eso hace Offset mas facil')
        basic_cam = []
        for a_cam in CAM_reader:
            a_time = int(a_cam[1])+Trigger_Offset*1000000 # Aca se adicionan los offsets
            basic_cam.append(a_time) #we record the actual time offset
    else: #Y si no coinciden, el procedimiento es mucho mas pesado (usanddo la parte anterior que viene del offset del primer CAM)
        #Primero, sacams informacion del tiempo
        print('Los mensajes NO coinciden. No importa. Revisa que el log que cargaste es el que quieres')
        cam_guessed_indexes = [] #Este metodo SOLO funciona en caso de que el tiempo entre triguereo sea mayor a 1 segundo
        rare_events = 0 #Cuenta los duplicados (si hay)
        past_CAM_index = -1 # el removedor de duplicados (normalmente super raro)
        for a_cam_time in trimmed_CAM:
            CAM_index = 0
            i = 0
            cam_gap = 5000000000000 # un numero ordinario
            for a_row in CAM_reader:
                current_gap = m.fabs(a_cam_time-Trigger_Offset*1000000-int(a_row[1])) # Hace backtracking de CADA CAM, viendo desde el primero (suficientemente rapido)
                if current_gap < cam_gap:
                    CAM_index = i
                    cam_gap = current_gap
                i = i + 1
            if CAM_index == past_CAM_index: #En caso tal de que mas de un CAM corresponda a diferentes fotos
                CAM_index = CAM_index + 1   #Se corre y yA
                rare_events = rare_events + 1 # Cuenta y reporta eventos raros
            cam_guessed_indexes.append(CAM_index)
            past_CAM_index = CAM_index
        print("CAM Pairing was successful. Got "+str(rare_events)+" duplicates")
        basic_cam = [] # Aca sigue una dinamica similar al segmento perfecto, ya que tenemos los CAM correctamente asignados
        for a_cam_index in cam_guessed_indexes:
            a_time = int(CAM_reader[a_cam_index][1])+Trigger_Offset*1000000 # Ahora adicionamos los offset a los nuevos CAM, para emparejarlos finalmente con las lecturas del log
            basic_cam.append(a_time) # Aca se guarda el tiempo real de la foto (mas el shutter)
    Perfect_Indexes = [] # Aca se guardaran los indices que usaremos
    cam_gaps_ex = []
    matched_mes = []
    Attitude_offset = [] # Aca guardamos la actitud (Roll, Pitch y Yaw) de los tiempos que son
    Altitude_offset = [] # La misma vaina pero con la altura barometrica (relativa)
    for a_cam in basic_cam: #Aca encontramos los mensajes GPS, BARO, y ATT que nos sirven con los tiempos obtenidos. De esa manera obtendremos la actitud REAL, no retrazada
        CAM_index = 0
        i = 0
        cam_gap = 5000000000
        for a_row in full_reader: # Buscando en todos los mensajes GPS:
            current_gap = m.fabs(a_cam-int(a_row[1]))
            if current_gap < cam_gap:
                CAM_index = i
                cam_gap = current_gap
            i = i + 1
        cam_gaps_ex.append(cam_gap)
        Perfect_Indexes.append(CAM_index)
        matched_mes.append(full_reader[CAM_index][0])
        # En la actiitud y altitud, yo uso mejor listas directas (sin indices), ya que no voy a reciclar ningun formato
    for a_cam in basic_cam: # la misma vaina pero con la Actitud
        cam_gap = 5000000000
        for a_row in ATT_reader: 
            current_gap = m.fabs(a_cam-int(a_row[1]))
            if current_gap < cam_gap:
                Likely_attitude = [float(a_row[3]),float(a_row[5]),float(a_row[7])]
                cam_gap = current_gap
        Attitude_offset.append(Likely_attitude)
    for a_cam in basic_cam: # la misma vaina pero con la altitud
        cam_gap = 5000000000
        for a_row in BARO_reader: 
            current_gap = m.fabs(a_cam-int(a_row[1]))
            if current_gap < cam_gap:
                Likely_height = float(a_row[2])
                cam_gap = current_gap
        Altitude_offset.append(Likely_height)
    CAM_Writer('Test_Offest',dircont,basic_cam,cam_gaps_ex,Perfect_Indexes,full_reader,header,Attitude_offset,Altitude_offset,img_labels,DoRelative,Base_Altitude,DoGimbal)
    if write_img_Offset: #IEn caso que uses los archivos para agisoft y no quieras las fotos geotagueadas directamente (demora de fresa)
        Image_Writer('Test_Offset',Reference_Image,dircont,basic_cam,cam_gaps_ex,Perfect_Indexes,full_reader,Altitude_offset,DoRelative,Base_Altitude)
    print('CAM message pairing is done. Onto Absolute pairing...')
    #Ya hicimos la extraccion de la informacion provehida de los CAM. Ahora venimos al emparejamiento absoluto, el cual es mas sencillo

    Timed_Indexes = []
    cam_gaps = []
    matched_mes = []
    Attitude_absolute = []
    Altitude_absolute = []
    for a_cam_time in trimmed_CAM:
        CAM_index = 0
        i = 0
        cam_gap = 5000000000000 # la misma monda pero con los tiempos de las camaras
        for a_row in GPS_reader:
            current_gap = m.fabs(a_cam_time-int(a_row[1]))
            if current_gap < cam_gap:
                CAM_index = i
                cam_gap = current_gap
            i = i + 1
        cam_gaps.append(cam_gap)
        Timed_Indexes.append(CAM_index)
        matched_mes.append(GPS_reader[CAM_index][0])
    for a_cam_time in trimmed_CAM:
        cam_gap = 5000000000000
        for a_row in ATT_reader:
            current_gap = m.fabs(a_cam_time-int(a_row[1]))
            if current_gap < cam_gap:
                Likely_attitude = [float(a_row[3]),float(a_row[5]),float(a_row[7])]
                cam_gap = current_gap
        Attitude_absolute.append(Likely_attitude)
    for a_cam_time in trimmed_CAM:
        cam_gap = 5000000000000
        for a_row in BARO_reader:
            current_gap = m.fabs(a_cam_time-int(a_row[1]))
            if current_gap < cam_gap:
                Likely_height = float(a_row[2])
                cam_gap = current_gap
        Altitude_absolute.append(Likely_height)
    #print(Timed_Indexes)
    CAM_Writer('Test_Absolute',dircont,trimmed_CAM,cam_gaps,Timed_Indexes,GPS_reader,header,Attitude_absolute,Altitude_absolute,img_labels,DoRelative,Base_Altitude,DoGimbal)  
    if write_img_Absolute: #Exactamente la misma vaina
        Image_Writer('Test_Absolute',Reference_Image,dircont,trimmed_CAM,cam_gaps,Timed_Indexes,GPS_reader,Altitude_absolute,DoRelative,Base_Altitude)
    print('Absolute pairing is done')
class mainProgram(QtWidgets.QWidget):
    def __init__ (self):
        super().__init__()
        self.init_ui()
        self.photo_folder = ""
        self.log_directory = ""
        self.photo_directory = ""
        self.trigger_relative = True
        self.trigger_absolute = False
        self.trigger_baro = 0
    def init_ui(self):
        #------Elements on Window--------------
        self.grid = QtWidgets.QGridLayout()
        #Direccion de fotos tomadas
        self.groupBox_photdir = QtWidgets.QGroupBox("Primero busca la direccion de las fotos")

        self.l2 = QtWidgets.QLabel('Por Black Square')
        world_pixmap = QtGui.QPixmap('worl.jpg')
        world_pixmap = world_pixmap.scaledToWidth(80)
        self.l2.setPixmap(world_pixmap)
        self.button = QtWidgets.QPushButton('Buscar fotos')
        self.l3 = QtWidgets.QLabel('Direccion de Fotos                                              ')
        self.hbox = QtWidgets.QHBoxLayout()
        self.hbox.addWidget(self.l2)
        self.hbox.addWidget(self.button)
        self.hbox.addWidget(self.l3)
        self.groupBox_photdir.setLayout(self.hbox)
        #Direccion del log de la mision
        self.groupBox_logdir = QtWidgets.QGroupBox("Luego busca Log")

        self.button2 = QtWidgets.QPushButton('Buscar log')
        self.l5 = QtWidgets.QLabel('Direccion del Log                                               ')
        self.hbox2 = QtWidgets.QHBoxLayout()
        self.hbox2.addWidget(self.button2)
        self.hbox2.addWidget(self.l5)
        self.groupBox_logdir.setLayout(self.hbox2)

        #Direccion deuna foto previamente etiquetada
        self.groupBox_photodir = QtWidgets.QGroupBox("Luego busca una foto previamente etiquetada")

        self.button3 = QtWidgets.QPushButton('Buscar foto')
        self.l7 = QtWidgets.QLabel('Direccion de la foto previamente geoetiquetada                   ')
        self.hbox3 = QtWidgets.QHBoxLayout()
        self.hbox3.addWidget(self.button3)
        self.hbox3.addWidget(self.l7)
        self.groupBox_photodir.setLayout(self.hbox3)

        # Opcion de geotiquetar o no en relativo
        self.groupBox_rel = QtWidgets.QGroupBox("Geoetiquetar o no en modo Relativo")

        self.rb11 = QtWidgets.QRadioButton('Si')
        self.rb12 = QtWidgets.QRadioButton('No')
        self.rb11.setChecked(True)
        self.l8 = QtWidgets.QLabel('1')
        self.hbox4 = QtWidgets.QHBoxLayout()
        self.hbox4.addWidget(self.rb11)
        self.hbox4.addWidget(self.rb12)
        self.hbox4.addWidget(self.l8)
        self.groupBox_rel.setLayout(self.hbox4)

        # Opcion de geotiquetar o no en absoluto
        self.groupBox_abs = QtWidgets.QGroupBox("Geoetiquetar o no en modo Absoluto")

        self.rb21 = QtWidgets.QRadioButton('Si ')
        self.rb22 = QtWidgets.QRadioButton('No ')
        self.rb22.setChecked(True)
        self.l9 = QtWidgets.QLabel('0')
        self.hbox5 = QtWidgets.QHBoxLayout()
        self.hbox5.addWidget(self.rb21)
        self.hbox5.addWidget(self.rb22)
        self.hbox5.addWidget(self.l9)
        self.groupBox_abs.setLayout(self.hbox5)

        # Opcion de GPS o Barometro
        self.gpsBox_abs = QtWidgets.QGroupBox("Altura de GPS o de Barometro")

        self.rb31 = QtWidgets.QRadioButton('GPS')
        self.rb31.setChecked(True)
        self.rb32 = QtWidgets.QRadioButton('Barometro')
        self.l10 = QtWidgets.QLabel('GPS')
        self.hbox6 = QtWidgets.QHBoxLayout()
        self.hbox6.addWidget(self.rb31)
        self.hbox6.addWidget(self.rb32)
        self.hbox6.addWidget(self.l10)
        self.gpsBox_abs.setLayout(self.hbox6)
        
        # Parametros Numericos a especificar
        self.misc_params = QtWidgets.QGroupBox("Parametros Numericos")
        self.l11 = QtWidgets.QLabel('Shutter lag (seg):')
        self.shutspeed = QLineEdit('0.25')
        self.l12 = QtWidgets.QLabel('Altitud Sitio Despegue (msnm):')
        self.altitude = QLineEdit('450')
        self.hbox7 = QtWidgets.QHBoxLayout()
        self.hbox7.addWidget(self.l11)
        self.hbox7.addWidget(self.shutspeed)
        self.hbox7.addWidget(self.l12)
        self.hbox7.addWidget(self.altitude)
        self.misc_params.setLayout(self.hbox7)

        #Boton de accion del programa
        self.action_button = QtWidgets.QGroupBox("Ejecutar Geoetiquetado")

        self.button4 = QtWidgets.QPushButton('Ejecutar')
        self.hbox8 = QtWidgets.QHBoxLayout()
        self.hbox8.addWidget(self.button4)
        self.action_button.setLayout(self.hbox8)
        # Agregar los grupos a la gui
        self.grid.addWidget(self.groupBox_photdir, 0, 0)
        self.grid.addWidget(self.groupBox_logdir, 1, 0)
        self.grid.addWidget(self.groupBox_photodir, 2, 0)
        self.grid.addWidget(self.groupBox_rel, 3, 0)
        self.grid.addWidget(self.groupBox_abs, 4, 0)
        self.grid.addWidget(self.gpsBox_abs, 5, 0)
        self.grid.addWidget(self.misc_params, 6, 0)
        self.grid.addWidget(self.action_button, 7, 0)
        self.setLayout(self.grid)
        
        # Activar los botones
        self.rb11.toggled.connect(self.relative_switch)
        self.rb12.toggled.connect(self.relative_switch)
        self.rb21.toggled.connect(self.relative_switch)
        self.rb22.toggled.connect(self.relative_switch)
        self.rb31.toggled.connect(self.relative_switch)
        self.rb32.toggled.connect(self.relative_switch)
        #------Window Properties---------------
        self.setWindowTitle('Geoetiquetador de Fotos')
        self.button.clicked.connect(self.fotos_dir)  # Trigger del boton de busqueda de la direccion de las fotos
        self.button2.clicked.connect(self.log_dir)  # Trigger del boton de busqueda de la direccion de las fotos
        self.button3.clicked.connect(self.photo_dir)  # Trigger del boton de busqueda de la direccion de las fotos
        self.button4.clicked.connect(self.exec_main)  # Trigger del boton de busqueda de la direccion de las fotos
        self.setGeometry(100, 100, 600, 300) # Where x, where y, width, height
        self.show()
    def fotos_dir(self):
        folder_name = QFileDialog.getExistingDirectory(self,"Open a folder",expanduser("~"),QFileDialog.ShowDirsOnly)
        if folder_name:
            self.photo_folder = folder_name
            self.button.setText('Fotos en:')
            self.l3.setText(folder_name) # Cambiar para que muestre la direccion
    def log_dir(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Buscar Archivo de Log", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.log_directory = fileName
            self.button2.setText('Log en:')
            self.l5.setText(fileName)
    def photo_dir(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Buscar Foto previamente Etiquetada", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.photo_directory = fileName
            self.button3.setText('Foto en:')
            self.l7.setText(fileName)
    def exec_main(self):
        geotagger(self.photo_folder,self.log_directory,self.trigger_relative,self.trigger_absolute,float(self.shutspeed.text()),self.photo_directory,self.trigger_baro,float(self.altitude.text()))
    def relative_switch(self):
        b = self.sender()
        if b.text() == "Si":
            if b.isChecked() == True:
                self.trigger_relative = True
                self.l8.setText('1')
        if b.text() == "No":
            if b.isChecked() == True:
                self.trigger_relative = False
                self.l8.setText('0')
        if b.text() == "Si ":
            if b.isChecked() == True:
                self.trigger_absolute = True
                self.l9.setText('1')
        if b.text() == "No ":
            if b.isChecked() == True:
                self.trigger_absolute = False
                self.l9.setText('0')
        if b.text() == "GPS":
            if b.isChecked() == True:
                self.trigger_baro = 0
                self.l10.setText('GPS')
        if b.text() == "Barometro":
            if b.isChecked() == True:
                self.trigger_baro = 1
                self.l10.setText('Barometro')
app = QtWidgets.QApplication(sys.argv)
ma_window = mainProgram()
sys.exit(app.exec_())
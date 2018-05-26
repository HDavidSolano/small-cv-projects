import cv2
import numpy as np

class plotter:
    def __init__(self,width,height,x_pos,y_pos,thickness):
        self.width = width              # width of actual graph in pixels
        self.height = height            # height of actual graph in pixels
        self.x_pos = x_pos              # position in x of figure of plot's upper left corner
        self.y_pos = y_pos              # position in x of figure of plot's upper left corner
        self.offset = 20
        self.thickness = thickness
        self.color_table = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(0,255,255)]
    def plot_information(self,fig_handle,data_stream,x_scale,y_scale):
        self.x_scale = x_scale          # x scale used in graph [min,max,subdiv]
        self.y_scale = y_scale          # y scale used in graph [min,max,subdiv]
        # First, lets make the axis
        cv2.arrowedLine(fig_handle,(self.x_pos+self.offset,self.y_pos+self.height-self.offset),(self.x_pos+self.width-self.offset,self.y_pos+self.height-self.offset),(0,0,0),thickness=2)
        cv2.arrowedLine(fig_handle,(self.x_pos+self.offset,self.y_pos+self.height-self.offset),(self.x_pos+self.offset,self.y_pos+self.offset),(0,0,0),thickness=2)
        cv2.putText(fig_handle,str('%d'%self.x_scale[0]),(self.x_pos+int(self.width/15),self.y_pos+self.height), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(fig_handle,str('%d'%self.x_scale[1]),(self.x_pos+self.width-int(self.width/20),self.y_pos+self.height), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(fig_handle,str('%d'%self.y_scale[0]),(self.x_pos-self.offset,self.y_pos+self.height), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(fig_handle,str('%d'%self.y_scale[1]),(self.x_pos,self.y_pos+self.offset), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        i = 0
        for a_stream in data_stream:    #For each stream of data handled by the plotter
            stream_size = len(a_stream) #get total length of stream provided, a numpy vector whose columns are the points
            j = 0
            x_pos = np.zeros(shape = stream_size)
            y_pos = np.zeros(shape = stream_size)
            for a_point in a_stream:
                if a_point[0] >= self.x_scale[0] and a_point[0] <= self.x_scale[1] and a_point[1] >= self.y_scale[0] and a_point[1] <= self.y_scale[1]: #if it fits in the plot
                    x_pos[j] = (self.x_pos+self.offset+(self.width-2*self.offset)*((a_point[0]-self.x_scale[0])/(self.x_scale[1]-self.x_scale[0])))
                    y_pos[j]= (self.y_pos+self.height-self.offset-(self.height-2*self.offset)*((a_point[1]-self.y_scale[0])/(self.y_scale[1]-self.y_scale[0])))
                    cv2.rectangle(fig_handle,(int(x_pos[j]),int(y_pos[j])),(int(x_pos[j])+self.thickness,int(y_pos[j])+self.thickness),self.color_table[i],self.thickness)
                    if j > 0:
                        cv2.line(fig_handle,(int(x_pos[j-1]),int(y_pos[j-1])),(int(x_pos[j]),int(y_pos[j])),self.color_table[i],self.thickness-1)
                    j += 1
            i += 1


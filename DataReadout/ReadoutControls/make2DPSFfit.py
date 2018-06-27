#TODO break out into a helper repository

import numpy as np
from matplotlib import pyplot as plt
from astropy.modeling import models, fitting
import astropy.modeling
from lib.utils import interpolateImage
import os,commands


def fit2DMoffat(image,guess):
    p_init=astropy.modeling.functional_models.Moffat2D(**guess)
    fit_p = fitting.LevMarLSQFitter()
    y,x=np.mgrid[:len(image),:len(image[0])]
    p = fit_p(p_init, x, y, image)
    return p


def loadImg(imgFileName, badPixelMask, interpolate=False):
    image=np.fromfile(open(imageFileName, mode='rb'),dtype=np.uint16)
    image = np.transpose(np.reshape(image, (80, 125)))
    image=np.array(image,dtype=float)
    if np.logical_and(badPixelMask!=None,interpolate==True):
        image[badPixelMask>0]=0
        image=interpolateImage(image)
    elif badPixelMask!=None:
        image[badPixelMask>0]=-1
    return image


class MouseMonitor():
    def __init__(self):
        pass

    def onclick(self,event):
        #print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata))
        self.xyguess=[event.xdata,event.ydata]
        print('xguess=%f, yguess=%f'%(self.xyguess[0], self.xyguess[1]))
        return
    def connect(self):
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)     

if __name__=='__main__':
    loadStack=False
    loadimg=True


    basePath='/home/kids/giulia_copy/'
    
    if loadimg:
        baseTime=1506725171 #flat
        imgBaseTime=baseTime+50 #light on array after flat 
        #imageFileName='/mnt/ramdisk/'+'1506894052.img'#str(imgBaseTime)+'.img'
        #imageFileName='/mnt/ramdisk/1506894829.img'
        imgPath='/mnt/ramdisk/'
        fileList=commands.getoutput('ls -t '+imgPath +'| grep img')
        fileList=fileList.split('\n')
        imageFileName=os.path.join(imgPath,fileList[0])
        beammapF=basePath+'beamMapFailed_1.npz'
        beammapFailed=np.load(beammapF)['arr_0']
        
        print 'loading img', imageFileName
        image=loadImg(imageFileName,beammapFailed,interpolate=1)
    
    if loadStack:
        stackPath='/mnt/data0/CalibrationFiles/imageStacks/PAL2017b/20171001/'
        stackFileName=None
        imageStack=os.path.join(stackPath,imageStack)
        print 'loading image stack', imageStack
        image=np.load(imageStack)['arr_0']
       

    map=MouseMonitor()
    map.fig=plt.figure()
    map.ax = map.fig.add_subplot(111)                                                                                                                                                   
    map.ax.set_title('Centroid Guess')
    map.handleMatshow = map.ax.matshow(image, origin = 'lower')
    map.connect()
    
    print "Click on centroid guess"
    plt.show()
    print "Centroid initial guess [x,y]",map.xyguess
    #fig.canvas.mpl_disconnect(cid)
    
    croppingSize=raw_input("Input cropping size for fit:")

    croppingSize=np.float(croppingSize)/2
    image=image[map.xyguess[1]-croppingSize:map.xyguess[1]+croppingSize,map.xyguess[0]-croppingSize:map.xyguess[0]+croppingSize]
    
    ###alpha makes it a higher value of halo, gamma makes it fatter
    guess={}
    guess['amplitude']=np.max(image) #2000
    guess['x_0']=croppingSize   #map.xyguess[0]
    guess['y_0']=croppingSize   #map.xyguess[1]
    guess['gamma']=1
    guess['alpha']=1
    print 'guess', guess 
    
    #plt.matshow(image)
    #plt.show()
    #print [guess['y_0']-croppingSize,guess['y_0']+croppingSize,guess['x_0']-croppingSize,guess['x_0']+croppingSize] 

    p=fit2DMoffat(image,guess)
    print 'fit:',p
    y,x=np.mgrid[:len(image),:len(image[0])]
    plt.figure(figsize=(8, 2.5))
    plt.subplot(1, 3, 1)
    plt.imshow(image, origin='lower', interpolation='nearest', vmin=0, vmax=2500)
    plt.title("Data")
    plt.subplot(1, 3, 2)
    plt.imshow(p(x, y), origin='lower', interpolation='nearest', vmin=0,vmax=2500)
    plt.title("Model")
    plt.subplot(1, 3, 3)
    plt.imshow(image - p(x, y), origin='lower', interpolation='nearest', vmin=0,vmax=2500)
    plt.title("Residual")
    plt.show()

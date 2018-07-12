
# coding: utf-8

import numpy as np
import glob

def quicklook(val,twoDarr):
    for i in range(len(twoDarr)):
        if int(twoDarr[i][0]) == int(val):
            return i 
    return -1


class feedline:
    #instantiating the class
    
    def __init__(self, modelfile, beammapfile, freqsweeps, fl_num):
        beammap=np.loadtxt(beammapfile)
        model=np.loadtxt(modelfile)
        FreqSweepFiles = glob.glob(freqsweeps)
        final_map=np.zeros((146,140,5))
        for n in range(len(beammap)):
            final_map[int(beammap[n][3])][int(beammap[n][2])][0]=beammap[n][0]
            final_map[int(beammap[n][3])][int(beammap[n][2])][2]=beammap[n][1]
            final_map[int(beammap[n][3])][int(beammap[n][2])][3]=beammap[n][2]
            final_map[int(beammap[n][3])][int(beammap[n][2])][4]=beammap[n][3]
        
        good = np.copy(final_map)
        freqarr = np.loadtxt(FreqSweepFiles[0])
        for i in range(len(FreqSweepFiles)- 1):
            sweep = np.loadtxt(FreqSweepFiles[i+1])
            freqarr = np.concatenate((freqarr, sweep))
        frequencies = freqarr[:,1]
        for i in range(len(final_map)):
            for j in range(len(final_map[0])):
                ResID = good[i][j][0]
                index = quicklook(ResID, freqarr)
                if index != -1 and int(final_map[i][j][2])==0:
                    good[i][j][1] = freqarr[index][1]/(10.**6)
                else:
                    good[i][j][1] = 0
        counter = 0
        array=np.zeros((len(model),len(model[0]),6))
        for i in range(len(model)):
            for j in range(len(model[0])):
                array[i][j][:5]=good[i][(len(good[0])-14*fl_num)+j]
                if int(good[i][(len(good[0])-14*fl_num)+j][2])==0 and good[i][(len(good[0])-14*fl_num)+j][1] != 0 :
                    array[i][j][5]=good[i][(len(good[0])-14*fl_num)+j][1]
                    counter = counter+1
                else :
                    array[i][j][5]=float('NaN')
        
        #data has [ResID, freq, flag, x, y]
        self.count=counter
        self.data=array
        self.freqs=array[:,:,1]
        #finding minimim (nonzero) frequency value
        self.min=np.amin(array[:,:,1][np.nonzero(array[:,:,1])])
        self.max=np.amax(array[:,:,1])
        self.number = fl_num
      
    #methods  
    def ResID(self,freq):
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if freq == self.data[i][j][1]:
                    return self.data[i][j][0]
                else:
                    print("We do not have a resonator for that frequency")
    
    def ResID(self,x,y):
        return self.data[y][x][0]
    
    def x(self,ResID):
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if self.data[i][j][0] == ResID:
                    return self.data[i][j][2]
                else:
                    print("Invalid ResID")
    def y(self,ResID):
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if self.data[i][j][0] == ResID:
                    return self.data[i][j][3]
                else:
                    print("Invalid ResID")

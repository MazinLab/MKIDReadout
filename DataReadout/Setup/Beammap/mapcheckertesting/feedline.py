
# coding: utf-8

import numpy as np
import glob
import os


# NOTE : Update this class to reject/notify the user if they are trying to analyze a 'bad' feedline
# Figure out what constitutes 'bad'... <100,500,1000 pixels?
class Feedline:
    # instantiating the class

    def __init__(self, modelfile, beammapfile, freqsweeps, fl_num):

        model = np.loadtxt(modelfile)
        good = array_organizer(beammapfile, freqsweeps, 'Hypatia')

        counter = 0
        array = np.zeros((len(model), len(model[0]), 6))
        for i in range(len(model)):
            for j in range(len(model[0])):
                array[i][j][:5] = good[i][(len(good[0]) - 14 * fl_num) + j]
                if int(good[i][(len(good[0]) - 14 * fl_num) + j][2]) == 0 and good[i][(len(good[0]) - 14 * fl_num) + j][
                    1] != 0:
                    array[i][j][5] = good[i][(len(good[0]) - 14 * fl_num) + j][1]
                    counter = counter + 1
                else:
                    array[i][j][5] = float('NaN')

        # data has [ResID, freq (with unread pixels as 0), flag, x, y, frequencies (with unread pixels as NaN)]
        self.count = counter
        self.data = array
        self.freqs = array[:, :, 1]
        self.name = "Feedline "+str(fl_num)
        # finding minimim (nonzero) frequency value
        self.min = np.amin(array[:, :, 1][np.nonzero(array[:, :, 1])])
        self.max = np.amax(array[:, :, 1])
        self.number = fl_num
        # normfreqs shifts the zero point of the array in frequency space to 0 MHz
        self.normfreqs = array[:, :, 5] - self.min


    # methods
    def ResIDfreq(self, freq):
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if freq == self.data[i][j][1]:
                    return self.data[i][j][0]
                else:
                    print("We do not have a resonator for that frequency")

    def ResIDpos(self, x, y):
        return self.data[y][x][0]

    def x(self, ResID):
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if self.data[i][j][0] == ResID:
                    return self.data[i][j][2]
                else:
                    print("Invalid ResID")

    def y(self, ResID):
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if self.data[i][j][0] == ResID:
                    return self.data[i][j][3]
                else:
                    print("Invalid ResID")



def indexfinder(val, twoDarr):
    for i in range(len(twoDarr)):
        if int(twoDarr[i][0]) == int(val):
            return i
    return -1


def array_organizer (beammapfile, freqsweeps, devicename):

    if type(devicename) == 'str':
        device_file=devicename+'_array.npy'
    else :
        device_file=str(devicename)+'_array.npy'

    if os.path.isfile(os.path.normpath(device_file)):

        good=np.load(device_file)

        return good
    else :

        beammap=np.loadtxt(beammapfile)
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
                index = indexfinder(ResID, freqarr)
                if index != -1 and int(final_map[i][j][2])==0:
                    good[i][j][1] = freqarr[index][1]/(10.**6)
                else:
                    good[i][j][1] = 0

        np.save(device_file, good)
        return good
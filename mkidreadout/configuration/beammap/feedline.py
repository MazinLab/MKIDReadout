# coding: utf-8
import numpy as np
import glob
import os


'''NOTE : Update this class to reject/notify the user if they are trying to analyze a 'bad' feedline.
Figure out what constitutes 'bad'... <100, 500, 1000 pixels?, at least anything completely empty should return "bad"'''


class Feedline(object):

    # instantiating the class
    def __init__(self, modelfile, beammapfile, freqsweeps, fl_num):

        # Load the model feedline data (format is that at each 'coordinate' the value in the array is the design
        # frequency, measured from 0), the 'good' array is the formatted data from the full array you give it info from
        # The first time an array is analyzed is takes longer because of the indexfinder function, but then it speeds up
        # The array called 'good' is just an intermediate step for ease of formatting, it is never used directly
        model = np.loadtxt(modelfile)
        good = array_organizer(beammapfile, freqsweeps, 'Hypatia')

        # Take the data from your array (140-by-146) and create a feedline (14-by-146) from it
        # The result of this block will be a (146, 14, 6) data cube where each 'location' is a resID and the
        # third dimension is of the form [resID, frequency (0 for unmeasured pixels), flag, x (column),
        # y (row), frequency (with NaN for unmeasured pixels)]
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

        self.count = counter  # Gives the number of pixels placed on the given feedline
        self.data = array  # (146, 14, 6) array of the form above
        self.freqs = array[:, :, 1]  # (146, 14) array that has the frequencies as they were measured (0 is unmeasured)
        self.name = "Feedline "+str(fl_num)  # Gives feedline name and number as string (included to help titles plots)
        self.min = np.amin(array[:, :, 1][np.nonzero(array[:, :, 1])])  # Minimum measured frequency value (MHz)
        self.max = np.amax(array[:, :, 1])  # Maximum measured frequency value (MHz)
        self.number = fl_num  # More ID information, may be able to delete in future
        self.normfreqs = array[:, :, 5] - self.min  # Shifts the minimum measured frequency to be 0 MHz
        self.up1 = self.shiftfreqsinspace('vertical', -1)
        self.down1 = self.shiftfreqsinspace('vertical', 1)
        self.left1 = self.shiftfreqsinspace('horizontal', -1)
        self.right1 = self.shiftfreqsinspace('horizontal', 1)

    # Class methods

    # Calling this with a frequency value (originally measured) returns the resID associated
    # Consider playing with the precision necessary for this (to the tenths place?)
    def ResIDfreq(self, freq):
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if freq == self.data[i][j][1]:
                    return self.data[i][j][0]
                else:
                    print("We do not have a resonator for that frequency")

    # Giving this an x and y value returns the resID of the pixel at that location (returns 0 if none associated)
    def ResIDpos(self, x, y):
        return self.data[y][x][0]

    # Giving this an resID returns the x value (column) that the resID falls into
    def x(self, ResID):
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if self.data[i][j][0] == ResID:
                    return self.data[i][j][2]
                else:
                    print("Invalid ResID")

    # Giving this an resID returns the y value (row) that the resID falls into
    def y(self, ResID):
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if self.data[i][j][0] == ResID:
                    return self.data[i][j][3]
                else:
                    print("Invalid ResID")

    #May be unnecessary in the future, think about utility of this
    def shiftfreqsinspace(self,direction,amount):
        if direction == 'vertical' or direction == 'up' or direction == 'down':
            newfreqs = np.roll(np.copy(self.normfreqs), int(amount), axis=0)
        elif direction == 'horizontal' or direction == 'left' or direction == 'right':
            newfreqs = np.roll(np.copy(self.normfreqs), int(amount), axis=1)
        return newfreqs



# Takes in a resID and the frequency sweep data and matches a measured frequency to a resID
def indexfinder(val, twoDarr):
    for i in range(len(twoDarr)):
        if int(twoDarr[i][0]) == int(val):
            return i
    return -1


'''This function takes in all of the files for an array and creates the (146,14,5) array that is used to get feedlines
In the event an array has been analyzed before, it goes quickly and simply loads the already formatted data, otherwise
it reads in all of the data and organizes it so that we can use it properly in the Feedline class'''
def array_organizer (beammapfile, freqsweeps, devicename):

    if type(devicename) == 'str':
        device_file = devicename+'_array.npy'
    else :
        device_file = str(devicename)+'_array.npy'

    if os.path.isfile(os.path.normpath(device_file)):

        good=np.load(device_file)

        return good
    else :

        beammap = np.loadtxt(beammapfile)
        FreqSweepFiles = glob.glob(freqsweeps)
        final_map = np.zeros((146,140,5))

        for n in range(len(beammap)):
            final_map[int(beammap[n][3])][int(beammap[n][2])][0] = beammap[n][0]
            final_map[int(beammap[n][3])][int(beammap[n][2])][2] = beammap[n][1]
            final_map[int(beammap[n][3])][int(beammap[n][2])][3] = beammap[n][2]
            final_map[int(beammap[n][3])][int(beammap[n][2])][4] = beammap[n][3]

        good = np.copy(final_map)
        freqarr = np.loadtxt(FreqSweepFiles[0])
        for i in range(len(FreqSweepFiles) - 1):
            sweep = np.loadtxt(FreqSweepFiles[i+1])
            freqarr = np.concatenate((freqarr, sweep))

        # Currently this only places pixels with a 0 flag, consider revamping this to take in all pixels that were read
        # and then making the distinction between well-placed and poorly-placed pixels later
        for i in range(len(final_map)):
            for j in range(len(final_map[0])):
                ResID = good[i][j][0]
                index = indexfinder(ResID, freqarr)
                if index != -1 and int(final_map[i][j][2]) == 0:
                    good[i][j][1] = freqarr[index][1]/(10.**6)
                else:
                    good[i][j][1] = 0

        np.save(device_file, good)
        return good

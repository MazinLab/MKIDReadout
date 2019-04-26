import struct

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def interpolateImage(image, method='cubic'):
    """
    This function interpolates the dead pixels in the image

    image = [[1,1,1],[1,np.nan,5],[1,np.nan,1],[1,1,1]]
    interpImage = interpolateImage(image)

    plt.matshow(image)
    plt.matshow(interpImage)
    plt.show()

    INPUTS:
        image - 2D array. Nan values for dead pixels. 0 indicates that pixel is good and has 0 counts
        method - scipy griddata option. Can be nearest, linear, cubic.
        
    OUTPUTS:
        interpolatedImage - Same shape as input image. Dead pixels are interpolated over. 
    """
    shape = np.shape(image)
    goodPoints = np.where(np.isfinite(image))
    values = np.asarray(image)[goodPoints]
    interpPoints = (np.repeat(range(shape[0]),shape[1]), np.tile(range(shape[1]),shape[0]))

    interpValues = griddata(goodPoints, values, interpPoints, method)
    interpImage = np.reshape(interpValues, shape)
    
    return interpImage


def parsephasedump2(file):
    """
    The pgbe0 firmware sends a stream of phase data points over
    1Gbit ethernet (gbe). recv_dump_64b.c catches the frames of phase
    data and dumps them to a file phaseDump.bin. This script parses the dump file.
    Each 64 bit word is just an integer.  The beginning of each frame caught
    should contain a 64 bit header.

    File:      parsePhaseDump2.py
    Author:    Matt Strader
    Date:      Feb 19, 2016
    Firmware:  pgbe0.slx
    """

    with open(file, 'r') as dumpFile:
        data = dumpFile.read()

    nBytes = len(data)
    nWords = nBytes / 8  # 64 bit words
    # break into 64 bit words
    print 'stream length' + str(len(data))
    print 'nWords ' + str(nWords)
    print 'nBytes ' + str(nBytes)
    words = np.array(struct.unpack('>{:d}Q'.format(nWords), data[0:nWords * 8]))
    plt.plot(words[0:2 ** 15], '.-')
    plt.show()




import numpy as np
import tensorflow as tf
import os, sys, glob
import argparse
import logging
import matplotlib.pyplot as plt
import mkidreadout.configuration.sweepdata as sd
import mkidreadout.configuration.powersweep.ml.tools as mlt
from mkidcore.corelog import getLogger
from mkidreadout.configuration.powersweep.ml.wpsnn import N_CLASSES

def makeWPSMap(modelDir, freqSweep, freqStep=None, attenClip=5):
    mlDict, sess, graph, x_input, y_output, keep_prob, is_training = mlt.get_ml_model(modelDir)
    
    if freqStep is None:
        freqStep = freqSweep.freqStep

    attens = freqSweep.atten[attenClip:-attenClip]
    freqStart = freqSweep.freqs[0, 0] + freqSweep.freqStep*mlDict['freqWinSize']
    freqEnd = freqSweep.freqs[-1, -1] - freqSweep.freqStep*mlDict['freqWinSize']
    freqs = np.arange(freqStart, freqEnd, freqStep)

    wpsImage = np.zeros((len(attens), len(freqs), N_CLASSES))
    nColors = 2
    if mlDict['useIQV']:
        nColors += 1
    if mlDict['useVectIQV']:
        nColors += 2

    chunkSize = 2500
    imageList = np.zeros((chunkSize, mlDict['attenWinBelow'] + mlDict['attenWinAbove'] + 1, mlDict['freqWinSize'], nColors))
    labelsList = np.zeros((chunkSize, N_CLASSES))

    for attenInd in range(len(attens)):
        for chunkInd in range(len(freqs)/chunkSize + 1):
            nFreqsInChunk = min(chunkSize, len(freqs) - chunkSize*chunkInd)
            for i, freqInd in enumerate(range(chunkSize*chunkInd, chunkSize*chunkInd + nFreqsInChunk)):
                imageList[i], _, _  = mlt.makeWPSImage(freqSweep, freqs[freqInd], attens[attenInd], mlDict['freqWinSize'],
                        1+mlDict['attenWinBelow']+mlDict['attenWinAbove'], mlDict['useIQV'], mlDict['useVectIQV']) 
            wpsImage[attenInd, chunkSize*chunkInd:chunkSize*chunkInd + nFreqsInChunk] = sess.run(y_output, feed_dict={x_input: imageList[:nFreqsInChunk], keep_prob: 1, is_training: False})
            print 'finished chunk', chunkInd, 'out of', len(freqs)/chunkSize

        print 'atten:', attens[attenInd]


    return wpsImage

if __name__=='__main__':
    modelDir = '/home/neelay/data/20190702/wpstest0'
    freqSweepFile = '/home/neelay/data/20180108/psData_221.npz'
    freqSweep = sd.FreqSweep(freqSweepFile)
    image = makeWPSMap(modelDir, freqSweep)
    np.savez('wpsimage.npz', image=image)
    plt.imshow(image)
    plt.show()

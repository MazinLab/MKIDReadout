#/usr/bin/env python

"""
Script for using ML inference to identify and tune resonators
with wide/power sweep data (collected using widesweep.freqSweep)
"""

import numpy as np
import tensorflow as tf
from functools import partial
import multiprocessing
import os, sys, glob
import time
import copy
import argparse
import logging
#import ipdb
import matplotlib.pyplot as plt
import skimage.feature as skf
import mkidreadout.configuration.sweepdata as sd
import mkidreadout.configuration.powersweep.ml.tools as mlt
from mkidcore.corelog import getLogger
from mkidreadout.configuration.powersweep.ml.wpsnn import N_CLASSES
import mkidcore.instruments as inst

N_RES_PER_BOARD = 1024
N_CPU = 3

def makeWPSMap(modelDir, freqSweep, freqStep=None, attenClip=0):
    mlDict, sess, graph, x_input, y_output, keep_prob, is_training = mlt.get_ml_model(modelDir)
    
    if freqStep is None:
        freqStep = freqSweep.freqStep

    if attenClip > 0:
        attens = freqSweep.atten[attenClip:-attenClip]
    else:
        attens = freqSweep.atten

    freqSweep.iqvel #calculate iq vels so they are cached

    freqStart = freqSweep.freqs[0, 0] + freqSweep.freqStep*mlDict['freqWinSize']
    freqEnd = freqSweep.freqs[-1, -1] - freqSweep.freqStep*mlDict['freqWinSize']
    freqs = np.arange(freqStart, freqEnd, freqStep)

    wpsImage = np.zeros((len(attens), len(freqs), N_CLASSES))
    nColors = 2
    if mlDict['useIQV']:
        nColors += 1
    if mlDict['useVectIQV']:
        nColors += 2

    chunkSize = 5000#8000*N_CPU
    #subChunkSize = np.round(float(chunkSize)/N_CPU).astype(int)
    subChunkSize = 200
    imageList = np.zeros((chunkSize, mlDict['attenWinBelow'] + mlDict['attenWinAbove'] + 1, mlDict['freqWinSize'], nColors))
    labelsList = np.zeros((chunkSize, N_CLASSES))
    toneWinCenters = freqSweep.freqs[:, freqSweep.nlostep/2]
    if N_CPU > 1:
        pool = multiprocessing.Pool(processes=N_CPU)
        freqSweepChunk = copy.copy(freqSweep)

    for attenInd in range(len(attens)):
        tstart = time.time()
        for chunkInd in range(len(freqs)/chunkSize + 1):
            nFreqsInChunk = min(chunkSize, len(freqs) - chunkSize*chunkInd)
            nSubChunks = np.ceil(float(nFreqsInChunk)/subChunkSize).astype(int)

            if N_CPU == 1:
                for i, freqInd in enumerate(range(chunkSize*chunkInd, chunkSize*chunkInd + nFreqsInChunk)):
                    imageList[i], _, _  = mlt.makeWPSImage(freqSweep, freqs[freqInd], attens[attenInd], mlDict['freqWinSize'],
                            1+mlDict['attenWinBelow']+mlDict['attenWinAbove'], mlDict['useIQV'], mlDict['useVectIQV'],
                            normalizeBeforeCenter=mlDict['normalizeBeforeCenter']) 
            else:
                freqList = freqs[range(chunkSize*chunkInd, chunkSize*chunkInd + nFreqsInChunk)]
                toneIndLow = np.argmin(np.abs(freqList[0] - toneWinCenters))
                toneIndHigh = np.argmin(np.abs(freqList[-1] - toneWinCenters)) + 1
                freqSweepChunk.i = freqSweep.i[:, toneIndLow:toneIndHigh, :]
                freqSweepChunk.q = freqSweep.q[:, toneIndLow:toneIndHigh, :]
                freqSweepChunk._iqvel = freqSweep._iqvel[:, toneIndLow:toneIndHigh, :]
                freqSweepChunk.freqs = freqSweep.freqs[toneIndLow:toneIndHigh, :]
                freqSweepChunk.ntone = toneIndHigh - toneIndLow + 1

                freqLists = []

                for i in range(nSubChunks):
                    freqLists.append(freqList[subChunkSize*i:subChunkSize*(i + 1)])
                
                processChunk = partial(makeImageList, freqSweep=freqSweepChunk, atten=attens[attenInd], 
                            freqWinSize=mlDict['freqWinSize'], attenWinSize=1+mlDict['attenWinBelow']+mlDict['attenWinAbove'], 
                            useIQV=mlDict['useIQV'], useVectIQV=mlDict['useVectIQV'],
                            normalizeBeforeCenter=mlDict['normalizeBeforeCenter']) 

                #imageList[:nFreqsInChunk] = pool.map(processChunk, freqList, chunksize=chunkSize/N_CPU)
                imageList[:nFreqsInChunk] = np.vstack(pool.map(processChunk, freqLists, chunksize=len(freqLists)/N_CPU))

            wpsImage[attenInd, chunkSize*chunkInd:chunkSize*chunkInd + nFreqsInChunk, :N_CLASSES] = sess.run(y_output, 
                    feed_dict={x_input: imageList[:nFreqsInChunk], keep_prob: 1, is_training: False})
            print 'finished chunk', chunkInd, 'out of', len(freqs)/chunkSize

        print 'atten:', attens[attenInd]
        print ' took', time.time() - tstart, 'seconds'

    if N_CPU > 1:
        pool.close()

    tf.reset_default_graph()
    sess.close()

    return wpsImage, freqs, attens


def makeImage(centerFreq, freqSweep, atten, freqWinSize, attenWinSize, useIQV, useVectIQV, normalizeBeforeCenter):
    image, _, _, = mlt.makeWPSImage(freqSweep, centerFreq, atten, freqWinSize, attenWinSize, useIQV, useVectIQV,
            normalizeBeforeCenter=normalizeBeforeCenter) 
    return image

def makeImageList(centerFreqList, freqSweep, atten, freqWinSize, attenWinSize, useIQV, useVectIQV, normalizeBeforeCenter):
    image, _, _, = mlt.makeWPSImageList(freqSweep, centerFreqList, atten, freqWinSize, attenWinSize, useIQV, useVectIQV,
            normalizeBeforeCenter=normalizeBeforeCenter) 
    return image

def addResMagToWPSMap(wpsDict, freqSweep, outFile, freqWinSize=30, nRes=1500):
    resMags = np.zeros((wpsDict['wpsmap'].shape[0], wpsDict['wpsmap'].shape[1]))
    resCoords = skf.peak_local_max(wpsDict['wpsmap'][:,:,0], min_distance=5, threshold_abs=0.5, num_peaks=nRes, exclude_border=False)
    freqs = wpsDict['freqs']
    attens = wpsDict['attens']

    for coord in resCoords:
        _, resMag, _, _ = mlt.makeWPSImage(freqSweep, freqs[coord[1]], attens[coord[0]], freqWinSize, 5, False, False)
        resMags[coord[0], coord[1]] = resMag

    wpsDict['wpsmap'] = np.dstack((wpsDict['wpsmap'], resMags))
    np.savez(outFile, **wpsDict)

def findResonators(wpsmap, freqs, attens, prominenceThresh=0.85, peakThresh=0.97, minPeakDist=40.e3, nRes=None, attenGrad=0, resMagCut=False):
    """
    Finds resonators using ML classification map outputted by neural net inference. Returns at most nRes peaks (resonators) that
    have ML classification score above peakThresh. If resMagCut is true, then returns the nRes peaks above peakThresh with 
    the largest loop size. Else, returns the nRes peaks with the highest ML score above peakThresh.
    Parameters
    ----------
        wpsmap: numpy array
            shape is (nAttens, nFreqs, N_CLASS (+1)). Generated by makeWPSMap() function.
            Each point in image (nAttens, nFreq) is NN output for atten/freq image centered at 
            that point. Color channel contains classifier classes, with optional resMag channel
            containing the size of the loop centered at that point.

    Returns
    -------
        resFreqs
            list of resonator frequencies
        resAttens
            list of powers corresponding to resFreqs
        scores
            list of ML scores corresponding to resFreqs

    """

    minPeakDist /= np.diff(freqs)[0]
    if attenGrad > 0:
        attenBias = np.linspace(0, -(len(attens)-1)*attenGrad, len(attens))
        wpsmap = (wpsmap.T + attenBias).T
    if resMagCut:
        if nRes is None:
            raise Exception('Must specify number of resonators to use loop size cut')
        resCoords = skf.peak_local_max(wpsmap[:,:,0], min_distance=minPeakDist, threshold_abs=peakThresh, exclude_border=False)
        resCoords = prominenceCut(wpsmap, resCoords, prominenceThresh)
        resMags = np.zeros(len(resCoords))
        print 'Found', len(resCoords), 'peaks above', peakThresh
        for i in range(len(resMags)):
            resMags[i] = wpsmap[resCoords[i, 0], resCoords[i, 1], N_CLASSES]
        largestMagInds = np.argsort(resMags)[::-1]
        largestMagInds = largestMagInds[:nRes]
        print 'Res mag cutoff:', resMags[largestMagInds[-1]]
        resCoords = resCoords[largestMagInds]
        #plt.hist(resMags, bins=30)
        #plt.show()

    elif nRes is not None:
        resCoords = skf.peak_local_max(wpsmap[:,:,0], min_distance=minPeakDist, threshold_abs=peakThresh, num_peaks=nRes, exclude_border=False)
        resCoords = prominenceCut(wpsmap, resCoords, prominenceThresh)
        #if len(resCoords) < nRes:
        #    nRes += nRes - len(resCoords)
        #    print 'Running peak seearch again with', nRes, 'resonators'
        #    resCoords = skf.peak_local_max(wpsmap[:,:,0], min_distance=minPeakDist, threshold_abs=peakThresh, num_peaks=nRes, exclude_border=False)
        #    resCoords = prominenceCut(wpsmap, resCoords)

    else:
        resCoords = skf.peak_local_max(wpsmap[:,:,0], min_distance=minPeakDist, threshold_abs=peakThresh, exclude_border=False)
        resCoords = prominenceCut(wpsmap, resCoords, prominenceThresh)

    resFreqs = freqs[resCoords[:,1]]
    resAttens = attens[resCoords[:,0]]

    scores = np.zeros(len(resFreqs))
    for i in range(len(resFreqs)):
        scores[i] = wpsmap[resCoords[i,0], resCoords[i,1], 0]

    sortedInds = np.argsort(resFreqs)
    resFreqs = resFreqs[sortedInds]
    resAttens = resAttens[sortedInds]
    scores = scores[sortedInds]

    return resFreqs, resAttens, scores

def prominenceCut(wpsmap, resCoords, minThresh=0.75):
    freqSortedInds = np.argsort(resCoords[:,1])
    resCoords = resCoords[freqSortedInds]
    valleys = np.zeros(len(resCoords))
    for i in range(len(resCoords) - 1):
        attenInds = np.sort([resCoords[i, 0], resCoords[i+1, 0]])
        #image = wpsmap[attenInds[0]:attenInds[1]+1, (resCoords[i,1]+resCoords[i+1,1])/2, 0] #use all attens but freq only in middle
        #valleys[i] = np.max(image)#image[coords[0], coords[1]]
        image = wpsmap[attenInds[0]:attenInds[1]+1, resCoords[i,1]:resCoords[i+1,1]+1, 0] 
        if resCoords[i, 0] > resCoords[i+1, 0]: #resonators are always in top left or bottom right
            image = np.flipud(image)
        highestValleys = np.zeros(image.shape)
        
        for c in range(image.shape[1]):
            highestValleys[0,c] = np.min(image[0, 0:c+1])
        for r in range(image.shape[0]):
            highestValleys[r, 0] = np.min(image[0:r+1, 0])
        for r in range(1, image.shape[0]):
            for c in range(1, image.shape[1]):
                highestValleys[r, c] = min(image[r,c], max(highestValleys[r-1, c-1], highestValleys[r-1, c], highestValleys[r, c-1]))

        valleys[i] = highestValleys[-1, -1]

    #plt.hist(valleys, bins=20)
    #plt.show()
    shallowMask = valleys > minThresh #there isn't a deep enough valley between this peak and the one after it
    clusterStartMask = np.diff(np.roll(shallowMask.astype(int), 1)) > 0
    clusterInds = np.where(clusterStartMask)[0]

    indsToDelete = []
    for ind in clusterInds:
        maxCoords = resCoords[ind]
        maxScore = wpsmap[maxCoords[0], maxCoords[1], 0]
        maxInd = ind
        curInd = ind
        while(shallowMask[curInd]):
            curInd += 1
            curCoords = resCoords[curInd]
            if wpsmap[curCoords[0], curCoords[1], 0] > maxScore:
                maxScore = wpsmap[curCoords[0], curCoords[1], 0]
                indsToDelete.append(maxInd)
                maxInd = curInd
                maxCoords = curCoords
            else:
                indsToDelete.append(curInd)

    assert len(indsToDelete) == np.sum(shallowMask)

    resCoords = np.delete(resCoords, indsToDelete, axis=0)

    print 'Prominence cut deleted', len(indsToDelete), 'resonators'

    return resCoords

def saveMetadata(outFile, resFreqs, resAttens, scores, feedline, band, collThresh=200.e3):
    assert len(resFreqs) == len(resAttens) == len(scores), 'Lists must be the same length'

    flag = np.zeros(len(resFreqs))
    collMask = np.abs(np.diff(resFreqs)) < collThresh
    collMask = np.append(collMask, False)
    flag[collMask] = sd.ISBAD
    flag[~collMask] = sd.ISGOOD
    
    badScores = np.zeros(len(scores))

    resIDStart = feedline*10000
    if band.lower() == 'b':
        resIDStart += N_RES_PER_BOARD
    resIDs = np.arange(resIDStart, resIDStart + len(resFreqs))

    md = sd.SweepMetadata(resIDs, wsfreq=resFreqs, mlfreq=resFreqs, mlatten=resAttens, ml_isgood_score=scores, 
            ml_isbad_score=badScores, flag=flag)
    md.save(outFile)
    
def runFullInference(inferenceData, model, peakThresh, prominenceThresh, 
        nRes, attenBias, saveWPSMap, remakeWPSMap):
    wpsmapFile = os.path.join(os.path.dirname(inferenceData), os.path.basename(inferenceData).split('.')[0] \
            + '_' + os.path.basename(model) + '.npz')

    if not os.path.isfile(wpsmapFile) or args.remake_wpsmap:
        print wpsmapFile, 'not found'
        print 'Generating new WPS map'
        freqSweep = sd.FreqSweep(inferenceData)
        wpsmap, freqs, attens = makeWPSMap(model, freqSweep)

        if args.save_wpsmap:
            np.savez(wpsmapFile, wpsmap=wpsmap, freqs=freqs, attens=attens)

    else:
        print 'Loading WPS map', wpsmapFile
        f = np.load(wpsmapFile)
        wpsmap = f['wpsmap']
        freqs = f['freqs']
        attens = f['attens']

    resFreqs, resAttens, scores = findResonators(wpsmap, freqs, attens, prominenceThresh=prominenceThresh, 
            peakThresh=peakThresh, nRes=nRes, attenGrad=attenBias, resMagCut=False)

    return resFreqs, resAttens, scores



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='WPS ML Inference Script')
    parser.add_argument('model', help='Directory containing ML model')
    parser.add_argument('inferenceData', help='npz file containing WPS data. Can be a pattern specifying\
            multiple files, with tags {roach}, {feedline}, and/or {range}')
    parser.add_argument('-o', '--metadata', default=None, help='Output metadata file pattern, may use the\
            same tags as inferenceData for multiple files')
    parser.add_argument('-s', '--save-wpsmap', action='store_true', help='Save npz file containing raw ML convolution output')
    parser.add_argument('-r', '--remake-wpsmap', action='store_true', help='Regenerate wps map file')
    parser.add_argument('-t', '--peak-thresh', type=float, default=0.9, help='Minimum res value in WPS map (between 0 and 1)')
    parser.add_argument('-pr', '--prominence-thresh', type=float, default=0.85, help='Prevent double counting resonators by cutting shallow valleys')
    parser.add_argument('-n', '--n-res', type=int, default=None, help='Target number of resonators')
    parser.add_argument('-b', '--atten-bias', type=float, default=0., help='Apply linear bias to atten dim of wpsmap')
    parser.add_argument('-m', '--use-mag', action='store_true', help='Select N_RES resonators above peakThresh with largest loop size')
    args = parser.parse_args()

    sweepFiles, paramDicts = sd.getSweepFilesFromPat(args.inferenceData)
    print sweepFiles
    print paramDicts

    if args.metadata is None:
        args.metadata = os.path.join(os.path.dirname(args.inferenceData), 
                os.path.basename(args.inferenceData).split('.')[0] + '_metadata.txt')

    elif not os.path.isabs(args.metadata):
        args.metadata = os.path.join(os.path.dirname(args.inferenceData), args.metadata)


    for (sweepFile, paramDict) in zip(sweepFiles, paramDicts):
        resFreqs, resAttens, scores = runFullInference(sweepFile, args.model, args.peak_thresh, 
                    args.prominence_thresh, args.n_res, args.atten_bias, args.save_wpsmap, args.remake_wpsmap)
 
        if 'range' in paramDict:
            band = paramDict['range']
        elif resFreqs[0] < 4.7e9:
            band = 'a'
        else:
            band = 'b'

        if 'feedline' in paramDict:
            fl = paramDict['feedline']
        else:
            try:
                fl = inst.guessFeedline(os.path.basename(sweepFile))
            except ValueError:
                fl = 1

        if 'roach' in paramDict:
            roach = paramDict['roach']
        else:
            roach = 0

        mdOutFile = args.metadata.format(range=band, roach=roach, feedline=fl)

        print 'Saving resonator metadata in:', mdOutFile
        saveMetadata(mdOutFile, resFreqs, resAttens, scores, fl, band)

import numpy as np
import tensorflow as tf
from functools import partial
import multiprocessing
import os, sys, glob
import time
import copy
import argparse
import logging
import ipdb
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

    freqStart = freqSweep.freqs[0, 0] + freqSweep.freqStep*mlDict['freqWinSize']
    freqEnd = freqSweep.freqs[-1, -1] - freqSweep.freqStep*mlDict['freqWinSize']
    freqs = np.arange(freqStart, freqEnd, freqStep)

    wpsImage = np.zeros((len(attens), len(freqs), N_CLASSES + 1))
    nColors = 2
    if mlDict['useIQV']:
        nColors += 1
    if mlDict['useVectIQV']:
        nColors += 2

    chunkSize = 5000#8000*N_CPU
    subChunkSize = chunkSize/N_CPU
    imageList = np.zeros((chunkSize, mlDict['attenWinBelow'] + mlDict['attenWinAbove'] + 1, mlDict['freqWinSize'], nColors))
    resMagList = np.zeros((chunkSize))
    labelsList = np.zeros((chunkSize, N_CLASSES))
    toneWinCenters = freqSweep.freqs[:, freqSweep.nlostep/2]
    if N_CPU > 1:
        pool = multiprocessing.Pool(processes=N_CPU)
        freqSweepChunk = copy.copy(freqSweep)

    for attenInd in range(len(attens)):
        tstart = time.time()
        for chunkInd in range(len(freqs)/chunkSize + 1):
            nFreqsInChunk = min(chunkSize, len(freqs) - chunkSize*chunkInd)

            if N_CPU == 1:
                for i, freqInd in enumerate(range(chunkSize*chunkInd, chunkSize*chunkInd + nFreqsInChunk)):
                    imageList[i], resMagList[i], _, _  = mlt.makeWPSImage(freqSweep, freqs[freqInd], attens[attenInd], mlDict['freqWinSize'],
                            1+mlDict['attenWinBelow']+mlDict['attenWinAbove'], mlDict['useIQV'], mlDict['useVectIQV']) 
            else:
                freqList = freqs[range(chunkSize*chunkInd, chunkSize*chunkInd + nFreqsInChunk)]
                toneIndLow = np.argmin(np.abs(freqList[0] - toneWinCenters))
                toneIndHigh = np.argmin(np.abs(freqList[-1] - toneWinCenters)) + 1
                freqSweepChunk.i = freqSweep.i[:, toneIndLow:toneIndHigh, :]
                freqSweepChunk.q = freqSweep.q[:, toneIndLow:toneIndHigh, :]
                freqSweepChunk.freqs = freqSweep.freqs[toneIndLow:toneIndHigh, :]
                freqSweepChunk.ntone = toneIndHigh - toneIndLow + 1
                
                processChunk = partial(makeImage, freqSweep=freqSweepChunk, atten=attens[attenInd], 
                            freqWinSize=mlDict['freqWinSize'], attenWinSize=1+mlDict['attenWinBelow']+mlDict['attenWinAbove'], 
                            useIQV=mlDict['useIQV'], useVectIQV=mlDict['useVectIQV']) 

                imageList[:nFreqsInChunk], resMagList[:nFreqsInChunk] = zip(*pool.map(processChunk, freqList, chunksize=chunkSize/N_CPU))

            wpsImage[attenInd, chunkSize*chunkInd:chunkSize*chunkInd + nFreqsInChunk, :N_CLASSES] = sess.run(y_output, 
                    feed_dict={x_input: imageList[:nFreqsInChunk], keep_prob: 1, is_training: False})
            wpsImage[attenInd, chunkSize*chunkInd:chunkSize*chunkInd + nFreqsInChunk, N_CLASSES] = resMagList[:nFreqsInChunk]
            print 'finished chunk', chunkInd, 'out of', len(freqs)/chunkSize

        print 'atten:', attens[attenInd]
        print ' took', time.time() - tstart, 'seconds'

    if N_CPU > 1:
        pool.close()

    return wpsImage, freqs, attens


def makeImage(centerFreq, freqSweep, atten, freqWinSize, attenWinSize, useIQV, useVectIQV):
    image, resMag, _, _, = mlt.makeWPSImage(freqSweep, centerFreq, atten, freqWinSize, attenWinSize, useIQV, useVectIQV) 
    return image, resMag

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

def findResonators(wpsmap, freqs, attens, peakThresh=0.97, minPeakDist=40.e3, nRes=None, attenGrad=0, resMagCut=False):
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
        resCoords = prominenceCut(wpsmap, resCoords)
        resMags = np.zeros(len(resCoords))
        print 'Found', len(resCoords), 'peaks above', peakThresh
        for i in range(len(resMags)):
            resMags[i] = wpsmap[resCoords[i, 0], resCoords[i, 1], N_CLASSES]
        largestMagInds = np.argsort(resMags)[::-1]
        largestMagInds = largestMagInds[:nRes]
        resCoords = resCoords[largestMagInds]

    elif nRes is not None:
        resCoords = skf.peak_local_max(wpsmap[:,:,0], min_distance=minPeakDist, threshold_abs=peakThresh, num_peaks=nRes, exclude_border=False)
        resCoords = prominenceCut(wpsmap, resCoords)
    else:
        resCoords = skf.peak_local_max(wpsmap[:,:,0], min_distance=minPeakDist, threshold_abs=peakThresh, exclude_border=False)
        resCoords = prominenceCut(wpsmap, resCoords)

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

def prominenceCut(wpsmap, resCoords, minThresh=0.9):
    freqSortedInds = np.argsort(resCoords[:,1])
    resCoords = resCoords[freqSortedInds]
    valleys = np.zeros(len(resCoords))
    for i in range(len(resCoords) - 1):
        attenInds = np.sort([resCoords[i, 0], resCoords[i+1, 0]])
        #image = wpsmap[attenInds[0]:attenInds[1]+1, resCoords[i,1]:resCoords[i+1,1]+1, 0]
        #coords = skf.peak_local_max(-image, num_peaks=1, exclude_border=False)
        #valleys[i] = np.min(image)#image[coords[0], coords[1]]
        valleys[i] = wpsmap[int(np.ceil((attenInds[0]+attenInds[1])/2.)), (resCoords[i,1]+resCoords[i+1,1])/2, 0]

    #plt.hist(valleys)
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
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='WPS ML Inference Script')
    parser.add_argument('model', help='Directory containing ML model')
    parser.add_argument('inferenceData', help='npz file containing WPS data')
    parser.add_argument('-o', '--metadata', default=None, help='Output metadata file')
    parser.add_argument('-s', '--save-wpsmap', action='store_true', help='Save npz file containing raw ML convolution output')
    parser.add_argument('-r', '--remake-wpsmap', action='store_true', help='Regenerate wps map file')
    parser.add_argument('-t', '--peak-thresh', type=float, default=0.95, help='Minimum res value in WPS map (between 0 and 1)')
    parser.add_argument('-n', '--n-res', type=int, default=None, help='Target number of resonators')
    parser.add_argument('-b', '--atten-bias', type=float, default=0., help='Apply linear bias to atten dim of wpsmap')
    parser.add_argument('-m', '--use-mag', action='store_true', help='Select N_RES resonators above peakThresh with largest loop size')
    args = parser.parse_args()

    if args.metadata is None:
        args.metadata = os.path.join(os.path.dirname(args.inferenceData), os.path.basename(args.inferenceData).split('.')[0] + '_metadata.txt')

    elif not os.path.isabs(args.metadata):
        args.metadata = os.path.join(os.path.dirname(args.inferenceData), args.metadata)

    wpsmapFile = os.path.join(os.path.dirname(args.metadata), os.path.basename(args.inferenceData).split('.')[0] \
            + '_' + os.path.basename(args.model) + '.npz')

    if not os.path.isfile(wpsmapFile) or args.remake_wpsmap:
        print wpsmapFile, 'not found'
        print 'Generating new WPS map'
        freqSweep = sd.FreqSweep(args.inferenceData)
        wpsmap, freqs, attens = makeWPSMap(args.model, freqSweep)

        if args.save_wpsmap:
            np.savez(wpsmapFile, wpsmap=wpsmap, freqs=freqs, attens=attens)

    else:
        print 'Loading WPS map', wpsmapFile
        f = np.load(wpsmapFile)
        wpsmap = f['wpsmap']
        freqs = f['freqs']
        attens = f['attens']

    resFreqs, resAttens, scores = findResonators(wpsmap, freqs, attens, peakThresh=args.peak_thresh, 
            nRes=args.n_res, attenGrad=args.atten_bias, resMagCut=args.use_mag)

    if resFreqs[0] < 4.7e9:
        band = 'a'
    else:
        band = 'b'

    print 'Saving resonator metadata in:', args.metadata
    try:
        fl = inst.guessFeedline(os.path.basename(args.inferenceData))
    except ValueError:
        fl = 1
    saveMetadata(args.metadata, resFreqs, resAttens, scores, fl, band)

import numpy as np
import random
import tensorflow as tf
import logging
import os, sys, glob
from PSFitMLData import *
from mkidreadout.configuration.powersweep.psmldata import *

def makeWPSImage(freqSweep, centerFreq, centerAtten, nFreqs, nAttens, useIQV, useVectIQV):
    """
    dataObj: FreqSweep object
    """
    winCenters = freqSweep.freqs[:, freqSweep.nlostep/2]
    toneInd = np.argmin(np.abs(centerFreq - winCenters)) #index of resonator tone to use
    toneFreqs = freqSweep.freqs[toneInd, :]
    assert toneFreqs[0] < centerFreq < toneFreqs[-1], 'centerFreq out of range'

    centerFreqInd = np.argmin(np.abs(toneFreqs - centerFreq))
    startFreqInd = centerFreqInd - int(np.floor(nFreqs/2.))
    endFreqInd = centerFreqInd + int(np.ceil(nFreqs/2.))

    if startFreqInd < 0:
        startFreqPads = 0 - startFreqInd
        startFreqInd = 0
    else:
        startFreqPads = 0
    if endFreqInd > len(toneFreqs):
        endFreqPads = endFreqInd - len(toneFreqs)
        endFreqInd = len(toneFreqs)
    else:
        endFreqPads = 0

    assert freqSweep.atten[0] <= centerAtten <= freqSweep.atten[-1], 'centerAtten out of range'
    centerAttenInd = np.argmin(np.abs(freqSweep.atten - centerAtten))
    startAttenInd = centerAttenInd - int(np.floor(nAttens/2.))
    endAttenInd = centerAttenInd + int(np.ceil(nAttens/2.))

    if startAttenInd < 0:
        startAttenPads = 0 - startAttenInd
        startAttenInd = 0
    else:
        startAttenPads = 0
    if endAttenInd > len(freqSweep.atten):
        endAttenPads = endAttenInd - len(freqSweep.atten)
        endAttenInd = len(freqSweep.atten)
    else:
        endAttenPads = 0

    freqs = toneFreqs[startFreqInd:endFreqInd]
    attens = freqSweep.atten[startAttenInd:endAttenInd]
    iVals = freqSweep.i[startAttenInd:endAttenInd, toneInd, startFreqInd:endFreqInd]
    qVals = freqSweep.q[startAttenInd:endAttenInd, toneInd, startFreqInd:endFreqInd]
    iqVels = np.sqrt(np.diff(iVals, axis=1)**2 + np.diff(qVals, axis=1)**2)
    iVels = np.diff(iVals, axis=1)
    qVels = np.diff(qVals, axis=1)

    freqs = np.pad(freqs, (startFreqPads, endFreqPads), 'edge')
    attens = np.pad(attens, (startAttenPads, endAttenPads), 'edge')
    iVals = np.pad(iVals, ((startAttenPads, endAttenPads), (startFreqPads, endFreqPads)), 'edge')
    qVals = np.pad(qVals, ((startAttenPads, endAttenPads), (startFreqPads, endFreqPads)), 'edge')
    iqVels = np.pad(iqVels, ((startAttenPads, endAttenPads), (startFreqPads, endFreqPads+1)), 'edge')
    iVels = np.pad(iVels, ((startAttenPads, endAttenPads), (startFreqPads, endFreqPads+1)), 'edge')
    qVels = np.pad(qVels, ((startAttenPads, endAttenPads), (startFreqPads, endFreqPads+1)), 'edge')

    iVals = np.transpose(np.transpose(iVals) - np.mean(iVals, axis=1))
    qVals = np.transpose(np.transpose(qVals) - np.mean(qVals, axis=1))
    res_mag = np.sqrt(np.mean(iVals**2 + qVals**2, axis=1))
    iVals = np.transpose(np.transpose(iVals)/res_mag)
    qVals = np.transpose(np.transpose(qVals)/res_mag)

    iqVels = np.transpose(np.transpose(iqVels) - np.mean(iqVels, axis=1))
    iqVels = np.transpose(np.transpose(iqVels)/res_mag)
    iqVels /= np.sqrt(np.mean(iqVels**2))

    iVels = np.transpose(np.transpose(iVels) - np.mean(iVels, axis=1))
    iVels = np.transpose(np.transpose(iVels)/res_mag)
    iVels /= np.sqrt(np.mean(iVels**2))

    qVels = np.transpose(np.transpose(qVels) - np.mean(qVels, axis=1))
    qVels = np.transpose(np.transpose(qVels)/res_mag)
    qVels /= np.sqrt(np.mean(qVels**2))


    image = np.dstack((iVals, qVals))
    if useIQV:
        image = np.dstack((image, iqVels))
    if useVectIQV:
        image = np.dstack((image, iVels))
        image = np.dstack((image, qVels))

    return image, attens, freqs




def makeResImage(res_num, dataObj, wsAttenInd, xWidth, resWidth, 
            pad_res_win, useIQV, useMag, centerLoop, nAttensModel, useVectIQV=False, collisionRange=100.e3):
    """Creates a table with 2 rows, I and Q for makeTrainData(mag_data=True)

    inputs
    res_num: index of resonator in question
    iAtten: index of attenuation in question
    angle: angle of rotation about the origin (radians)
    showFrames: pops up a window of the frame plotted using matplotlib.plot
    """
    # TODO: remove assumption that res is centered on window at wsAttenInd
    resSearchWin = 7

    assert resWidth <= xWidth, 'res width must be <= xWidth'

    nFreqPoints = len(dataObj.iq_vels[res_num, 0, :])
    nAttens = dataObj.Is.shape[1]
    assert resWidth <= nFreqPoints, 'res width must be <= number of freq steps'
    attenList = dataObj.attens

    iq_vels = dataObj.iq_vels[res_num, :, :]
    Is = dataObj.Is[res_num, :, :-1]  # -1 to make size the same as iq vels
    Qs = dataObj.Qs[res_num, :, :-1]
    freqs = dataObj.freqs[res_num][:-1]

    # Assumes initfreqs is sorted
    if collisionRange>0:
        goodMask = np.ones(nFreqPoints, dtype=bool)
        if res_num > 0:
            goodMask &= (freqs - dataObj.initfreqs[res_num-1]) >= collisionRange
        if res_num < dataObj.freqs.shape[0]-1:
            goodMask &= (dataObj.initfreqs[res_num+1] - freqs) >= collisionRange

        iq_vels = iq_vels[:, goodMask]
        Is = Is[:, goodMask]
        Qs = Qs[:, goodMask]
        freqs = freqs[goodMask]

        nFreqPoints = np.sum(goodMask)
        if resWidth > nFreqPoints:
            resWidth = nFreqPoints
        

    freqCube = np.zeros((nAttens, resWidth))
    magsdb = Is ** 2 + Qs ** 2

    # make sliding window images
    singleFrameImage = np.zeros((nAttens, resWidth, 2))
    iqVelImage = np.zeros((nAttens, resWidth))
    magsdbImage = np.zeros((nAttens, resWidth))

    if resWidth < nFreqPoints:
        try:
            wsFreq = dataObj.initfreqs[res_num]
            assert freqs[0] <= wsFreq <= freqs[-1], 'ws freq out of window'
            initWinCenter = np.argmin(np.abs(wsFreq - freqs))
        except AttributeError:
            resSearchStartWin = int(nFreqPoints / 2 - np.floor(resSearchWin / 2.))
            resSearchEndWin = int(nFreqPoints / 2 + np.ceil(resSearchWin / 2.))
            initWinCenter = resSearchStartWin + np.argmin(magsdb[wsAttenInd, resSearchStartWin:resSearchEndWin])

        winCenter = initWinCenter
        startWin = int(winCenter - np.floor(resWidth / 2.))
        endWin = int(winCenter + np.ceil(resWidth / 2.))
        resSearchStartWin = int(winCenter - np.floor(resSearchWin / 2.))
        resSearchEndWin = int(winCenter + np.ceil(resSearchWin / 2.))

        for i in range(wsAttenInd, -1, -1):
            resSearchStartWin = max(0, resSearchStartWin)
            resSearchEndWin = min(nFreqPoints, resSearchEndWin)
            oldWinMags = magsdb[i, resSearchStartWin:resSearchEndWin]
            newWinCenter = resSearchStartWin + np.argmin(oldWinMags)
            startWin += (newWinCenter - winCenter)
            endWin += (newWinCenter - winCenter)
            resSearchStartWin += (newWinCenter - winCenter)
            resSearchEndWin += (newWinCenter - winCenter)
            winCenter = newWinCenter
            if pad_res_win:
                if startWin < 0:
                    singleFrameImage[i, :, 0] = np.pad(Is[i, 0:endWin], (0 - startWin, 0), 'edge')
                    singleFrameImage[i, :, 1] = np.pad(Qs[i, 0:endWin], (0 - startWin, 0), 'edge')
                    iqVelImage[i, :] = np.pad(iq_vels[i, 0:endWin], (0 - startWin, 0), 'edge')
                    magsdbImage[i, :] = np.pad(magsdb[i, 0:endWin], (0 - startWin, 0), 'edge')
                    freqCube[i, :] = np.pad(freqs[0:endWin], (0 - startWin, 0), 'edge')
                elif endWin > nFreqPoints:
                    singleFrameImage[i, :, 0] = np.pad(Is[i, startWin:], (0, endWin - nFreqPoints), 'edge')
                    singleFrameImage[i, :, 1] = np.pad(Qs[i, startWin:], (0, endWin - nFreqPoints), 'edge')
                    iqVelImage[i, :] = np.pad(iq_vels[i, startWin:], (0, endWin - nFreqPoints), 'edge')
                    magsdbImage[i, :] = np.pad(magsdb[i, startWin:], (0, endWin - nFreqPoints), 'edge')
                    freqCube[i, :] = np.pad(freqs[startWin:], (0, endWin - nFreqPoints), 'edge')
                else:
                    singleFrameImage[i, :, 0] = Is[i, startWin:endWin]
                    singleFrameImage[i, :, 1] = Qs[i, startWin:endWin]
                    iqVelImage[i, :] = iq_vels[i, startWin:endWin]
                    magsdbImage[i, :] = magsdb[i, startWin:endWin]
                    freqCube[i, :] = freqs[startWin:endWin]
            else:
                if startWin < 0:
                    endWin += -startWin
                    startWin = 0
                elif endWin > nFreqPoints:
                    startWin -= endWin - nFreqPoints
                    endWin = nFreqPoints
                singleFrameImage[i, :, 0] = Is[i, startWin:endWin]
                singleFrameImage[i, :, 1] = Qs[i, startWin:endWin]
                magsdbImage[i, :] = magsdb[i, startWin:endWin]  # iq_vels[i, startWin:endWin]
                iqVelImage[i, :] = iq_vels[i, startWin:endWin]  # iq_vels[i, startWin:endWin]
                freqCube[i, :] = freqs[startWin:endWin]

        winCenter = initWinCenter
        startWin = int(winCenter - np.floor(resWidth / 2.))
        endWin = int(winCenter + np.ceil(resWidth / 2.))
        resSearchStartWin = int(winCenter - np.floor(resSearchWin / 2.))
        resSearchEndWin = int(winCenter + np.ceil(resSearchWin / 2.))
        for i in range(wsAttenInd + 1, nAttens):
            resSearchStartWin = max(0, resSearchStartWin)
            resSearchEndWin = min(nFreqPoints, resSearchEndWin)
            oldWinMags = magsdb[i, resSearchStartWin:resSearchEndWin]
            newWinCenter = resSearchStartWin + np.argmin(oldWinMags)
            startWin += (newWinCenter - winCenter)
            endWin += (newWinCenter - winCenter)
            resSearchStartWin += (newWinCenter - winCenter)
            resSearchEndWin += (newWinCenter - winCenter)
            winCenter = newWinCenter
            if pad_res_win:
                if startWin < 0:
                    singleFrameImage[i, :, 0] = np.pad(Is[i, 0:endWin], (0 - startWin, 0), 'edge')
                    singleFrameImage[i, :, 1] = np.pad(Qs[i, 0:endWin], (0 - startWin, 0), 'edge')
                    iqVelImage[i, :] = np.pad(iq_vels[i, 0:endWin], (0 - startWin, 0), 'edge')
                    magsdbImage[i, :] = np.pad(magsdb[i, 0:endWin], (0 - startWin, 0), 'edge')
                    freqCube[i, :] = np.pad(freqs[0:endWin], (0 - startWin, 0), 'edge')
                elif endWin > nFreqPoints:
                    singleFrameImage[i, :, 0] = np.pad(Is[i, startWin:], (0, endWin - nFreqPoints), 'edge')
                    singleFrameImage[i, :, 1] = np.pad(Qs[i, startWin:], (0, endWin - nFreqPoints), 'edge')
                    iqVelImage[i, :] = np.pad(iq_vels[i, startWin:], (0, endWin - nFreqPoints), 'edge')
                    magsdbImage[i, :] = np.pad(magsdb[i, startWin:], (0, endWin - nFreqPoints), 'edge')
                    freqCube[i, :] = np.pad(freqs[startWin:], (0, endWin - nFreqPoints), 'edge')
                else:
                    singleFrameImage[i, :, 0] = Is[i, startWin:endWin]
                    singleFrameImage[i, :, 1] = Qs[i, startWin:endWin]
                    iqVelImage[i, :] = iq_vels[i, startWin:endWin]
                    magsdbImage[i, :] = magsdb[i, startWin:endWin]
                    freqCube[i, :] = freqs[startWin:endWin]
            else:
                if startWin < 0:
                    endWin += -startWin
                    startWin = 0
                elif endWin > nFreqPoints:
                    startWin -= endWin - nFreqPoints
                    endWin = nFreqPoints
                startWin = max(0, startWin)
                endWin = min(endWin, nFreqPoints)
                singleFrameImage[i, :, 0] = Is[i, startWin:endWin]
                singleFrameImage[i, :, 1] = Qs[i, startWin:endWin]
                iqVelImage[i, :] = iq_vels[i, startWin:endWin]  # iq_vels[i, startWin:endWin]
                magsdbImage[i, :] = magsdb[i, startWin:endWin]  # iq_vels[i, startWin:endWin]
                freqCube[i, :] = freqs[startWin:endWin]

    else:
        singleFrameImage[:, :, 0] = Is
        singleFrameImage[:, :, 1] = Qs
        iqVelImage = iq_vels
        magsdbImage = magsdb
        freqCube = np.tile(freqs, (nAttens, 1))

    res_mag = np.sqrt(np.amax(singleFrameImage[:, :, 0] ** 2 + singleFrameImage[:, :, 1] ** 2,
                              axis=1))  # changed by NF 20180423 (originally amax)
    singleFrameImage[:, :, 0] = np.transpose(np.transpose(singleFrameImage[:, :, 0]) / res_mag)
    singleFrameImage[:, :, 1] = np.transpose(np.transpose(singleFrameImage[:, :, 1]) / res_mag)

    #iqVelImage = np.transpose(np.transpose(iqVelImage) / np.sqrt(np.amax(iqVelImage ** 2, axis=1))) 
        #replaced with edit after 'if center_loop' on 20190107
    magsdbImage = np.transpose(np.transpose(magsdbImage) / np.sqrt(np.mean(magsdbImage ** 2, axis=1)))
    #iqVelImage = iqVelImage/np.sqrt(np.mean(iqVelImage**2)) #changed 20190107, probably a mistake earlier
        #20190324 - move this before scaling

    iVelImage = np.diff(singleFrameImage[:, :, 0], axis=1)
    qVelImage = np.diff(singleFrameImage[:, :, 1], axis=1)
    iVelImage = np.pad(iVelImage, ((0, 0), (0, 1)), 'edge')
    qVelImage = np.pad(qVelImage, ((0, 0), (0, 1)), 'edge')

    if centerLoop:
        singleFrameImage[:, :, 0] = np.transpose(
            np.transpose(singleFrameImage[:, :, 0]) - np.mean(singleFrameImage[:, :, 0], 1))

        singleFrameImage[:, :, 1] = np.transpose(
            np.transpose(singleFrameImage[:, :, 1]) - np.mean(singleFrameImage[:, :, 1], 1))
        iqVelImage = np.transpose(np.transpose(iqVelImage) - np.mean(iqVelImage, 1))  # added by NF 20180423
        magsdbImage = np.transpose(np.transpose(magsdbImage) - np.mean(magsdbImage, 1))  # added by NF 20180423
        iVelImage = np.transpose(np.transpose(iVelImage) - np.mean(iVelImage, 1))
        qVelImage = np.transpose(np.transpose(qVelImage) - np.mean(qVelImage, 1))


    iqVelImage = iqVelImage/np.sqrt(np.mean(iqVelImage**2)) #changed 20190107, probably a mistake earlier
    iVelImage = iVelImage/np.sqrt(np.mean(iVelImage**2)) #changed 20190107, probably a mistake earlier
    qVelImage = qVelImage/np.sqrt(np.mean(qVelImage**2)) #changed 20190107, probably a mistake earlier
    

    if resWidth < xWidth:
        nPadVals = (xWidth - resWidth) / 2.
        singleFrameImageFS = np.zeros((nAttens, xWidth, singleFrameImage.shape[2]))
        freqCubeFS = np.zeros((nAttens, xWidth))
        for i in range(singleFrameImage.shape[2]):
            singleFrameImageFS[:, :, i] = np.pad(singleFrameImage[:, :, i],
                                                 [(0, 0), (int(np.ceil(nPadVals)), int(np.floor(nPadVals)))], 'edge')
        iqVelImage = np.pad(iqVelImage, [(0, 0), (int(np.ceil(nPadVals)), int(np.floor(nPadVals)))], 'edge')
        iVelImage = np.pad(iVelImage, [(0, 0), (int(np.ceil(nPadVals)), int(np.floor(nPadVals)))], 'edge')
        qVelImage = np.pad(qVelImage, [(0, 0), (int(np.ceil(nPadVals)), int(np.floor(nPadVals)))], 'edge')
        magsdbImage = np.pad(magsdbImage, [(0, 0), (int(np.ceil(nPadVals)), int(np.floor(nPadVals)))], 'edge')
        freqCubeFS = np.pad(freqCube, [(0, 0), (int(np.ceil(nPadVals)), int(np.floor(nPadVals)))], 'edge')
        singleFrameImage = singleFrameImageFS
        freqCube = freqCubeFS

    if nAttens < nAttensModel:
        singleFrameImageFS = np.zeros((nAttensModel, xWidth, singleFrameImage.shape[2]))
        freqCubeFS = np.zeros((nAttens, xWidth))
        for i in range(singleFrameImage.shape[2]):
            singleFrameImageFS[:, :, i] = np.pad(singleFrameImage[:, :, i], [(0, mlDictnAttens - nAttens), (0, 0)],
                                                 'edge')
        iqVelImage = np.pad(iqVelImage, [(0, mlDictnAttens - nAttens), (0, 0)], 'edge')
        iVelImage = np.pad(iVelImage, [(0, mlDictnAttens - nAttens), (0, 0)], 'edge')
        qVelImage = np.pad(qVelImage, [(0, mlDictnAttens - nAttens), (0, 0)], 'edge')
        magsdbImage = np.pad(magsdbImage, [(0, mlDictnAttens - nAttens), (0, 0)], 'edge')
        freqCubeFS = np.pad(freqCube, [(0, mlDictnAttens - nAttens), (0, 0)], 'edge')
        singleFrameImage = singleFrameImageFS
        freqCube = freqCubeFS
        attenList = np.pad(attenList, (0, mlDictnAttens - nAttens), 'edge')

    #truncate the highest attens, consider expanding this if necessary
    elif nAttens > nAttensModel:
        singleFrameImage = singleFrameImage[:nAttensModel, :, :]
        iqVelImage = iqVelImage[:nAttensModel, :]
        iVelImage = iVelImage[:nAttensModel, :]
        qVelImage = qVelImage[:nAttensModel, :]
        magsdbImage = magsdbImage[:nAttensModel, :]
        freqCube = freqCube[:nAttensModel, :]
        attenList = attenList[:nAttensModel]

    if useIQV:
        singleFrameImage = np.dstack((singleFrameImage, iqVelImage))

    if useMag:
        singleFrameImage = np.dstack((singleFrameImage, magsdbImage))

    if useVectIQV:
        singleFrameImage = np.dstack((singleFrameImage, iVelImage))
        singleFrameImage = np.dstack((singleFrameImage, qVelImage))

    return singleFrameImage, freqCube, attenList, iqVelImage, magsdbImage

def get_ml_model(modelDir=''):
    modelList = glob.glob(os.path.join(modelDir, '*.meta'))
    if len(modelList) > 1:
        raise Exception('Multiple models (.meta files) found in directory: ' + modelDir)
    elif len(modelList) == 0:
        raise Exception('No models (.meta files) found in directory ' + modelDir)
    model = modelList[0]
    getLogger(__name__).info('Loading good model from %s', model)
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model)
    saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(model)))

    graph = tf.get_default_graph()
    x_input = graph.get_tensor_by_name('inputImage:0')
    y_output = graph.get_tensor_by_name('outputLabel:0')
    keep_prob = graph.get_tensor_by_name('keepProb:0')
    is_training = graph.get_tensor_by_name('isTraining:0')

    mlDict = {}
    for param in tf.get_collection('mlDict'):
        mlDict[param.op.name] = param.eval(session=sess)

    return mlDict, sess, graph, x_input, y_output, keep_prob, is_training

def get_peak_idx(res_num, iAtten, dataObj, smooth=False, cutType=None, padInd=None):
    iq_vels = dataObj.iq_vels[res_num, iAtten, :]
    if not cutType is None:
        if cutType == 'bottom':
            iq_vels[0:padInd] = 0
            print 'getpeakidx bottom 0'
        if cutType == 'top':
            iq_vels[padInd:-1] = 0
            print 'getpeakidx top 0'
            print 'cuttofFreq', dataObj.freqs[res_num, padInd]
    if smooth:
        iq_vels = np.correlate(iq_vels, np.ones(5), mode='same')
    return np.argmax(iq_vels)

def next_batch(trainImages, trainLabels, batch_size):
    '''selects a random batch of batch_size from trainImages and trainLabels'''
    perm = random.sample(range(len(trainImages)), batch_size)
    trainImagesBatch = trainImages[perm]
    trainLabelsBatch = trainLabels[perm]
    #print 'next_batch trImshape', np.shape(trainImages)
    return trainImagesBatch, trainLabelsBatch



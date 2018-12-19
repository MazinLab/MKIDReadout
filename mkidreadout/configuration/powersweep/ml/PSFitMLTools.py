import numpy as np
from PSFitMLData import *
from mkidreadout.configuration.powersweep.psmldata import *


def makeResImage(res_num, center_loop=False, showFrames=False,
                 dataObj=None, mlDict=None, wsAttenInd=None,
                 xWidth=None, resWidth=None, pad_res_win=None, useIQV=True, useMag=True,
                 mlDictnAttens=None):
    """Creates a table with 2 rows, I and Q for makeTrainData(mag_data=True)

    inputs
    res_num: index of resonator in question
    iAtten: index of attenuation in question
    angle: angle of rotation about the origin (radians)
    showFrames: pops up a window of the frame plotted using matplotlib.plot
    """
    # TODO: remove assumption that res is centered on window at wsAttenInd
    if mlDict is not None:
        xWidth = mlDict['xWidth']  #
        resWidth = mlDict['resWidth']
        pad_res_win = mlDict['padResWin']
        useIQV = mlDict['useIQV']
        useMag = mlDict['useMag']
        mlDictnAttens = mlDict['nAttens']
    resSearchWin = 20

    assert resWidth <= xWidth, 'res width must be <= xWidth'

    nFreqPoints = len(dataObj.iq_vels[res_num, 0, :])
    nAttens = dataObj.Is.shape[1]
    assert resWidth <= nFreqPoints, 'res width must be <= number of freq steps'
    attenList = dataObj.attens

    iq_vels = dataObj.iq_vels[res_num, :, :]
    Is = dataObj.Is[res_num, :, :-1]  # -1 to make size the same as iq vels
    Qs = dataObj.Qs[res_num, :, :-1]
    freqs = dataObj.freqs[res_num][:-1]
    freqCube = np.zeros((nAttens, resWidth))

    # plt.plot(self.Is[res_num,iAtten], self.Qs[res_num,iAtten])
    # plt.show()
    # for spectra where the peak is close enough to the edge that some points falls across the bounadry, pad zeros

    magsdb = Is ** 2 + Qs ** 2

    # iq_vels = np.round(iq_vels * xWidth / max(dataObj.iq_vels[res_num, iAtten, :]) )

    if showFrames:
        fig = plt.figure(frameon=False, figsize=(15.0, 5.0))
        fig.add_subplot(131)
        plt.plot(iq_vels)
        plt.ylim(0, 1)
        fig.add_subplot(132)
        plt.plot(Is)
        plt.plot(Qs)
        fig.add_subplot(133)
        plt.plot(Is, Qs)
        plt.show()
        plt.close()

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

    res_mag = np.sqrt(np.amax(singleFrameImage[:, :, 0] ** 2 + singleFrameImage[:, :, 1] ** 2,
                              axis=1))  # changed by NF 20180423 (originally amax)
    singleFrameImage[:, :, 0] = np.transpose(np.transpose(singleFrameImage[:, :, 0]) / res_mag)
    singleFrameImage[:, :, 1] = np.transpose(np.transpose(singleFrameImage[:, :, 1]) / res_mag)

    iqVelImage = np.transpose(np.transpose(iqVelImage) / np.sqrt(np.amax(iqVelImage ** 2, axis=1)))
    magsdbImage = np.transpose(np.transpose(magsdbImage) / np.sqrt(np.mean(magsdbImage ** 2, axis=1)))

    if center_loop:
        singleFrameImage[:, :, 0] = np.transpose(
            np.transpose(singleFrameImage[:, :, 0]) - np.mean(singleFrameImage[:, :, 0], 1))

        singleFrameImage[:, :, 1] = np.transpose(
            np.transpose(singleFrameImage[:, :, 1]) - np.mean(singleFrameImage[:, :, 1], 1))
        iqVelImage = np.transpose(np.transpose(iqVelImage) - np.mean(iqVelImage, 1))  # added by NF 20180423
        magsdbImage = np.transpose(np.transpose(magsdbImage) - np.mean(magsdbImage, 1))  # added by NF 20180423

    if useIQV:
        singleFrameImage = np.dstack((singleFrameImage, iqVelImage))

    if useMag:
        singleFrameImage = np.dstack((singleFrameImage, magsdbImage))

    if resWidth < xWidth:
        nPadVals = (xWidth - resWidth) / 2.
        singleFrameImageFS = np.zeros((nAttens, xWidth, singleFrameImage.shape[2]))
        freqCubeFS = np.zeros((nAttens, xWidth))
        for i in range(singleFrameImage.shape[2]):
            singleFrameImageFS[:, :, i] = np.pad(singleFrameImage[:, :, i],
                                                 [(0, 0), (int(np.ceil(nPadVals)), int(np.floor(nPadVals)))], 'edge')
        freqCubeFS = np.pad(freqCube, [(0, 0), (int(np.ceil(nPadVals)), int(np.floor(nPadVals)))], 'edge')
        singleFrameImage = singleFrameImageFS
        freqCube = freqCubeFS

    if nAttens < mlDictnAttens:
        singleFrameImageFS = np.zeros((mlDictnAttens, xWidth, singleFrameImage.shape[2]))
        freqCubeFS = np.zeros((nAttens, xWidth))
        for i in range(singleFrameImage.shape[2]):
            singleFrameImageFS[:, :, i] = np.pad(singleFrameImage[:, :, i], [(0, mlDictnAttens - nAttens), (0, 0)],
                                                 'edge')
        freqCubeFS = np.pad(freqCube, [(0, mlDictnAttens - nAttens), (0, 0)], 'edge')
        singleFrameImage = singleFrameImageFS
        freqCube = freqCubeFS
        attenList = np.pad(attenList, (0, mlDictnAttens - nAttens), 'edge')

    return singleFrameImage, freqCube, attenList


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

import numpy as np
from PSFitMLData import *

def makeResImage(res_num, angle=1, center_loop=False,  phase_normalise=False, showFrames=False, test_if_noisy=False, dataObj=None, padFreq=None, mlDict=None, wsAttenInd=None):
    '''Creates a table with 2 rows, I and Q for makeTrainData(mag_data=True)

    inputs 
    res_num: index of resonator in question
    iAtten: index of attenuation in question
    angle: angle of rotation about the origin (radians)
    showFrames: pops up a window of the frame plotted using matplotlib.plot
    '''     
    xWidth= mlDict['xWidth'] # 
    resWidth = mlDict['resWidth']
    assert resWidth<=xWidth, 'res width must be <= xWidth'

    attenWinAbove = mlDict['attenWinAbove']
    attenWinBelow = mlDict['attenWinBelow']

    #xCenter = self.get_peak_idx(res_num,iAtten,dataObj)
    nFreqPoints = len(dataObj.Is[res_num,0,:])
    assert resWidth<=nFreqPoints 'res width must be <= number of freq steps'

    iq_vels = dataObj.iq_vels[res_num, :, :]
    Is = dataObj.Is[res_num,:, :-1] #-1 to make size the same as iq vels
    Qs = dataObj.Qs[res_num,:, :-1]

        

    # plt.plot(self.Is[res_num,iAtten], self.Qs[res_num,iAtten])
    # plt.show()
    # for spectra where the peak is close enough to the edge that some points falls across the bounadry, pad zeros

    
    magsdb = 10*np.log10(Is**2+Qs**2)

    if center_loop:
        Is = np.transpose(np.transpose(Is) - np.mean(Is,1))
        #print 'Is shape', np.shape(Is)
        #print 'mean shape', np.shape(np.mean(Qs,1))
        Qs = np.transpose(np.transpose(Qs) - np.mean(Qs,1))
        iq_vels = np.transpose(np.transpose(iq_vels) - np.mean(iq_vels,1)) #added by NF 20180423
    #iq_vels = np.round(iq_vels * xWidth / max(dataObj.iq_vels[res_num, iAtten, :]) )



            # interpolate iq_vels onto a finer grid

    res_mag = np.sqrt(np.amax(Is**2 + Qs**2, axis=1)) #changed by NF 20180423 (originally amax)
    #res_mag = np.sqrt(np.mean(Is**2 + Qs**2, axis=1)) #changed by NF 20180423
    Is = np.transpose(np.transpose(Is) / res_mag)
    Qs = np.transpose(np.transpose(Qs) / res_mag)
    #iq_vels = np.transpose(np.transpose(iq_vels)/np.sqrt(np.mean(iq_vels**2,axis=1))) #added by NF 20180423
    iq_vels = np.transpose(np.transpose(iq_vels) / np.amax(iq_vels, axis=1)) #changed by NF 20180423 (originally amax)

    # Is = Is /np.amax(dataObj.iq_vels[res_num, :, :])
    # Qs = Qs /np.amax(dataObj.iq_vels[res_num, :, :])

    # Is = Is /np.amax(dataObj.Is[res_num, :, :])
    # Qs = Qs /np.amax(dataObj.Qs[res_num, :, :])

    if showFrames:
        fig = plt.figure(frameon=False,figsize=(15.0, 5.0))
        fig.add_subplot(131)
        plt.plot(iq_vels)
        plt.ylim(0,1)
        fig.add_subplot(132)
        plt.plot(Is)
        plt.plot(Qs)
        fig.add_subplot(133)
        plt.plot(Is,Qs)
        plt.show()
        plt.close()
    
        
    #make sliding window images
    singleFrameImage = np.zeros((np.shape(Is)[0],resWidth,3))

    if resWidth < nFreqPoints:
        initWinCenter = nFreqPoints/2#np.argmin(magsdb[wsAttenInd,])
        winCenter = initWinCenter
        startWin = int(winCenter-resWidth/2.)
        endWin = int(winCenter+resWidth/2.)
        singleFrameImage[wsAttenInd, :, 0] = Is[wsAttenInd, startWin:endWin]
        singleFrameImage[wsAttenInd, :, 1] = Qs[wsAttenInd, startWin:endWin]
        singleFrameImage[wsAttenInd, :, 2] = iq_vels[wsAttenInd, startWin:endWin]
        
        for i in range(wsAttenInd-1, -1, -1):
            oldWinMags = magsdb[i, startWin:endWin]
            newWinCenter = startWin + np.argmin(oldWinMags)
            startWin += (newWinCenter - winCenter)
            endWin += (newWinCenter - winCenter)
            winCenter = newWinCenter
            singleFrameImage[wsAttenInd, :, 0] = Is[wsAttenInd, startWin:endWin]
            singleFrameImage[wsAttenInd, :, 1] = Qs[wsAttenInd, startWin:endWin]
            singleFrameImage[wsAttenInd, :, 2] = iq_vels[wsAttenInd, startWin:endWin]

        winCenter = initWinCenter
        for i in range(wsAttenInd+1, Is.shape[0]): 
            oldWinMags = magsdb[i, startWin:endWin]
            newWinCenter = startWin + np.argmin(oldWinMags)
            startWin += (newWinCenter - winCenter)
            endWin += (newWinCenter - winCenter)
            winCenter = newWinCenter
            singleFrameImage[wsAttenInd, :, 0] = Is[wsAttenInd, startWin:endWin]
            singleFrameImage[wsAttenInd, :, 1] = Qs[wsAttenInd, startWin:endWin]
            singleFrameImage[wsAttenInd, :, 2] = iq_vels[wsAttenInd, startWin:endWin]
            

    else:
        singleFrameImage[:,:,0] = Is
        singleFrameImage[:,:,1] = Qs
        singleFrameImage[:,:,2] = iq_vels

    if resWidth < xWidth:
        nPadVals = (xWidth - nFreqPoints)/2.
        singleFrameImage[:,:,2] = np.pad(singleFrameImage[:,:,2], [(0,0),(int(np.ceil(nPadVals)), int(np.floor(nPadVals)+1))], 'edge')
        singleFrameImage[:,:,0] = np.pad(singleFrameImage[:,:,0], [(0,0),(int(np.ceil(nPadVals)), int(np.floor(nPadVals)))], 'edge')
        singleFrameImage[:,:,1] = np.pad(singleFrameImage[:,:,1], [(0,0),(int(np.ceil(nPadVals)), int(np.floor(nPadVals)))], 'edge')
        freqs = np.pad(dataObj.freqs[res_num], (int(np.ceil(nPadVals)), int(np.floor(nPadVals))), 'edge')
        

    #if resWidth >= xWidth:
    #    xCenter = nFreqPoints/2
    #    start = xCenter - int(np.ceil(mlDict['xWidth']/2.))
    #    end = xCenter + int(np.floor(mlDict['xWidth']/2.))
    #    iq_vels = dataObj.iq_vels[res_num, :, start:end]
    #    Is = dataObj.Is[res_num,:,start:end]
    #    Qs = dataObj.Qs[res_num,:,start:end]
    #    freqs = dataObj.freqs[res_num]


    if not padFreq is None:
        padFreqInd = np.argmin(np.abs(freqs-padFreq))
        print np.shape(singleFrameImage)
        padResWidth = 20
        print 'makeresimg: padFreqInd', padFreqInd
        if padFreqInd > xWidth/2:
            print 'makeresimg: topCut'
            print 'freqCutRange', freqs[padFreqInd-padResWidth], freqs[-1]
            singleFrameImage[:, padFreqInd-padResWidth:-1, 0] = np.transpose(np.tile(Is[:, padFreqInd-padResWidth], (np.shape(Is)[1]-(padFreqInd-padResWidth)-1,1)))
            singleFrameImage[:, padFreqInd-padResWidth:-1, 1] = np.transpose(np.tile(Qs[:, padFreqInd-padResWidth], (np.shape(Is)[1]-(padFreqInd-padResWidth)-1,1)))
            singleFrameImage[:, padFreqInd-padResWidth:-1, 2] = np.transpose(np.tile(np.zeros(np.shape(iq_vels)[0]), (np.shape(Is)[1]-(padFreqInd-padResWidth)-1,1)))
            #dataObj.iq_vels[res_num, :, padFreqInd-padResWidth:-1] = np.transpose(np.tile(np.zeros(np.shape(iq_vels)[0]), (np.shape(Is)[1]-(padFreqInd-padResWidth)-1,1)))

        else:
            print 'bottomCut'
            print 'freqCutRange', freqs[0], freqs[padFreqInd+padResWidth]
            singleFrameImage[:, 0:padFreqInd+padResWidth, 0] = np.transpose(np.tile(Is[:, padFreqInd+padResWidth], (padFreqInd+padResWidth,1)))
            singleFrameImage[:, 0:padFreqInd+padResWidth, 1] = np.transpose(np.tile(Qs[:, padFreqInd+padResWidth], (padFreqInd+padResWidth,1)))
            singleFrameImage[:, 0:padFreqInd+padResWidth, 2] = np.transpose(np.tile(np.zeros(np.shape(iq_vels)[0]), (padFreqInd+padResWidth,1)))
            #dataObj.iq_vels[res_num:, :, 0:padFreqInd+padResWidth] = np.transpose(np.tile(np.zeros(np.shape(iq_vels)[0]), (padFreqInd+padResWidth,1)))
            

        

    #for offs in range(self.attenWinBelow):
    #    offsImage = np.roll(singleFrameImage, offs, axis=0)
    #    offsImage[0:offs] = singleFrameImage[0]
    #    image[:,-offs,:,:] = offsImage

    #for offs in range(1,self.attenWinAbove):
    #    offsImage = np.roll(singleFrameImage, -offs, axis=0)
    #    offsImage[-offs:] = singleFrameImage[-1]
    #    image[:,offs,:,:] = offsImage
    
    #image = image.flatten()
    # image = np.append(Is,Qs,axis=0)

    #print np.shape(image)

    return singleFrameImage

def get_peak_idx(res_num,iAtten,dataObj,smooth=False, cutType=None, padInd=None):
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

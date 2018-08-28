import numpy as np
from PSFitMLData import *

def makeResImage(res_num, angle=0, center_loop=False,  phase_normalise=False, showFrames=False, test_if_noisy=False, dataObj=None, padFreq=None, mlDict=None):
    '''Creates a table with 2 rows, I and Q for makeTrainData(mag_data=True)

    inputs 
    res_num: index of resonator in question
    iAtten: index of attenuation in question
    self.scalexWidth: typical values: 1/2, 1/4, 1/8
                      uses interpolation to put data from an xWidth x xWidth grid to a 
                      (xWidth/scalexWidth) x (xWidth/scalexWidth) grid. This allows the 
                      user to probe the spectrum using a smaller window while utilizing 
                      the higher resolution training data
    angle: angle of rotation about the origin (radians)
    showFrames: pops up a window of the frame plotted using matplotlib.plot
    '''     
    xWidth= mlDict['xWidth'] 
    scalexWidth = mlDict['scaleXWidth']
    attenWinAbove = mlDict['attenWinAbove']
    attenWinBelow = mlDict['attenWinBelow']

    #xCenter = self.get_peak_idx(res_num,iAtten,dataObj)
    nFreqPoints = len(dataObj.Is[res_num,0,:])

    if nFreqPoints >= xWidth:
        xCenter = nFreqPoints/2
        start = xCenter - int(np.ceil(mlDict['xWidth']/2.))
        end = xCenter + int(np.floor(mlDict['xWidth']/2.))
        iq_vels = dataObj.iq_vels[res_num, :, start:end]
        Is = dataObj.Is[res_num,:,start:end]
        Qs = dataObj.Qs[res_num,:,start:end]
        freqs = dataObj.freqs[res_num]

    else:
        nPadVals = (xWidth - nFreqPoints)/2.
        iq_vels = np.pad(dataObj.iq_vels[res_num,:,:], [(0,0),(int(np.ceil(nPadVals)), int(np.floor(nPadVals)+1))], 'edge')
        Is = np.pad(dataObj.Is[res_num,:,:], [(0,0),(int(np.ceil(nPadVals)), int(np.floor(nPadVals)))], 'edge')
        Qs = np.pad(dataObj.Qs[res_num,:,:], [(0,0),(int(np.ceil(nPadVals)), int(np.floor(nPadVals)))], 'edge')
        freqs = np.pad(dataObj.freqs[res_num], (int(np.ceil(nPadVals)), int(np.floor(nPadVals))), 'edge')
        

    # plt.plot(self.Is[res_num,iAtten], self.Qs[res_num,iAtten])
    # plt.show()
    # for spectra where the peak is close enough to the edge that some points falls across the bounadry, pad zeros

    

    if center_loop:
        Is = np.transpose(np.transpose(Is) - np.mean(Is,1))
        #print 'Is shape', np.shape(Is)
        #print 'mean shape', np.shape(np.mean(Qs,1))
        Qs = np.transpose(np.transpose(Qs) - np.mean(Qs,1))
        iq_vels = np.transpose(np.transpose(iq_vels) - np.mean(iq_vels,1)) #added by NF 20180423
    #iq_vels = np.round(iq_vels * xWidth / max(dataObj.iq_vels[res_num, iAtten, :]) )



            # interpolate iq_vels onto a finer grid


    # if test_if_noisy:
    #     peak_iqv = mean(iq_vels[int(xWidth/4): int(3*xWidth/4)])
    #     nonpeak_indicies=np.delete(np.arange(xWidth),np.arange(int(xWidth/4),int(3*xWidth/4)))
    #     nonpeak_iqv = iq_vels[nonpeak_indicies]
    #     nonpeak_iqv = mean(nonpeak_iqv[np.where(nonpeak_iqv!=0)]) # since it spans a larger area
    #     noise_condition = 1.5#0.7 

    #     if (peak_iqv/nonpeak_iqv < noise_condition):
    #         return None 

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

    # print Is[::5]
    # print Qs[::5]

    if phase_normalise: #need to fix for imgcube
        #mags = Qs**2 + Is**2
        #mags = map(lambda x: math.sqrt(x), mags)#map(lambda x,y:x+y, a,b)

        #peak_idx = self.get_peak_idx(res_num,iAtten)
        peak_idx =np.argmax(iq_vels)
        #min_idx = argmin(mags)

        phase_orig = math.atan2(Qs[peak_idx],Is[peak_idx])
        #phase_orig = math.atan2(Qs[min_idx],Is[min_idx])

        angle = -phase_orig

        rotMatrix = numpy.array([[numpy.cos(angle), -numpy.sin(angle)], 
                                 [numpy.sin(angle),  numpy.cos(angle)]])

        Is,Qs = np.dot(rotMatrix,[Is,Qs])

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

    image = np.zeros((np.shape(Is)[0],1+attenWinAbove+attenWinBelow,np.shape(Is)[1],3))
    singleFrameImage = np.zeros((np.shape(Is)[0],np.shape(Is)[1],3))
    singleFrameImage[:,:,0] = Is
    singleFrameImage[:,:,1] = Qs
    singleFrameImage[:,:,2] = iq_vels

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

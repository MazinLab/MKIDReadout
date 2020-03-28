import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import os, sys
import argparse
import tools as mlt

import tools as mlt
from PSFitMLData import PSFitMLData
from mkidcore.corelog import getLogger
import mkidreadout.configuration.sweepdata as sd


class diagnostics():
    def __init__(self, modelDir, psDataFileName=None, metadataFn=None, wsAtten=None, resWidth=None):
        if psDataFileName is not None:
            if psDataFileName.split('.')[1] == 'npz':
                assert os.path.isfile(metadataFn), 'Must resonator metadata file'
                self.sweep = sd.FreqSweep(psDataFileName, metadataFn)
            else:
                raise Exception(psDataFileName + ' is not a valid format')

        if metadataFn is not None:
            self.metadata = sd.SweepMetadata(file=metadataFn)

        self.mlDict, self.sess, self.graph, self.x_input, self.y_output, self.keep_prob, self.is_training = mlt.get_ml_model(modelDir)

    def getImage(self, freq, atten):
        image, _, _  = mlt.makeWPSImageList(self.sweep, freq, atten, self.mlDict['freqWinSize'],
                                        self.mlDict['attenWinAbove']+self.mlDict['attenWinBelow']+1, self.mlDict['useIQV'], 
                                        self.mlDict['useVectIQV'], self.mlDict['normalizeBeforeCenter'], False)[0]

        return image

    def getActivations(self, freq, atten):
        image = self.getImage(freq, atten)
        inferenceLabels = self.sess.run(self.y_output, feed_dict={self.x_input: [image], self.keep_prob: 1, self.is_training: False})
        return inferenceLabels[0]

    def getWeights(self, layer=1):
        nColors = 2
        if self.mlDict['useIQV']:
            nColors += 1
        if self.mlDict['useVectIQV']:
            nColors += 1
        image = np.zeros((self.mlDict['attenWinAbove']+self.mlDict['attenWinBelow']+1, self.mlDict['freqWinSize'], nColors))
        weightTensor = self.graph.get_tensor_by_name('Layer' + str(layer) + '/W_conv' + str(layer) + ':0')
        return np.array(self.sess.run(weightTensor, feed_dict={self.x_input: [image], self.keep_prob: 1, self.is_training: False}))

    
    def plotLayer1Weights(self):
        weights = self.getWeights(1)
        nrows = int(np.floor(np.sqrt(weights.shape[3])))
        ncols = int(np.ceil(np.sqrt(weights.shape[3])))
        vmin = np.min(weights)
        vmax = np.max(weights)
        layerLabels = ['I', 'Q', 'IQV', 'vect IQV']
        for inChan in range(weights.shape[2]):
            inChanWeights = weights[:,:,inChan,:]
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
            axes[0, 0].set_title(layerLabels[inChan])
            for outChan in range(weights.shape[3]):
                y = outChan/ncols
                x = outChan%ncols
                axes[y,x].imshow(inChanWeights[:,:,x+y*ncols], vmin=vmin, vmax=vmax)
            


    def plotLoops(self, res_num, start_atten=0, end_atten=-1, grid=True, show=True):
        image, _, attenList, _, _ = self.getImage(res_num)

        if end_atten==-1:
            end_atten = len(self.inferenceData.attens)
        nrows = 4
        ncols = (end_atten-start_atten)//3

        if grid:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 8))
            a=start_atten
            for y in range(nrows):
                for x in range(ncols):
                    if a < end_atten:
                        #print(a)
                        axes[y, x].plot(image[a, :,0], image[a, :,1])
                        axes[y,x].set_title(a)
                    a+=1
        else:
            plt.figure()
            plt.xlabel('I')
            plt.ylabel('Q')
            for a in range(start_atten, end_atten):
                #print(a)
                plt.plot(image[a, :,0], image[a, :,1], label=str(a))
            plt.legend()

        if show:
            plt.show()

    def plotIQ_vels(self, res_num, start_atten=0, end_atten=-1, grid=True, show=True):
        image, freqCube, attenList, iqv, mags = self.getImage(res_num)

        if end_atten==-1:
            end_atten = len(self.inferenceData.attens)
        nrows = 4
        ncols = (end_atten-start_atten)//3

        if grid:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 8))
            a=start_atten
            for y in range(nrows):
                for x in range(ncols):
                    if a < end_atten:
                        axes[y, x].plot(freqCube[a], iqv[a, :])
                        axes[y,x].set_title(a)
                    a+=1
        else:
            plt.figure()
            for a in range(start_atten, end_atten):
                plt.plot(freqCube[a], iqv[a, :], label=str(a))
            plt.xlabel('Freq')
            plt.ylabel('vIQ')
            plt.legend()
        if show:
            plt.show()

    def plotS21(self, res_num, start_atten=0, end_atten=-1, grid=True, show=True):
        image, freqCube, attenList, iqv, mags = self.getImage(res_num)

        if end_atten==-1:
            end_atten = len(self.inferenceData.attens)
        nrows = 4
        ncols = (end_atten-start_atten)//3

        if grid:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 8))
            a=start_atten
            for y in range(nrows):
                for x in range(ncols):
                    if a < end_atten:
                        #print(a)
                        axes[y, x].plot(freqCube[a], mags[a, :])
                        axes[y,x].set_title(a)
                    a+=1
        else:
            plt.figure()
            for a in range(start_atten, end_atten):
                #print(a)
                plt.plot(freqCube[a], mags[a, :], label=str(a))
                plt.plot(freqCube[a, freqCube.shape[1]/2], mags[a, freqCube.shape[1]/2], '.')
            plt.xlabel('Freq')
            plt.ylabel('S21')
            plt.legend()
        if show:
            plt.show()

    def plotCenterFreq(self, res_num, start_atten=0, end_atten=-1, show=True):
        _, freqCube, _, _, _ = self.getImage(res_num)        
        end_atten = len(self.inferenceData.attens) + end_atten

        plt.figure()
        plt.plot(freqCube[start_atten:end_atten, freqCube.shape[1]/2])
        plt.xlabel('Atten')
        plt.ylabel('Freq')
        if show:
            plt.show()

    def plotActivations(self, res_num, show=True):
        acts = self.getActivations(res_num)
        plt.figure()
        plt.plot(acts)
        plt.title('Activations')
        if show:
            plt.show()

    def plotImageDiagnostics(self, resID, start_atten=0, end_atten=-1):
        res_num = np.where(self.inferenceData.resIDs==resID)[0][0]
        print('Resonator Number', res_num)
        self.plotLoops(res_num, start_atten, end_atten, grid=False, show=False)
        self.plotIQ_vels(res_num, start_atten, end_atten, grid=False, show=False)
        self.plotS21(res_num, start_atten, end_atten, grid=False, show=False)
        self.plotCenterFreq(res_num, show=False)
        self.plotActivations(res_num, show=True)

    def plotMeanImage(self):
        self.plotLoops(-1, grid=False, show=False)
        self.plotIQ_vels(-1, grid=False, show=False)
         

if __name__=='__main__':

    print(sys.argv)

    parser = argparse.ArgumentParser(description='Tool for diagnosing the ML images')
    parser.add_argument('model', help='Directory containing ML model')
    parser.add_argument('-d', '--inferenceData', default=None, help='HDF5 file containing powersweep data')
    parser.add_argument('-m', '--metadata', default=None, help='Directory to save output file')
    #parser.add_argument('-o', '--output-dir', nargs=1, default=[None], help='Directory to save output file')
    parser.add_argument('--weights', action='store_true', help='Plot Layer 1 filters')
    parser.add_argument('--mean', action='store_true', help='Plot Layer 1 filters')
    parser.add_argument('-w', '--ws-atten', type=float, default=None, help='Attenuation where peak finding code was run')
    parser.add_argument('-b', '--badscore-model', default=None, help='ML config file for bad score model')
    parser.add_argument('-r', '--resid', type=int, default=60000)
    parser.add_argument('-a', '--atten-range', nargs=2, type=int, default=[0,-1])
    parser.add_argument('--res-width', type=int, default=None, 
                        help='Width of window (in units nFreqStep) to use for power/freq classification')
    args = parser.parse_args()
    getLogger(__name__, setup=True)

    # print(args)

    psDataFileName=args.inferenceData
    print psDataFileName

    diag = diagnostics(args.model, psDataFileName, args.metadata, args.ws_atten, args.res_width)
    # diag.plotLoops(105)
    # diag.plotIQ_vels(105)
    # diag.plotS21(105)
    if args.weights:
        diag.plotLayer1Weights()
        plt.show()
    elif args.mean:
        diag.plotMeanImage()
        plt.show()
    else:
        diag.plotImageDiagnostics(args.resid, args.atten_range[0], args.atten_range[1])

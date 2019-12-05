import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import os, sys
import argparse
from PSFitMLData import PSFitMLData
import tools as mlt

from mkidreadout.configuration.powersweep.psmldata import MLData
from mkidcore.corelog import getLogger

class diagnostics():
    def __init__(self, modelDir, psDataFileName=None, metadataFn=None, wsAtten=None, resWidth=None):
        if psDataFileName is not None:
            if psDataFileName.split('.')[1] == 'h5':
                self.inferenceData = PSFitMLData(h5File=psDataFileName, useAllAttens=False, useResID=True)
            elif psDataFileName.split('.')[1] == 'npz':
                assert os.path.isfile(metadataFn), 'Must resonator metadata file'
                self.inferenceData = MLData(psDataFileName, metadataFn)
            else:
                raise Exception(psDataFileName + ' is not a valid format')

        # if mlDict['scaleXWidth']!= 1:
        #     mlDict['xWidth']=mlDict['xWidth']*mlDict['scaleXWidth'] #reset ready for get_PS_data

            total_res_nums = np.shape(self.inferenceData.freqs)[0]
            res_nums = total_res_nums
            span = range(res_nums)
            self.inferenceData.opt_attens = np.zeros((res_nums))
            self.inferenceData.opt_freqs = np.zeros((res_nums))
            self.inferenceData.scores = np.zeros((res_nums))

        self.mlDict, self.sess, self.graph, self.x_input, self.y_output, self.keep_prob, self.is_training = mlt.get_ml_model(modelDir)
        inferenceLabels = np.zeros((res_nums, self.mlDict['nAttens']))
        #print [n.name for n in tf.get_default_graph().as_graph_def().node]
        self.meanImage = tf.get_collection('meanTrainImage')[0].eval(session=self.sess)
        if not self.mlDict.has_key('useVectIQV'):
            self.mlDict['useVectIQV'] = False


        if wsAtten is None:
            wsAtten = self.mlDict['wsAtten']
            getLogger(__name__).warning('No WS atten specified; using value of ', str(wsAtten), 
                                ' from training config')
        else:
            self.wsAtten = wsAtten

        if psDataFileName is not None:
            self.wsAttenInd = np.argmin(np.abs(self.inferenceData.attens - wsAtten))

        if resWidth is None:
            self.resWidth = self.mlDict['resWidth']
        else:
            self.resWidth = resWidth

       
        #if mlBadDict is not None:
        #    modelBadPath = os.path.join(mlBadDict['modelDir'], mlBadDict['modelName']) + '.meta'
        #    print 'Loading badscore model from', modelBadPath

        #    sess_bad = tf.Session()
        #    saver_bad = tf.train.import_meta_graph(modelBadPath)
        #    saver_bad.restore(sess_bad, tf.train.latest_checkpoint(mlBadDict['modelDir']))

        #    graph = tf.get_default_graph()
        #    x_input_bad = graph.get_tensor_by_name('inputImage:0')
        #    y_output_bad = graph.get_tensor_by_name('outputLabel:0')
        #    keep_prob_bad = graph.get_tensor_by_name('keepProb:0')
        #    useBadScores = True

        #else:
        #    useBadScores = False

    def getImage(self, res_num):
        if res_num==-1:
            iqv = None
            mags = None
            if self.mlDict['useIQV']:
                iqv = self.meanImage[:,:,2]
            if self.mlDict['useMag']:
                mags = self.meanImage[:,:,3]
                
            return self.meanImage, np.tile(np.arange(self.meanImage.shape[1]), (self.meanImage.shape[0],1)), np.arange(self.meanImage.shape[0]), iqv, mags
        else: 
            image, freqCube, attenList, iqv, mags = mlt.makeResImage(res_num, self.inferenceData, self.wsAttenInd, self.mlDict['xWidth'],
                                            self.resWidth, self.mlDict['padResWin'], self.mlDict['useIQV'], self.mlDict['useMag'],
                                            self.mlDict['centerLoop'], self.mlDict['nAttens'], self.mlDict['useVectIQV'])
            #image -= self.meanImage

        return image, freqCube, attenList, iqv, mags

    def getActivations(self, res_num):
        image, _, _, _, _ = self.getImage(res_num)
        inferenceLabels = np.zeros((self.mlDict['nAttens'], 2))
        for i in range(self.mlDict['attenWinBelow'], self.mlDict['nAttens'] - self.mlDict['attenWinAbove']):
            inferenceImage = image[i-self.mlDict['attenWinBelow']: i+self.mlDict['attenWinAbove']+1]
            inferenceLabels[i, :] = self.sess.run(self.y_output, feed_dict={self.x_input: [inferenceImage], self.keep_prob: 1, self.is_training: False})
        return inferenceLabels

    def getWeights(self, layer=1):
        image, _, _, _, _ = self.getImage(0)
        weightTensor = self.graph.get_tensor_by_name('Layer' + str(layer) + '/W_conv' + str(layer) + ':0')
        return np.array(self.sess.run(weightTensor, feed_dict={self.x_input: [image[0:1+self.mlDict['attenWinAbove']+self.mlDict['attenWinBelow']]], 
                            self.keep_prob: 1, self.is_training: False}))

    
    def plotLayer1Weights(self):
        weights = self.getWeights(1)
        nrows = int(np.floor(np.sqrt(weights.shape[3])))
        ncols = int(np.ceil(np.sqrt(weights.shape[3])))
        for inChan in range(weights.shape[2]):
            inChanWeights = weights[:,:,inChan,:]
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
            axes[0, 0].set_title(inChan)
            for outChan in range(weights.shape[3]):
                y = outChan/ncols
                x = outChan%ncols
                axes[y,x].imshow(inChanWeights[:,:,x+y*ncols])

    def plotLayer1WeightSum(self):
        weights = self.getWeights(1)
        for inChan in range(weights.shape[2]):
            inChanWeights = weights[:,:,inChan,:]
            fig, ax = plt.subplots()
            ax.set_title(inChan)
            ax.imshow(np.sum(inChanWeights, axis=2))

            


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

    def plotActivations(self, res_num, manAtten=None, show=True):
        acts = self.getActivations(res_num)
        plt.figure()
        plt.plot(acts)
        if manAtten:
            plt.plot(manAtten, acts[manAtten, 0], '.')
        plt.title('Activations')
        if show:
            plt.show()

    def plotImageDiagnostics(self, resID, start_atten=0, end_atten=-1):
        res_num = np.where(self.inferenceData.resIDs==resID)[0][0]
        manAtten = self.inferenceData.opt_iAttens[res_num]
        print('Resonator Number', res_num)
        self.plotLoops(res_num, start_atten, end_atten, grid=False, show=False)
        self.plotIQ_vels(res_num, start_atten, end_atten, grid=False, show=False)
        self.plotS21(res_num, start_atten, end_atten, grid=False, show=False)
        self.plotCenterFreq(res_num, show=False)
        self.plotActivations(res_num, manAtten, show=True)

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
        diag.plotLayer1WeightSum()
        plt.show()
    elif args.mean:
        diag.plotMeanImage()
        plt.show()
    else:
        diag.plotImageDiagnostics(args.resid, args.atten_range[0], args.atten_range[1])

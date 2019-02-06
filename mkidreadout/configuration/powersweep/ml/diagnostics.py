import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import os, sys
import argparse
from PSFitMLData import PSFitMLData
import tools as mlt

from mkidreadout.utils.readDict import readDict
from mkidreadout.configuration.powersweep.psmldata import MLData

class diagnostics():
    def __init__(self, modelDir, psDataFileName, metadataFn=None, wsAtten=None, resWidth=None):
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

        self.mlDict, self.sess, self.graph, self.x_input, self.y_output, self.keep_prob = mlt.get_ml_model(modelDir)

        self.inferenceData.opt_attens = np.zeros((res_nums))
        self.inferenceData.opt_freqs = np.zeros((res_nums))
        self.inferenceData.scores = np.zeros((res_nums))
        inferenceLabels = np.zeros((res_nums, self.mlDict['nAttens']))

        if wsAtten is None:
            wsAtten = self.mlDict['wsAtten']
            getLogger(__name__).warning('No WS atten specified; using value of ', str(wsAtten), 
                                ' from training config')
        else:
            self.wsAtten = wsAtten

        self.wsAttenInd = np.argmin(np.abs(self.inferenceData.attens - wsAtten))
        print self.wsAttenInd

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
        image, freqCube, attenList = mlt.makeResImage(res_num, self.inferenceData, self.wsAttenInd, self.mlDict['xWidth'],
                                        self.resWidth, self.mlDict['padResWin'], self.mlDict['useIQV'], self.mlDict['useMag'],
                                        self.mlDict['centerLoop'], self.mlDict['nAttens'])

        return image, freqCube, attenList

    def getActivations(self, res_num):
        image, _, _ = self.getImage(res_num)
        inferenceLabels = self.sess.run(self.y_output, feed_dict={self.x_input: [image], self.keep_prob: 1})
        return inferenceLabels[0]



    def plotLoops(self, res_num, start_atten=0, end_atten=-1, grid=True, show=True):
        image, _, attenList = self.getImage(res_num)

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
        image, freqCube, attenList = self.getImage(res_num)

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
                        axes[y, x].plot(freqCube[a], image[a, :,2])
                        axes[y,x].set_title(a)
                    a+=1
        else:
            plt.figure()
            for a in range(start_atten, end_atten):
                plt.plot(freqCube[a], image[a, :,2], label=str(a))
            plt.xlabel('Freq')
            plt.ylabel('vIQ')
            plt.legend()
        if show:
            plt.show()

    def plotS21(self, res_num, start_atten=0, end_atten=-1, grid=True, show=True):
        image, freqCube, attenList = self.getImage(res_num)

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
                        axes[y, x].plot(freqCube[a], image[a, :,3])
                        axes[y,x].set_title(a)
                    a+=1
        else:
            plt.figure()
            for a in range(start_atten, end_atten):
                #print(a)
                plt.plot(freqCube[a], image[a, :,3], label=str(a))
                plt.plot(freqCube[a, freqCube.shape[1]/2], image[a, freqCube.shape[1]/2, 3], '.')
            plt.xlabel('Freq')
            plt.ylabel('S21')
            plt.legend()
        if show:
            plt.show()

    def plotCenterFreq(self, res_num, start_atten=0, end_atten=-1, show=True):
        _, freqCube, _ = self.getImage(res_num)        
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

if __name__=='__main__':

    print(sys.argv)

    parser = argparse.ArgumentParser(description='Tool for diagnosing the ML images')
    parser.add_argument('model', help='Directory containing ML model')
    parser.add_argument('inferenceData', help='HDF5 file containing powersweep data')
    parser.add_argument('-m', '--metadata', default=None, help='Directory to save output file')
    #parser.add_argument('-o', '--output-dir', nargs=1, default=[None], help='Directory to save output file')
    parser.add_argument('-s', '--add-scores', action='store_true', help='Adds a score column to the output file')
    parser.add_argument('-w', '--ws-atten', type=float, default=None, help='Attenuation where peak finding code was run')
    parser.add_argument('-b', '--badscore-model', default=None, help='ML config file for bad score model')
    parser.add_argument('-r', '--resid', type=int, default=60000)
    parser.add_argument('-a', '--atten-range', nargs=2, type=int, default=[0,-1])
    parser.add_argument('--res-width', type=int, default=None, 
                        help='Width of window (in units nFreqStep) to use for power/freq classification')
    args = parser.parse_args()

    # print(args)

    psDataFileName=args.inferenceData
    if not os.path.isfile(psDataFileName):
        psDataFileName = os.path.join(os.environ['MKID_DATA_DIR'], psDataFileName)

    diag = diagnostics(args.model, psDataFileName, args.metadata, args.ws_atten, args.res_width)
    # diag.plotLoops(105)
    # diag.plotIQ_vels(105)
    # diag.plotS21(105)
    diag.plotImageDiagnostics(args.resid, args.atten_range[0], args.atten_range[1])

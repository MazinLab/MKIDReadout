import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import os, sys
import argparse
from PSFitMLData import PSFitMLData
import PSFitMLTools as mlt

from mkidreadout.utils.readDict import readDict
from mkidreadout.configuration.powersweep.psmldata import MLData

class diagnostics():
    def __init__(self, mlDict, mlBadDict, psDataFileName, metadataFn=None, saveScores=False, wsAtten=None):
        if psDataFileName.split('.')[1] == 'h5':
            self.inferenceData = PSFitMLData(h5File=psDataFileName, useAllAttens=False, useResID=True)
        elif psDataFileName.split('.')[1] == 'npz':
            assert os.path.isfile(metadataFn), 'Must resonator metadata file'
            self.inferenceData = MLData(psDataFileName, metadataFn)

        # if mlDict['scaleXWidth']!= 1:
        #     mlDict['xWidth']=mlDict['xWidth']*mlDict['scaleXWidth'] #reset ready for get_PS_data

        total_res_nums = np.shape(self.inferenceData.freqs)[0]
        res_nums = total_res_nums
        span = range(res_nums)

        self.inferenceData.opt_attens = np.zeros((res_nums))
        self.inferenceData.opt_freqs = np.zeros((res_nums))
        self.inferenceData.scores = np.zeros((res_nums))
        self.wsAttenInd = np.argmin(np.abs(self.inferenceData.attens - wsAtten))

        print 'inferenceAttens', self.inferenceData.attens

        inferenceLabels = np.zeros((res_nums, mlDict['nAttens']))

        modelPath = os.path.join(mlDict['modelDir'], mlDict['modelName']) + '.meta'
        print 'Loading model from', modelPath

        # sess = tf.Session()
        # saver = tf.train.import_meta_graph(modelPath)
        # saver.restore(sess, tf.train.latest_checkpoint(mlDict['modelDir']))
        #
        # graph = tf.get_default_graph()
        # x_input = graph.get_tensor_by_name('inputImage:0')
        # y_output = graph.get_tensor_by_name('outputLabel:0')
        # keep_prob = graph.get_tensor_by_name('keepProb:0')

        if mlBadDict is not None:
            modelBadPath = os.path.join(mlBadDict['modelDir'], mlBadDict['modelName']) + '.meta'
            print 'Loading badscore model from', modelBadPath

            sess_bad = tf.Session()
            saver_bad = tf.train.import_meta_graph(modelBadPath)
            saver_bad.restore(sess_bad, tf.train.latest_checkpoint(mlBadDict['modelDir']))

            graph = tf.get_default_graph()
            x_input_bad = graph.get_tensor_by_name('inputImage:0')
            y_output_bad = graph.get_tensor_by_name('outputLabel:0')
            keep_prob_bad = graph.get_tensor_by_name('keepProb:0')
            useBadScores = True

        else:
            useBadScores = False

    def getImage(self, res_num):
        image, freqCube, attenList = mlt.makeResImage(res_num=res_num, center_loop=mlDict['center_loop'],
                                                      phase_normalise=False, showFrames=False,
                                                      dataObj=self.inferenceData, mlDict=mlDict, wsAttenInd=self.wsAttenInd)

        print(attenList, freqCube.shape)
        return image, freqCube, attenList


    def plotLoops(self, res_num, start_atten=0, end_atten=-1, grid=True, show=True):
        image, _, attenList = self.getImage(res_num)

        end_atten = len(self.inferenceData.attens) + end_atten
        print(self.inferenceData.attens.shape,end_atten)
        nrows = 4
        ncols = (end_atten-start_atten)//3

        if grid:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 8))
            a=start_atten
            for y in range(nrows):
                for x in range(ncols):
                    if a < end_atten:
                        print(a)
                        axes[y, x].plot(image[a, :,0], image[a, :,1])
                        axes[y,x].set_title(a)
                    a+=1
        else:
            plt.figure()
            plt.xlabel('I')
            plt.ylabel('Q')
            for a in range(start_atten, end_atten):
                print(a)
                plt.plot(image[a, :,0], image[a, :,1])

        if show:
            plt.show()

    def plotIQ_vels(self, res_num, start_atten=0, end_atten=-1, grid=True, show=True):
        image, freqCube, attenList = self.getImage(res_num)

        end_atten = len(self.inferenceData.attens) + end_atten
        print(self.inferenceData.attens.shape,end_atten)
        nrows = 4
        ncols = (end_atten-start_atten)//3

        if grid:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 8))
            a=start_atten
            for y in range(nrows):
                for x in range(ncols):
                    if a < end_atten:
                        print(a)
                        axes[y, x].plot(freqCube[a], image[a, :,2])
                        axes[y,x].set_title(a)
                    a+=1
        else:
            plt.figure()
            for a in range(start_atten, end_atten):
                print(a)
                plt.plot(freqCube[a], image[a, :,2])
            plt.xlabel('Freq')
            plt.ylabel('vIQ')
        if show:
            plt.show()

    def plotS21(self, res_num, start_atten=0, end_atten=-1, grid=True, show=True):
        image, freqCube, attenList = self.getImage(res_num)

        end_atten = len(self.inferenceData.attens) + end_atten
        print(self.inferenceData.attens.shape,end_atten)
        nrows = 4
        ncols = (end_atten-start_atten)//3

        if grid:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 8))
            a=start_atten
            for y in range(nrows):
                for x in range(ncols):
                    if a < end_atten:
                        print(a)
                        axes[y, x].plot(freqCube[a], image[a, :,3])
                        axes[y,x].set_title(a)
                    a+=1
        else:
            plt.figure()
            for a in range(start_atten, end_atten):
                print(a)
                plt.plot(freqCube[a], image[a, :,3])
            plt.xlabel('Freq')
            plt.ylabel('S21')
        if show:
            plt.show()

    def plotStartFreq(self, res_num, start_atten=0, end_atten=-1, show=True):
        _, freqCube, _ = self.getImage(res_num)
        print freqCube.shape
        end_atten = len(self.inferenceData.attens) + end_atten

        plt.figure()
        plt.plot(freqCube[start_atten:end_atten, 0])
        plt.xlabel('Atten')
        plt.ylabel('Freq')
        if show:
            plt.show()

    def plotImageDiagnostics(self, res_num):
        print('Resonator Number', res_num)
        self.plotLoops(res_num, grid=False, show=False)
        self.plotIQ_vels(res_num, grid=False, show=False)
        self.plotS21(res_num, grid=False, show=False)
        self.plotStartFreq(res_num, show=True)

if __name__=='__main__':

    print(sys.argv)

    parser = argparse.ArgumentParser(description='Tool for diagnosing the ML images')
    parser.add_argument('mlConfig', nargs=1, help='Machine learning model config file')
    parser.add_argument('inferenceData', nargs=1, help='HDF5 file containing powersweep data')
    parser.add_argument('-m', '--metadata', nargs=1, default=[None], help='Directory to save output file')
    #parser.add_argument('-o', '--output-dir', nargs=1, default=[None], help='Directory to save output file')
    parser.add_argument('-s', '--add-scores', action='store_true', help='Adds a score column to the output file')
    parser.add_argument('-w', '--ws-atten', nargs=1, type=float, default=[None], help='Attenuation where peak finding code was run')
    parser.add_argument('-b', '--badscore-model', nargs=1, default=[None], help='ML config file for bad score model')
    args = parser.parse_args()

    # print(args)

    mlDict = readDict()
    mlDict.readFromFile(args.mlConfig[0])
    if args.badscore_model[0] is not None:
        mlBadDict = readDict()
        mlBadDict.readFromFile(args.badscore_model[0])
    else:
        mlBadDict = None

    wsAtten = args.ws_atten[0]
    metadataFn = args.metadata[0]

    psDataFileName=args.inferenceData[0]
    if not os.path.isfile(psDataFileName):
        psDataFileName = os.path.join(os.environ['MKID_DATA_DIR'], psDataFileName)

    diag = diagnostics(mlDict, mlBadDict, psDataFileName, metadataFn, args.add_scores, wsAtten)
    # diag.plotLoops(105)
    # diag.plotIQ_vels(105)
    # diag.plotS21(105)
    diag.plotImageDiagnostics(107)
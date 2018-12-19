"""
Script to infer powers from trained model. Saves results
as frequency file in $MKID_DATA_DIR.
Usage: python findPowers.py <mlConfigFile> <h5File>
    mlConfigFile - cfg file specifying which ML model to use.
        Model must already be trained.
    h5File - .h5 file containing power sweep data used to
        infer powers.

"""
import numpy as np
import tensorflow as tf
import os, sys
import argparse
from mkidreadout.configuration.powersweep.ml.PSFitMLData import PSFitMLData
import mkidreadout.configuration.powersweep.ml.PSFitMLTools as mlt
from mkidreadout.utils.readDict import readDict
from mkidreadout.configuration.powersweep.psmldata import MLData
from mkidcore.corelog import getLogger

FREQ_USE_MAG = False


def findPowers(mlDict, mlBadDict, psDataFileName, metadataFn=None,
               saveScores=False, wsAtten=None):
    if psDataFileName.split('.')[1] == 'h5':
        inferenceData = PSFitMLData(h5File=psDataFileName, useAllAttens=False, useResID=True)
        inferenceData.wsatten = wsAtten
    else:
        assert os.path.isfile(metadataFn), 'Must resonator metadata file'
        inferenceData = MLData(psDataFileName, metadataFn)

    modelPath = os.path.join(mlDict['modelDir'], mlDict['modelName']) + '.meta'

    modelBadPath = os.path.join(mlBadDict['modelDir'], mlBadDict['modelName']) + '.meta'

    mlArgs = dict(xWidth = mlDict['xWidth'], resWidth = mlDict['resWidth'], pad_res_win = mlDict['padResWin'],
                  useIQV = mlDict['useIQV'], useMag = mlDict['useMag'], mlDictnAttens = mlDict['nAttens'])

    apply_ml_model(inferenceData, wsAtten, mlDict['nAttens'], mlArgs=mlArgs,
                   goodModel=modelPath, badModel=modelBadPath, center_loop=mlDict['center_loop'])

    if psDataFileName.split('.')[1] == 'h5':
        inferenceData.savePSTxtFile(flag='_' + mlDict['modelName'], outputFN=None, saveScores=saveScores)
    elif psDataFileName.split('.')[1] == 'npz':
        inferenceData.saveInferenceData()


def apply_ml_model(inferenceData, wsAtten, modelNatten, goodModel='', badModel='', center_loop=True,
                   mlArgs={}):
    """
    Uses Trained model, specified by mlDict, to infer powers from a powersweep
    saved in psDataFileName. Saves results in .txt file in $MKID_DATA_DIR
    """

    total_res_nums = np.shape(inferenceData.freqs)[0]
    res_nums = total_res_nums
    span = range(res_nums)

    inferenceData.opt_attens = np.zeros((res_nums))
    inferenceData.opt_freqs = np.zeros((res_nums))
    inferenceData.scores = np.zeros((res_nums))
    wsAttenInd = np.argmin(np.abs(inferenceData.attens - wsAtten))

    getLogger(__name__).debug("Inference attens: {}".format(inferenceData.attens))

    inferenceLabels = np.zeros((res_nums, modelNatten))

    getLogger(__name__).info('Loading good model from %s', goodModel)
    sess = tf.Session()
    saver = tf.train.import_meta_graph(goodModel)
    saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(goodModel)))

    graph = tf.get_default_graph()
    x_input = graph.get_tensor_by_name('inputImage:0')
    y_output = graph.get_tensor_by_name('outputLabel:0')
    keep_prob = graph.get_tensor_by_name('keepProb:0')

    if badModel:
        getLogger(__name__).info('Loading good model from %s', badModel)
        sess_bad = tf.Session()
        saver_bad = tf.train.import_meta_graph(badModel)
        saver_bad.restore(sess_bad, tf.train.latest_checkpoint(os.path.dirname(badModel)))

        graph = tf.get_default_graph()
        x_input_bad = graph.get_tensor_by_name('inputImage:0')
        y_output_bad = graph.get_tensor_by_name('outputLabel:0')
        keep_prob_bad = graph.get_tensor_by_name('keepProb:0')

    getLogger(__name__).debug('Using trained algorithm on images on each resonator')

    doubleCounter = 0
    for i, rn in enumerate(span):
        getLogger(__name__).debug("%d of %i" % (i + 1, res_nums))

        image, freqCube, attenList = mlt.makeResImage(res_num=rn, center_loop=center_loop,
                                                      phase_normalise=False, showFrames=False, dataObj=inferenceData,
                                                      wsAttenInd=wsAttenInd,**mlArgs)

        inferenceImage = []
        inferenceImage.append(image)  # inferenceImage is just reformatted image
        inferenceLabels[rn, :] = sess.run(y_output, feed_dict={x_input: inferenceImage, keep_prob: 1})
        iAtt = np.argmax(inferenceLabels[rn, :])
        inferenceData.opt_attens[rn] = attenList[iAtt]
        if FREQ_USE_MAG:
            inferenceData.opt_freqs[rn] = freqCube[iAtt, np.argmin(image[iAtt, :, 3])]  # TODO: make this more robust
        else:
            inferenceData.opt_freqs[rn] = freqCube[
                iAtt, np.argmax(np.correlate(image[iAtt, :, 2], np.ones(5), 'same'))]  # TODO: make this more robust

        assert inferenceData.freqs[rn, 0] <= inferenceData.opt_freqs[rn] <= inferenceData.freqs[
            rn, -1], 'freq out of range, need to debug'

        inferenceData.scores[rn] = inferenceLabels[rn, iAtt]

        if rn > 0 and np.abs(inferenceData.opt_freqs[rn] - inferenceData.opt_freqs[rn - 1]) < 100.e3:
            doubleCounter += 1

        if badModel:
            badInferenceLabels = sess_bad.run(y_output_bad, feed_dict={x_input_bad: inferenceImage, keep_prob_bad: 1})
            inferenceData.bad_scores[rn] = badInferenceLabels.max()

    getLogger(__name__).info('Had {} doubles'.format(doubleCounter))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Inference Script')
    parser.add_argument('mlConfig', nargs=1, help='Machine learning model config file')
    parser.add_argument('inferenceData', nargs=1, help='HDF5 file containing powersweep data')
    parser.add_argument('-m', '--metadata', nargs=1, default=[None], help='Directory to save output file')
    # parser.add_argument('-o', '--output-dir', nargs=1, default=[None], help='Directory to save output file')
    parser.add_argument('-s', '--add-scores', action='store_true', help='Adds a score column to the output file')
    parser.add_argument('-w', '--ws-atten', nargs=1, type=float, default=[None],
                        help='Attenuation where peak finding code was run')
    parser.add_argument('-b', '--badscore-model', nargs=1, default=[None], help='ML config file for bad score model')
    args = parser.parse_args()

    mlDict = readDict()
    mlDict.readFromFile(args.mlConfig[0])
    if args.badscore_model[0] is not None:
        mlBadDict = readDict()
        mlBadDict.readFromFile(args.badscore_model[0])
    else:
        mlBadDict = None

    wsAtten = args.ws_atten[0]
    metadataFn = args.metadata[0]

    psDataFileName = args.inferenceData[0]
    if not os.path.isfile(psDataFileName):
        psDataFileName = os.path.join(os.environ['MKID_DATA_DIR'], psDataFileName)

    findPowers(mlDict, mlBadDict, psDataFileName, metadataFn, args.add_scores, wsAtten)

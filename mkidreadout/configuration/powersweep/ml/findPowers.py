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
import os, sys, glob
import argparse
import logging
from mkidreadout.configuration.powersweep.ml.PSFitMLData import PSFitMLData
import mkidreadout.configuration.powersweep.ml.tools as mlt
from mkidreadout.utils.readDict import readDict
from mkidreadout.configuration.powersweep.psmldata import MLData
from mkidcore.corelog import getLogger

FREQ_USE_MAG = False


def findPowers(goodModelDir, badModelDir, psDataFileName, metadataFn=None,
               saveScores=False, wsAtten=None, resWidth=None):
    if psDataFileName.split('.')[1] == 'h5':
        inferenceData = PSFitMLData(h5File=psDataFileName, useAllAttens=False, useResID=True)
        inferenceData.wsatten = wsAtten
    else:
        assert os.path.isfile(metadataFn), 'Must resonator metadata file'
        inferenceData = MLData(psDataFileName, metadataFn)

    apply_ml_model(inferenceData, wsAtten, resWidth, goodModelDir=goodModelDir, badModelDir=badModelDir)

    if psDataFileName.split('.')[1] == 'h5':
        inferenceData.savePSTxtFile(flag='_' + os.path.basename(goodModelDir), outputFN=None, saveScores=saveScores)
    elif psDataFileName.split('.')[1] == 'npz':
        inferenceData.updatemetadata()
        inferenceData.metadata.save()


def apply_ml_model(inferenceData, wsAtten, resWidth, goodModelDir='', badModelDir=''):
    """
    Uses Trained model, specified by mlDict, to infer powers from a powersweep
    saved in psDataFileName. Saves results in .txt file in $MKID_DATA_DIR
    """

    res_nums = np.shape(inferenceData.freqs)[0]

    inferenceData.opt_attens = np.zeros((res_nums))
    inferenceData.opt_freqs = np.zeros((res_nums))
    inferenceData.scores = np.zeros((res_nums))

    getLogger(__name__).debug("Inference attens: {}".format(inferenceData.attens))


    goodModelList = glob.glob(os.path.join(goodModelDir, '*.meta'))
    if len(goodModelList) > 1:
        raise Exception('Multiple models (.meta files) found in directory: ' + goodModelDir)
    elif len(goodModelList) == 0:
        raise Exception('No models (.meta files) found in directory ' + goodModelDir)
    goodModel = goodModelList[0]
    getLogger(__name__).info('Loading good model from %s', goodModel)
    sess = tf.Session()
    saver = tf.train.import_meta_graph(goodModel)
    saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(goodModel)))

    graph = tf.get_default_graph()
    x_input = graph.get_tensor_by_name('inputImage:0')
    y_output = graph.get_tensor_by_name('outputLabel:0')
    keep_prob = graph.get_tensor_by_name('keepProb:0')

    mlDict = {}
    for param in tf.get_collection('mlDict'):
        mlDict[param.op.name] = param.eval(session=sess)

    if wsAtten is None:
        wsAtten = mlDict['wsAtten']
        getLogger(__name__).warning('No WS atten specified; using value of ', str(wsAtten), 
                            ' from training config')

    wsAttenInd = np.argmin(np.abs(inferenceData.attens - wsAtten))

    if resWidth is None:
        resWidth = mlDict['resWidth']

    inferenceLabels = np.zeros((res_nums, mlDict['nAttens']))

    if badModelDir:
        badModelList = glob.glob(os.path.join(badModelDir, '*.meta'))
        if len(badModelList) > 1:
            raise Exception('Multiple models (.meta files) found in directory: ' + badModelDir)
        elif len(badModelList) == 0:
            raise Exception('No models (.meta files) found in directory: ' + badModelDir)
        badModel = badModelList[0]
        getLogger(__name__).info('Loading good model from %s', badModel)
        sess_bad = tf.Session()
        saver_bad = tf.train.import_meta_graph(badModel)
        saver_bad.restore(sess_bad, tf.train.latest_checkpoint(os.path.dirname(badModel)))

        graph = tf.get_default_graph()
        x_input_bad = graph.get_tensor_by_name('inputImage:0')
        y_output_bad = graph.get_tensor_by_name('outputLabel:0')
        keep_prob_bad = graph.get_tensor_by_name('keepProb:0')

        mlDictBad = {}
        for param in tf.get_collection('mlDict'):
            mlDictBad[param.op.name] = param.eval(session=sess_bad)

        inferenceLabelsBad = np.zeros((res_nums, mlDictBad['nAttens']))

    getLogger(__name__).debug('Using trained algorithm on images on each resonator')

    doubleCounter = 0
    for rn in range(res_nums):
        getLogger(__name__).debug("%d of %i" % (rn + 1, res_nums))

        image, freqCube, attenList = mlt.makeResImage(rn, inferenceData, wsAttenInd, mlDict['xWidth'],
                                        resWidth, mlDict['padResWin'], mlDict['useIQV'], mlDict['useMag'],
                                        mlDict['centerLoop'], mlDict['nAttens'])

        inferenceImage = [image]
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

        if rn > 0 and np.abs(inferenceData.opt_freqs[rn] - inferenceData.opt_freqs[rn - 1]) < 200.e3:
            doubleCounter += 1

        if badModelDir:
            image, freqCube, attenList = mlt.makeResImage(rn, inferenceData, wsAttenInd, mlDictBad['xWidth'],
                                            resWidth, mlDictBad['padResWin'], mlDictBad['useIQV'], mlDictBad['useMag'],
                                            mlDictBad['centerLoop'], mlDictBad['nAttens'])
            inferenceImage = [image]
            inferenceLabelsBad = sess_bad.run(y_output_bad, feed_dict={x_input_bad: inferenceImage, keep_prob_bad: 1})
            inferenceData.bad_scores[rn] = inferenceLabelsBad.max()

    getLogger(__name__).info('Had {} doubles'.format(doubleCounter))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Inference Script')
    parser.add_argument('model', help='Directory containing ML model')
    parser.add_argument('inferenceData', help='HDF5 file containing powersweep data')
    parser.add_argument('-m', '--metadata', default=None, help='Directory to save output file')
    # parser.add_argument('-o', '--output-dir', nargs=1, default=[None], help='Directory to save output file')
    parser.add_argument('-s', '--add-scores', action='store_true', help='Adds a score column to the output file')
    parser.add_argument('-w', '--ws-atten', type=float, default=None,
                        help='Attenuation where peak finding code was run')
    parser.add_argument('--res-width', type=int, default=None, 
                        help='Width of window (in units nFreqStep) to use for power/freq classification')
    parser.add_argument('-b', '--badscore-model', default='', help='Directory containing bad score model')
    args = parser.parse_args()

    psDataFileName = args.inferenceData
    if not os.path.isfile(psDataFileName):
        psDataFileName = os.path.join(os.environ['MKID_DATA_DIR'], psDataFileName)

    findPowers(args.model, args.badscore_model, psDataFileName, args.metadata, args.add_scores, args.ws_atten, args.res_width)

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

    mlDict, sess, graph, x_input, y_output, keep_prob, is_training = mlt.get_ml_model(goodModelDir)
    meanImage = tf.get_collection('meanTrainImage')[0].eval(session=sess)
    print 'mean image shape', meanImage.shape

    if wsAtten is None:
        wsAtten = mlDict['wsAtten']
        getLogger(__name__).warning('No WS atten specified; using value of ', str(wsAtten), 
                            ' from training config')

    wsAttenInd = np.argmin(np.abs(inferenceData.attens - wsAtten))

    if resWidth is None:
        resWidth = mlDict['resWidth']

    inferenceLabels = np.zeros((res_nums, mlDict['nAttens']))

    if badModelDir:
        mlDictBad, sess_bad, graph_bad, x_input_bad, y_output_bad, keep_prob_bad = mlt.get_ml_model(badModelDir)
        inferenceLabelsBad = np.zeros((res_nums, mlDictBad['nAttens']))

    getLogger(__name__).debug('Using trained algorithm on images on each resonator')

    doubleCounter = 0
    for rn in range(res_nums):
        getLogger(__name__).debug("%d of %i" % (rn + 1, res_nums))

        image, freqCube, attenList, iqVel, magsdb = mlt.makeResImage(rn, inferenceData, wsAttenInd, mlDict['xWidth'],
                                        resWidth, mlDict['padResWin'], mlDict['useIQV'], mlDict['useMag'],
                                        mlDict['centerLoop'], mlDict['nAttens'])

        image -= meanImage
        inferenceImage = [image]
        inferenceLabels[rn, :] = sess.run(y_output, feed_dict={x_input: inferenceImage, keep_prob: 1, is_training: False})
        iAtt = np.argmax(inferenceLabels[rn, :-3])
        inferenceData.opt_attens[rn] = attenList[iAtt]
        if FREQ_USE_MAG:
            inferenceData.opt_freqs[rn] = freqCube[iAtt, np.argmin(magsdb[iAtt,:])]  # TODO: make this more robust
        else:
            inferenceData.opt_freqs[rn] = freqCube[
                iAtt, np.argmax(np.correlate(iqVel[iAtt,:], np.ones(5), 'same'))]  # TODO: make this more robust

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

    getLogger(__name__, setup=True)

    psDataFileName = args.inferenceData
    if not os.path.isfile(psDataFileName):
        psDataFileName = os.path.join(os.environ['MKID_DATA_DIR'], psDataFileName)

    findPowers(args.model, args.badscore_model, psDataFileName, args.metadata, args.add_scores, args.ws_atten, args.res_width)

'''
Script to infer powers from trained model. Saves results 
as frequency file in $MKID_DATA_DIR.
Usage: python findPowers.py <mlConfigFile> <h5File>
    mlConfigFile - cfg file specifying which ML model to use. 
        Model must already be trained.
    h5File - .h5 file containing power sweep data used to 
        infer powers.

'''
import numpy as np
import tensorflow as tf
import os, sys
import argparse
from PSFitMLData import PSFitMLData
import PSFitMLTools as mlt
from mkidreadout.utils.readDict import readDict
from mkidreadout.configuration.powersweep.psmldata import MLData


def findPowers(mlDict, mlBadDict, psDataFileName, metadataFn=None, saveScores=False, wsAtten=None):
    '''
    Uses Trained model, specified by mlDict, to infer powers from a powersweep 
    saved in psDataFileName. Saves results in .txt file in $MKID_DATA_DIR
    '''
    if psDataFileName.split('.')[1]=='h5':
        inferenceData = PSFitMLData(h5File=psDataFileName, useAllAttens=False, useResID=True)
    elif psDataFileName.split('.')[1]=='npz':
        assert os.path.isfile(metadataFn), 'Must resonator metadata file'
        inferenceData = MLData(psDataFileName, metadataFn)
    
    # if mlDict['scaleXWidth']!= 1:
    #     mlDict['xWidth']=mlDict['xWidth']*mlDict['scaleXWidth'] #reset ready for get_PS_data
    
    total_res_nums = np.shape(inferenceData.freqs)[0]
    res_nums = total_res_nums
    span = range(res_nums)
    
    inferenceData.opt_attens=np.zeros((res_nums))
    inferenceData.opt_freqs=np.zeros((res_nums))
    inferenceData.scores=np.zeros((res_nums))
    wsAttenInd = np.argmin(np.abs(inferenceData.attens-wsAtten))
    
    print 'inferenceAttens', inferenceData.attens
    
    inferenceLabels = np.zeros((res_nums, mlDict['nAttens']))
    
    modelPath = os.path.join(mlDict['modelDir'], mlDict['modelName'])+'.meta'
    print 'Loading model from', modelPath
    
    sess = tf.Session()
    saver = tf.train.import_meta_graph(modelPath)
    saver.restore(sess, tf.train.latest_checkpoint(mlDict['modelDir']))
    
    graph = tf.get_default_graph()
    x_input = graph.get_tensor_by_name('inputImage:0')
    y_output = graph.get_tensor_by_name('outputLabel:0')
    keep_prob = graph.get_tensor_by_name('keepProb:0')

    if mlBadDict is not None:
        modelBadPath = os.path.join(mlBadDict['modelDir'], mlBadDict['modelName'])+'.meta'
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
    
    print 'Using trained algorithm on images on each resonator'
    skip = []
    doubleCounter = 0
    for i,rn in enumerate(span): 
        sys.stdout.write("\r%d of %i" % (i+1,res_nums) )
        sys.stdout.flush()
        #rn = 471
        image, freqCube, attenList = mlt.makeResImage(res_num = rn, center_loop=mlDict['center_loop'], phase_normalise=False,showFrames=False, dataObj=inferenceData, mlDict=mlDict, wsAttenInd=wsAttenInd)
        inferenceImage=[]
        inferenceImage.append(image)            # inferenceImage is just reformatted image
        inferenceLabels[rn,:] = sess.run(y_output, feed_dict={x_input: inferenceImage, keep_prob: 1})
        iAtt = np.argmax(inferenceLabels[rn,:])
        inferenceData.opt_attens[rn] = attenList[iAtt]
        inferenceData.opt_freqs[rn] = freqCube[iAtt, np.argmax(image[iAtt, :, 2])] #TODO: make this more robust
        inferenceData.scores[rn] = inferenceLabels[rn, iAtt]
        if rn>0:
            if(np.abs(inferenceData.opt_freqs[rn]-inferenceData.opt_freqs[rn-1])<100.e3):
                doubleCounter += 1
        
        if useBadScores:
            badInferenceLabels = sess_bad.run(y_output_bad, feed_dict={x_input_bad: inferenceImage, keep_prob_bad: 1})
            badscore = np.max(badInferenceLabels)
            inferenceData.bad_scores[rn] = badscore

    
    print '\n', doubleCounter, 'doubles'
    
    if psDataFileName.split('.')[1]=='h5':
        inferenceData.savePSTxtFile(flag = '_' + mlDict['modelName'],outputFN=None, saveScores=saveScores)
    elif psDataFileName.split('.')[1]=='npz':
        inferenceData.saveInferenceData(flag = '_' +mlDict['modelName'])

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='ML Inference Script')
    parser.add_argument('mlConfig', nargs=1, help='Machine learning model config file')
    parser.add_argument('inferenceData', nargs=1, help='HDF5 file containing powersweep data')
    parser.add_argument('-m', '--metadata', nargs=1, default=[None], help='Directory to save output file')
    #parser.add_argument('-o', '--output-dir', nargs=1, default=[None], help='Directory to save output file')
    parser.add_argument('-s', '--add-scores', action='store_true', help='Adds a score column to the output file')
    parser.add_argument('-w', '--ws-atten', nargs=1, type=float, default=[None], help='Attenuation where peak finding code was run')
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

    psDataFileName=args.inferenceData[0]
    if not os.path.isfile(psDataFileName):
        psDataFileName = os.path.join(os.environ['MKID_DATA_DIR'], psDataFileName)
     
    findPowers(mlDict, mlBadDict, psDataFileName, metadataFn, args.add_scores, wsAtten)

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
from PSFitMLData import PSFitMLData
import PSFitMLTools as mlt
from mkidreadout.utils.readDict import readDict


def findPowers(mlDict, h5FileName, outputFN=None):
    '''
    Uses Trained model, specified by mlDict, to infer powers from a powersweep 
    saved in h5FileName. Saves results in .txt file in $MKID_DATA_DIR
    '''
    inferenceData = PSFitMLData(h5File=h5FileName, useAllAttens=False, useResID=True)
    
    # if mlDict['scaleXWidth']!= 1:
    #     mlDict['xWidth']=mlDict['xWidth']*mlDict['scaleXWidth'] #reset ready for get_PS_data
    
    total_res_nums = np.shape(inferenceData.freqs)[0]
    res_nums = total_res_nums
    span = range(res_nums)
    
    inferenceData.opt_attens=np.zeros((res_nums))
    inferenceData.opt_freqs=np.zeros((res_nums))
    
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
    
    print 'Using trained algorithm on images on each resonator'
    skip = []
    doubleCounter = 0
    for i,rn in enumerate(span): 
        sys.stdout.write("\r%d of %i" % (i+1,res_nums) )
        sys.stdout.flush()
        image = mlt.makeResImage(res_num = rn, center_loop=mlDict['center_loop'], phase_normalise=False,showFrames=False, dataObj=inferenceData, mlDict=mlDict)
        inferenceImage=[]
        inferenceImage.append(image)            # inferenceImage is just reformatted image
        inferenceLabels[rn,:] = sess.run(y_output, feed_dict={x_input: inferenceImage, keep_prob: 1})
        iAtt = np.argmax(inferenceLabels[rn,:])
        inferenceData.opt_attens[rn] = inferenceData.attens[iAtt]
        inferenceData.opt_freqs[rn] = inferenceData.freqs[rn,mlt.get_peak_idx(rn,iAtt, dataObj=inferenceData, smooth=True)]
        if rn>0:
            if(np.abs(inferenceData.opt_freqs[rn]-inferenceData.opt_freqs[rn-1])<100.e3):
                doubleCounter += 1
                print 'res_num', rn
                print 'oldfreq:', inferenceData.opt_freqs[rn-1]
                print 'curfreq:', inferenceData.opt_freqs[rn]
                padResWidth = 20
                if inferenceData.opt_freqs[rn] > inferenceData.opt_freqs[rn-1]:
                    print 'isgreater'
                    image = mlt.makeResImage(res_num = rn-1, center_loop=mlDict['center_loop'], phase_normalise=False, showFrames=False, dataObj=inferenceData, padFreq=inferenceData.opt_freqs[rn], mlDict=mlDict)
                    inferenceImage = [image]
                    inferenceLabels[rn-1,:] = sess.run(y_output, feed_dict={x_input: inferenceImage, keep_prob: 1})
                    iAtt = np.argmax(inferenceLabels[rn-1,:])
                    inferenceData.opt_attens[rn-1] = inferenceData.attens[iAtt]
                    
                    padFreqInd = np.argmin(np.abs(inferenceData.freqs[rn-1]-inferenceData.opt_freqs[rn])) 
                    print 'findatt: padFreqInd', padFreqInd
                    if(padFreqInd>mlDict['xWidth']/2):
                        padInd = padFreqInd - padResWidth
                        cutType = 'top'
                    else:
                        padInd = padFreqInd + padResWidth
                        cutType = 'bottom'
                    inferenceData.opt_freqs[rn-1] = inferenceData.freqs[rn-1, mlt.get_peak_idx(rn-1, iAtt, dataObj=inferenceData, smooth=True, cutType=cutType, padInd=padInd)]
                    print 'newfreq', inferenceData.opt_freqs[rn-1]
                else:
                    image = mlt.makeResImage(res_num = rn, center_loop=mlDict['center_loop'], phase_normalise=False, showFrames=False, dataObj=inferenceData, padFreq=inferenceData.opt_freqs[rn-1], mlDict=mlDict)
                    inferenceImage = [image]
                    inferenceLabels[rn,:] = sess.run(y_output, feed_dict={x_input: inferenceImage, keep_prob: 1})
                    iAtt = np.argmax(inferenceLabels[rn,:])
                    inferenceData.opt_attens[rn] = inferenceData.attens[iAtt]
                    padFreqInd = np.argmin(np.abs(inferenceData.freqs[rn]-inferenceData.opt_freqs[rn-1])) 
                    print 'findatt: padFreqInd', padFreqInd
                    if(padFreqInd>mlDict['xWidth']/2):
                        padInd = padFreqInd - padResWidth
                        cutType = 'top'
                    else:
                        padInd = padFreqInd + padResWidth
                        cutType = 'bottom'
                    inferenceData.opt_freqs[rn] = inferenceData.freqs[rn, mlt.get_peak_idx(rn, iAtt, dataObj=inferenceData, smooth=True, cutType=cutType, padInd=padInd)]
                    print 'newfreq', inferenceData.opt_freqs[rn]
        del inferenceImage
        del image
    
    print '\n', doubleCounter, 'doubles fixed'
    
    inferenceData.savePSTxtFile(flag = '_' + mlDict['modelName'],outputFN=outputFN)

if __name__=='__main__':
    if len(sys.argv)<3:
        print 'Must specify ML config file and h5 file in MKID_DATA_DIR!'
        exit(1)

    mlDict = readDict()
    mlDict.readFromFile(sys.argv[1])

    h5FileName=sys.argv[2]
    if not os.path.isfile(h5FileName):
        h5FileName = os.path.join(os.environ['MKID_DATA_DIR'], h5FileName)
    
    try: outputDir=sys.argv[3]
    except: outputDir=None
    
    findPowers(mlDict, h5FileName, outputDir)

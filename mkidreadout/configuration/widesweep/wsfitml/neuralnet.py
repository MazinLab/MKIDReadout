import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import signal
import os, sys

class WSFitNN:
    def __init__(self, **kwargs):
        self.trainFileList = kwargs.pop('trainFileList', None)
        self.inferenceFile = kwargs.pop('inferenceFile', None)
        self.mlParamDict = kwargs.pop('mlParamDict')
        self.freqSpacing = 12.5 #kHz
        self.winSize = int(1000/self.freqSpacing) #image window size
        self.locWinSize = int(500/self.freqSpacing) #size of localization window; label is 1 if peak exists within this
        self.trainImageRange = int(self.locWinSize*0.75) #training images are made in window this wide around each peak
        self.trainFrac = 0.9
        self.nColorChannels = 3 #number of channels in the image (ie I, Q and IQV is a 3 channel image)
    
    def initTFGraph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.winSize, self.nColorChannels])
        self.yExist_ = tf.placeholder(tf.float32, shape=[None, 2] #first two indices are 1 hot for classification, next is localization value
        self.yLoc_ = tf.placeholder(tf.float32, shape=[None, 1])

        xImage = tf.reshape(self.x, [-1, 1, self.winSize, self.nColorChannels])

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, stride=[1, 1, 1, 1], shape='SAME')

        def pool1xn(x, n):
            return tf.nn.max_pool(x, ksize=[1, 1, n, 1], padding='SAME')

        def weightVariable(shape):
            intitial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def biasVariable(shape):
            intitial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        WConv1 = weightVariable([1, self.mlParamDict['w1Size'], self.nColorChannels, self.mlParamDict['nFilt1'])
        bConv1 = biasVariable([self.mlParamDict['nFilt1'])
        hConv1 = tf.nn.relu(conv2d(xImage, WConv1) + bConv1)
        hPool1 = pool1xn(x, self.mlParamDict['nPool1'])

        WConv2 = weightVariable([1, self.mlParamDict['w2Size'], self.mlParamDict['nFilt1'], self.mlParamDict['nFilt2'])
        bConv2 = biasVariable([self.mlParamDict['nFilt2'])
        hConv2 = tf.nn.relu(conv2d(hPool1, WConv2) + bConv2)
        hPool2 = pool1xn(x, self.mlParamDict['nPool2'])

        WConv3 = weightVariable([1, self.mlParamDict['w3Size'], self.mlParamDict['nFilt2'], self.mlParamDict['nFilt3'])
        bConv3 = biasVariable([self.mlParamDict['nFilt3'])
        hConv3 = tf.nn.relu(conv2d(hPool2, WConv3) + bConv3)
        hPool3 = pool1xn(x, self.mlParamDict['nPool3'])

        WFC1 = weightVariable([tf.shape(hpool3)[2]*tf.shape(hpool3)[3], self.mlParamDict['nFC1'])
        bFC1 = biasVariable([self.mlParamDict['nFC1'])

        hPool3Flat = tf.reshape(hPool3, [-1, tf.shape(hpool3)[2]*tf.shape(hpool3)[3]])
        with tf.control_dependencies([tf.assert_equal(tf.shape(hPool3Flat)[0], numImg)]):
            hFC1 = tf.nn.relu(tf.matmul(hPool3Flat, WFC1) + bFC1)

        keepProb = tf.placeholder(tf.float32)
        hFC1Drop = tf.nn.dropout(hFC1, keepProb)

        WFCExist = weightVariable([self.mlParamDict['nFC1'], 2])
        bFCExist = biasVariable([2])

        WFCLoc = weightVariable([self.mlParamDict['nFC1'], 1])
        bFCLoc = biasVariable([1])

        self.yExist = tf.matmul(hFC1Drop, WFCExist) + bFCExist
        self.yLoc = tf.matmul(hFC1Drop, WFCLoc) + bFCLoc

        self.existCrossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yExist_, logits=self.yExist))
        self.locSquaredLoss = tf.reduce_mean(tf.multiply(tf.divide(yExist_, yExist_ + 0.00001), tf.square(self.yLoc - yLoc_)))

        self.trainStep = tf.train.AdamOptimizer(10**-3).minimize(self.existCrossEntropy + self.locSquaredLoss)
        
        existCorrect = tf.equal(tf.argmax(self.yExist_, 1), tf.argmax(self.yExist, 1))
        self.existAccuracy = tf.reduce_mean(tf.cast(existCorrect, tf.float32))

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

        self.saver = tf.train.Saver()
        


    def makeTrainData(self):
        self.rawTrainData = WSFitMLData(self.trainFileList)
        self.rawTrainData.loadPeaks('all')
        numPeaks = len(self.rawTrainData.peakLocs)
        trainLabels = {'peakExists':np.zeros((2*numPeaks*trainImageRange,2)), 'localization':np.zeros(2*numPeaks*trainImageRange)}
        
        #make training images from peak locations
        trainIms = np.empty(0)
        trainImCt = 0
        peakAbsentBool = np.ones(len(self.rawTrainData.freqs)) #1 if there is no peak within winSize of index
        for peakLoc in self.rawTrainData.peakLocs:
            peakAbsentBool[peakLock-self.locWinSize/2:peakLoc+self.locWinSize/2] = 0 #mark indices where there are peaks
            for winOffs in range(-self.trainImageRange/2, self.trainImageRange/2):
                trainIms = np.append(trainIms, makePeakImage(self.rawTrainData), peakLoc+winOffs)
                trainLabels['peakExists'][trainImCt,1] = 1
                minPeakIndex = np.argmin(peakLoc + winOffs - self.rawTrainData.peakLocs) #grabs the minimum peak location, even if it isn't peakLoc
                trainLabels['localization'][trainImCt] = self.rawTrainData.peakLocs[minPeakIndex] - (peakLoc + winOffs)
                trainImCt += 1
                

        notBoundaryBool = np.ones(len(self.rawTrainData.freqs)) # 1 if location is far enough from data set boundaries
        notBoundaryBool[0:self.trainImageRange/2] = 0
        notBoundaryBool[-self.trainImageRange/2:] = 0
        for boundary in self.rawTrainData.boundaryInds:
            notBoundaryBool[boundary-self.trainImageRange/2:boundary+self.trainImageRange/2] = 0

        peakAbsentLocs = np.logical_and(notBoundaryBool, peakAbsentBool)
        #make training images from locations where there are no peaks
        peakAbsentLocs = np.where(peakAbsentBool)[0] # list of indices where there are no peaks
        #draw numPeaks samples from locations where there are no peaks
        peakAbsentImLocs = np.random.choice(peakAbsentLocs, numPeaks)
        for peakLoc in peakAbsentImLocs:
            trainIms = np.append(trainIms, makePeakImage(self.rawTrainData, peakLoc))
        trainLabels['peakExists'][numPeaks*trainImageRange:2*numPeaks*trainImageRange, 0] = 1
        trainLabels['localization'][numPeaks*trainImageRange:2*numPeaks*trainImageRange] = 0

        #sample from full training set to make train and test sets
        imageIndices = np.arange(2*numPeaks)
        trainIndices = np.random.choice(imageIndices, int(self.trainFrac*2*numPeaks))
        testIndBool = np.ones(2*numPeaks)
        testIndBool[trainIndices] = 0
        testIndices = np.where(testIndBool)[0]

        self.trainImages = trainIms[trainIndices]
        self.trainLabels = trainLabels[trainIndices]
        self.testImages = trainIms[testIndices]
        self.testLabels = trainLabels[testIndices]


    def trainModel(self):
        pass

    def makePeakImage(self, wsFitData, peakLoc):
        image = np.zeros((self.winSize, 3))
        image[:, 0] = wsFitData.iVals[peakLoc-self.winSize/2:peakLoc+self.winSize/2]
        image[:, 1] = wsFitData.qVals[peakLoc-self.winSize/2:peakLoc+self.winSize/2]
        image[:, 2] = wsFitData.iqVels[peakLoc-self.winSize/2:peakLoc+self.winSize/2]
        return image
    
    def saveModelCheckpoint(self, filename):
        filePath = os.path.join(self.modelDir, filename)
        self.saver.save(self.sess, filePath)

    def restoreModelCheckpoint(self, filename):
        filePath = os.path.join(self.modelDir, filename)
        self.saver.restore(self.sess, filePath)



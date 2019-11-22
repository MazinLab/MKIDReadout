N_CLASSES = 4 #good, saturated, underpowered, bad/no res
COLLISION_FREQ_RANGE = 200.e3
MAX_IMAGES = 10000

ACC_INTERVAL = 500
SAVER_INTERVAL = 10000

import numpy as np
import math
import sys, os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import mkidreadout.configuration.powersweep.ml.tools as mlt
import mkidreadout.configuration.sweepdata as sd


class WPSNeuralNet(object):
    
    def __init__(self, mlDict):
        self.mlDict = mlDict
        self.nClasses = N_CLASSES
        self.nColors = 2
        if mlDict['useIQV']:
            self.nColors += 1
        if mlDict['useVectIQV']:
            self.nColors += 2

        if not(os.path.isdir(mlDict['modelDir'])):
            os.mkdir(mlDict['modelDir'])
 
        self.trainFile = os.path.join(mlDict['modelDir'], mlDict['trainNPZ'])
        self.imageShape = (mlDict['attenWinBelow'] + mlDict['attenWinAbove'] + 1, mlDict['freqWinSize'], self.nColors)

    def makeTrainData(self):
        trainImages = np.empty((0,) + self.imageShape)
        testImages = np.empty((0,) + self.imageShape)
        trainLabels = np.empty((0, self.nClasses))
        testLabels = np.empty((0, self.nClasses))
        for i, rawTrainFile in enumerate(self.mlDict['rawTrainFiles']):
            rawTrainFile = os.path.join(self.mlDict['trainFileDir'], rawTrainFile)
            rawTrainMDFile = os.path.join(self.mlDict['trainFileDir'], self.mlDict['rawTrainLabels'][i])
            trainMD = sd.SweepMetadata(file=rawTrainMDFile)
            trainSweep = sd.FreqSweep(rawTrainFile)

            goodResMask = ~np.isnan(trainMD.atten)
            attenblock = np.tile(trainSweep.atten, (len(goodResMask),1))
            optAttenInds = np.argmin(np.abs(attenblock.T - trainMD.atten), axis=0)
            
            if self.mlDict['trimAttens']:
                goodResMask = goodResMask & ~(optAttenInds < self.mlDict['attenWinBelow'])
                goodResMask = goodResMask & ~(optAttenInds >= (len(trainSweep.atten) - self.mlDict['attenWinAbove']))
            #if self.mlDict['filterMaxedAttens']:
            #    maxAttenInd = np.argmax(trainSweep.atten)
            #    goodResMask = goodResMask & ~(optAttenInds==maxAttenInd)
            #    print 'Filtered', np.sum(rawTrainData.opt_iAttens==maxAttenInd), 'maxed out attens.'

            images = np.zeros((self.mlDict['nImagesPerRes']*self.nClasses*np.sum(goodResMask),) + self.imageShape)
            labels = np.zeros((self.mlDict['nImagesPerRes']*self.nClasses*np.sum(goodResMask), self.nClasses))

            optAttens = trainMD.atten[goodResMask]
            optFreqs = trainMD.freq[goodResMask]
            optAttenInds = optAttenInds[goodResMask]
            
            offResFreqs = np.linspace(np.min(optFreqs), np.max(optFreqs), 10000)
            offResFreqMask = np.ones(len(offResFreqs), dtype=bool)

            for f in optFreqs:
                offResFreqMask &= (np.abs(offResFreqs - f) > COLLISION_FREQ_RANGE)

            offResFreqs = offResFreqs[offResFreqMask]

            imgCtr = 0
            for i in range(np.sum(goodResMask)):
                satResMask = np.ones(len(trainSweep.atten), dtype=bool)
                satResMask[optAttenInds[i] - self.mlDict['trainSatThresh']:] = 0
                satResAttens = trainSweep.atten[satResMask]

                upResMask = np.ones(len(trainSweep.atten), dtype=bool)
                upResMask[:optAttenInds[i] + self.mlDict['trainUPThresh']] = 0
                upResMask[-self.mlDict['attenWinAbove']:] = 0
                upResAttens = trainSweep.atten[upResMask]

                for j in range(self.mlDict['nImagesPerRes']):
                    images[imgCtr], _, _ = mlt.makeWPSImage(trainSweep, optFreqs[i], optAttens[i], self.imageShape[1], 
                        self.imageShape[0], self.mlDict['useIQV'], self.mlDict['useVectIQV']) #good image
                    labels[imgCtr] = np.array([1, 0, 0, 0])
                    imgCtr += 1
                    if np.any(satResMask):
                        try:
                            satResAtten = satResAttens[-1] #go down in atten from lowest-power sat res
                            satResAttens = np.delete(satResAttens, -1) #pick without replacement
                        except IndexError:
                            satResAtten = np.random.choice(trainSweep.atten[satResMask]) #pick a random one if out of attens
                        images[imgCtr], _, _ = mlt.makeWPSImage(trainSweep, optFreqs[i], satResAtten, self.imageShape[1], 
                            self.imageShape[0], self.mlDict['useIQV'], self.mlDict['useVectIQV']) #saturated image
                        labels[imgCtr] = np.array([0, 1, 0, 0])
                        imgCtr += 1

                    if np.any(upResMask):
                        try:
                            upResAtten = upResAttens[0] #go up in atten from highest-power UP res
                            upResAttens = np.delete(upResAttens, 0) #pick without replacement
                        except IndexError:
                            upResAtten = np.random.choice(trainSweep.atten[upResMask])
                        images[imgCtr], _, _ = mlt.makeWPSImage(trainSweep, optFreqs[i], upResAtten, self.imageShape[1], 
                            self.imageShape[0], self.mlDict['useIQV'], self.mlDict['useVectIQV']) #upurated image
                        labels[imgCtr] = np.array([0, 0, 1, 0])
                        imgCtr += 1

                    offResF = np.random.choice(offResFreqs)
                    offResAtt = np.random.choice(trainSweep.atten)
                    images[imgCtr], _, _ = mlt.makeWPSImage(trainSweep, offResF, offResAtt, self.imageShape[1], 
                        self.imageShape[0], self.mlDict['useIQV'], self.mlDict['useVectIQV']) #off resonance image
                    labels[imgCtr] = np.array([0, 0, 0, 1])
                    imgCtr += 1

            images = images[:imgCtr]
            labels = labels[:imgCtr]

            trainImages = np.append(trainImages, images, axis=0)
            trainLabels = np.append(trainLabels, labels, axis=0)

            print 'File ' + rawTrainFile + ': Added ' + str(len(labels)) + ' training images from ' \
                    + str(np.sum(goodResMask)) + ' resonators'


        allInds = np.arange(len(trainImages), dtype=np.int)
        testInds = np.random.choice(allInds, int(len(allInds)*(1-self.mlDict['trainFrac'])), replace=False)
        trainInds = np.setdiff1d(allInds, testInds)

        testImages = trainImages[testInds]
        testLabels = trainLabels[testInds]
        trainImages = trainImages[trainInds]
        trainLabels = trainLabels[trainInds]

        print 'Saving', len(trainImages), 'train images and', len(testImages), 'test images'
        
        np.savez(self.trainFile, trainImages=trainImages, trainLabels=trainLabels,
                testImages=testImages, testLabels=testLabels)

    def initializeAndTrainModel(self, debug=False, saveGraph=False):
        if not os.path.isfile(self.trainFile):
            print 'Could not find train file. Making new training images from initialFile'
            self.makeTrainData()

        print 'Loading images from ', self.trainFile
        trainData = np.load(self.trainFile)
        trainImages = trainData['trainImages']
        trainLabels = trainData['trainLabels']
        testImages = trainData['testImages']
        testLabels = trainData['testLabels']

        if self.mlDict['overfitTest']:
            trainImages = trainImages[:30]
            trainLabels = trainLabels[:30]
            testLabels = trainLabels
            testImages = trainImages

        if self.mlDict['centerDataset']:
            self.meanTrainImage = np.mean(trainImages, axis=0)
            trainImages = trainImages - self.meanTrainImage
            testImages = testImages - self.meanTrainImage
            print 'Subtracting mean image:', self.meanTrainImage
        else:
            self.meanTrainImage = np.zeros(trainImages[0].shape)

        
        print 'Number of training images:', np.shape(trainImages), ' Number of test images:', np.shape(testImages)
 
        attenWin = 1 + self.mlDict['attenWinBelow'] + self.mlDict['attenWinAbove']
        self.x = tf.placeholder(tf.float32, [None, attenWin, self.mlDict['freqWinSize'], self.nColors], name='inputImage')
        x_image = tf.reshape(self.x, [-1, attenWin, self.mlDict['freqWinSize'], self.nColors])

        numImg = tf.shape(x_image)[0]

        def weight_variable(shape, name=None):
            #initial = tf.Variable(tf.zeros(shape))
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name=None):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def conv3d(x, W):
            return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')
        
        def max_pool_nx1(x,n):
            return tf.nn.max_pool(x, ksize=[1, 1, n, 1], strides=[1, 1, n, 1], padding='SAME')

        def max_pool_nxm(x,n,m):
            return tf.nn.max_pool(x, ksize=[1, n, m, 1], strides=[1, n, m, 1], padding='SAME')

        def variable_summaries(var):
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                std = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('std', std)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)
        
        self.keep_prob = tf.placeholder(tf.float32, name='keepProb')
        self.is_training = tf.placeholder(tf.bool, name='isTraining')

        with tf.name_scope('Layer1'):
            num_filt1 = self.mlDict['num_filt1']
            n_pool1 = self.mlDict['n_pool1']
            self.num_filt1 = num_filt1
            W_conv1 = weight_variable([attenWin, self.mlDict['conv_win1'], self.nColors, num_filt1], name='W_conv1')
            b_conv1 = bias_variable([num_filt1])
            variable_summaries(W_conv1)
            variable_summaries(b_conv1)
            h_conv1_raw = conv2d(x_image, W_conv1) + b_conv1
            if self.mlDict['useBatchNorm']:
                h_conv1_batch = tf.layers.batch_normalization(h_conv1_raw, training=self.is_training)
                h_conv1 = tf.nn.relu(h_conv1_batch, name='h_conv1')
            else:
                h_conv1 = tf.nn.relu(h_conv1_raw, name='h_conv1')
            h_pool1 = max_pool_nxm(h_conv1,n_pool1[0],n_pool1[1])
            h_pool1_dropout = tf.nn.dropout(h_pool1, self.keep_prob)
            xWidth1 = int(math.ceil(self.mlDict['freqWinSize']/float(n_pool1[1])))
            aWidth1 = int(math.ceil(attenWin/float(n_pool1[0])))
            cWidth1 = num_filt1
            tf.summary.histogram('h_conv1', h_conv1)
            tf.summary.histogram('h_pool1', h_pool1)
        
        with tf.name_scope('Layer2'):
            num_filt2 = self.mlDict['num_filt2']
            n_pool2 = self.mlDict['n_pool2']
            self.num_filt2 = num_filt2
            W_conv2 = weight_variable([self.mlDict['conv_win2'][0], self.mlDict['conv_win2'][1], cWidth1, num_filt2], name='W_conv2')
            b_conv2 = bias_variable([num_filt2])
            variable_summaries(W_conv2)
            variable_summaries(b_conv2)
            h_conv2_raw = conv2d(h_pool1_dropout, W_conv2) + b_conv2
            if self.mlDict['useBatchNorm']:
                h_conv2_batch = tf.layers.batch_normalization(h_conv2_raw, training=self.is_training)
                h_conv2 = tf.nn.relu(h_conv2_batch, name='h_conv2')
            else:
                h_conv2 = tf.nn.relu(h_conv2_raw, name='h_conv2')
            with tf.control_dependencies([tf.assert_equal(tf.shape(h_pool1),(numImg,aWidth1,xWidth1,cWidth1),message='hpool1 assertion')]):
                h_pool2 = max_pool_nxm(h_conv2, n_pool2[0], n_pool2[1])
                h_pool2_dropout = tf.nn.dropout(h_pool2, self.keep_prob)
                xWidth2 = int(math.ceil(xWidth1/float(n_pool2[1])))
                cWidth2 = num_filt2
                aWidth2 = int(math.ceil(aWidth1/float(n_pool2[0])))
            tf.summary.histogram('h_conv2', h_conv2)
            tf.summary.histogram('h_pool2', h_pool2)

        with tf.name_scope('Layer3'):
            num_filt3 = self.mlDict['num_filt3']
            n_pool3 = self.mlDict['n_pool3']
            self.num_filt3 = num_filt3
            W_conv3 = weight_variable([self.mlDict['conv_win3'][0], self.mlDict['conv_win3'][1], cWidth2, num_filt3], name='W_conv3')
            b_conv3 = bias_variable([num_filt3])
            variable_summaries(W_conv3)
            variable_summaries(b_conv3)
            h_conv3_raw = conv2d(h_pool2_dropout, W_conv3) + b_conv3
            if self.mlDict['useBatchNorm']:
                h_conv3_batch = tf.layers.batch_normalization(h_conv3_raw, training=self.is_training)
                h_conv3 = tf.nn.relu(h_conv3_batch, name='h_conv3')
            else:
                h_conv3 = tf.nn.relu(h_conv3_raw, name='h_conv3')
            with tf.control_dependencies([tf.assert_equal(tf.shape(h_pool2),(numImg,aWidth2,xWidth2,cWidth2),message='hpool2 assertion')]):
                h_pool3 = max_pool_nxm(h_conv3, n_pool3[0], n_pool3[1])
                h_pool3_dropout = tf.nn.dropout(h_pool3, self.keep_prob)
                xWidth3 = int(math.ceil(xWidth2/float(n_pool3[1])))
                aWidth3 = int(math.ceil(aWidth2/float(n_pool3[0])))
                cWidth3 = num_filt3
            tf.summary.histogram('h_conv3', h_conv3)
            tf.summary.histogram('h_pool3', h_pool3)

        # num_filt4 = 2
        # n_pool4 = 2
        # self.num_filt4 = num_filt4
        # W_conv4 = weight_variable([1, 3, cWidth3, num_filt4])
        # b_conv4 = bias_variable([num_filt4])
        # h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
        # with tf.control_dependencies([tf.assert_equal(tf.shape(h_pool3),(numImg,self.mlDict['nAttens'],xWidth3,cWidth3),message='hpool3 assertion')]):
        #     h_pool4 = max_pool_nx1(h_conv4, n_pool4)
        #     h_pool4_dropout = tf.nn.dropout(h_pool4, self.keep_prob)
        #     xWidth4 = int(math.ceil(xWidth3/float(n_pool4)))
        #     cWidth4 = num_filt4
        
        with tf.name_scope('FinalLayer'):
            h_pool3_flat = tf.reshape(h_pool3_dropout,[-1,aWidth3*cWidth3*xWidth3])        
            W_final = weight_variable([aWidth3*cWidth3*xWidth3, self.nClasses])
            b_final = bias_variable([self.nClasses])     
            variable_summaries(W_final)
            variable_summaries(b_final)
            
            with tf.control_dependencies([tf.assert_equal(tf.shape(h_pool3),(numImg,aWidth3,xWidth3,cWidth3))]):
                h_conv_final = tf.matmul(h_pool3_flat, W_final) + b_final 
                tf.summary.histogram('h_conv_final', h_conv_final)
        
        self.y=tf.nn.softmax(h_conv_final, name="outputLabel") #h_fc1_drop   

        y_ = tf.placeholder(tf.float32, [None, self.nClasses]) # true class lables identified by user 
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(self.y+ 1e-10), reduction_indices=[1])) # find optimum solution by minimizing error
        tf.summary.scalar('cross_entropy', cross_entropy)
        #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y,y_))
        #yWeighted = tf.mul(self.y, tf.to_float(tf.range(tf.shape(self.y)[1])))
        #yInd = tf.reduce_sum(yWeighted, reduction_indices=1)
        #y_Ind = tf.to_float(tf.argmax(y_, 1))
        
        #squared_loss = tf.reduce_mean(tf.to_float(tf.square(yInd-y_Ind)))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(self.mlDict['learning_rate']).minimize(cross_entropy) 
        
        for k, v in self.mlDict.items():
            tf.add_to_collection('mlDict', tf.constant(value=v, name=k))

        tf.add_to_collection('meanTrainImage', tf.constant(value=self.meanTrainImage, name='image'))

        init = tf.global_variables_initializer()

        saver = tf.train.Saver(save_relative_paths=True)
        
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1)) #which ones did it get right?
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(self.mlDict['modelDir'], 'train'))
        test_writer = tf.summary.FileWriter(os.path.join(self.mlDict['modelDir'], 'test'))
        if saveGraph:
            graph_writer = tf.summary.FileWriter(os.path.join(self.mlDict['modelDir'], 'graphs'), tf.get_default_graph())
       
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        if debug:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        self.sess.run(init) # need to do this everytime you want to access a tf variable (for example the true class labels calculation or plotweights)
 
        start_time = time.time()
        nTrainEpochs = self.mlDict['trainEpochs']
        batchSize = self.mlDict['batchSize']
        trainReps = nTrainEpochs*trainLabels.shape[0]/batchSize
        if np.shape(trainLabels)[0]< batchSize:
            batchSize = np.shape(trainLabels)[0]/2
        ce_log = np.zeros(trainReps/ACC_INTERVAL + 1)
        acc_log=np.zeros(trainReps/ACC_INTERVAL + 1)
        print 'Performing', trainReps, 'training repeats, using batches of', batchSize

        for i in range(trainReps):  #perform the training step using random batches of images and according labels
            batch_xs, batch_ys = mlt.next_batch(trainImages, trainLabels, batchSize) 
            sys.stdout.write('\rbatch: %i ' % (i))
            sys.stdout.flush()

            if i % SAVER_INTERVAL == 0:
                ce_log[i/ACC_INTERVAL] = self.sess.run(cross_entropy, feed_dict={self.x: batch_xs, y_: batch_ys, self.keep_prob: 1, self.is_training: False})
                summary, acc = self.sess.run([merged, accuracy], feed_dict={self.x: testImages, y_: testLabels, self.keep_prob: 1, self.is_training: False})
                acc_log[i/ACC_INTERVAL] = acc*100
                #test_writer.add_summary(summary, i)
                if saveGraph:
                    graph_writer.add_summary(summary, i)

                print acc*100
                #summary, _ = self.sess.run([merged, train_step], feed_dict={self.x: batch_xs, y_: batch_ys, self.keep_prob: self.mlDict['keep_prob'], self.is_training: True}) #calculate train_step using feed_dict
                #train_writer.add_summary(summary, i)

            elif i % ACC_INTERVAL == 0:
                ce_log[i/ACC_INTERVAL] = self.sess.run(cross_entropy, feed_dict={self.x: batch_xs, y_: batch_ys, self.keep_prob: 1, self.is_training: False})
                acc = self.sess.run(accuracy, feed_dict={self.x: testImages, y_: testLabels, self.keep_prob: 1, self.is_training: False})
                print acc*100
                acc_log[i/ACC_INTERVAL] = acc*100
                self.sess.run(train_step, feed_dict={self.x: batch_xs, y_: batch_ys, self.keep_prob: self.mlDict['keep_prob'], self.is_training: True}) #calculate train_step using feed_dict

            else:  
                self.sess.run(train_step, feed_dict={self.x: batch_xs, y_: batch_ys, self.keep_prob: self.mlDict['keep_prob'], self.is_training: True}) #calculate train_step using feed_dict
        
        print "--- %s seconds ---" % (time.time() - start_time)

        modelSavePath = os.path.join(self.mlDict['modelDir'], self.mlDict['modelName'])
        print 'Saving model in', modelSavePath
        save_path = saver.save(self.sess, modelSavePath)
        saver.export_meta_graph(os.path.join(modelSavePath, self.mlDict['modelName']) + '.meta')
        ys_true = self.sess.run(tf.argmax(y_,1), feed_dict={self.x: testImages, y_: testLabels, self.keep_prob: 1, self.is_training: False})
        y_probs = self.sess.run(self.y, feed_dict={self.x: testImages, y_: testLabels, self.keep_prob: 1, self.is_training: False})
        ys_guess = np.argmax(y_probs, 1)
        right = []
        for i,y in enumerate(ys_true):
            #print i, y, ys_guess[i] 
            if ys_guess[i] == y: # or ys_guess[i] == y-1 or ys_guess[i] == y+1:
                #print i, 'guessed right'
                right.append(i)

        print len(right), len(ys_true), float(len(right))/len(ys_true)

        testScore = self.sess.run(accuracy, feed_dict={self.x: testImages, y_: testLabels, self.keep_prob: 1, self.is_training: False}) * 100
        print 'Accuracy of model in testing: ', testScore, '%'

        trainScore = 0
        nTrainScoreBatches = len(trainImages)/MAX_IMAGES + 1
        for i in range(nTrainScoreBatches):
            endInd = min((i+1)*MAX_IMAGES, len(trainImages))
            trainImagesBatch = trainImages[i*MAX_IMAGES:endInd]
            trainLabelsBatch = trainLabels[i*MAX_IMAGES:endInd]
            trainScore += self.sess.run(accuracy, feed_dict={self.x: trainImagesBatch, y_: trainLabelsBatch, self.keep_prob: 1, self.is_training: False}) * 100/nTrainScoreBatches
        print 'Accuracy of model in training: ', trainScore, '%'
        print tf.get_default_graph().get_all_collection_keys()
        
        np.savez(modelSavePath+'_confusion.npz', ys_true=ys_true, ys_guess=ys_guess, y_probs=y_probs, ce=ce_log, acc=acc_log)
        logfile = modelSavePath + '_train_' + time.strftime("%Y%m%d-%H%M%S",time.localtime()) + '.log'
        with open(logfile, 'w') as lf:
            lf.write('Test Accuracy: ' + str(testScore/100.) + '\n')
            lf.write('Train Accuracy: ' + str(trainScore/100.) + '\n')
            lf.write('\n')
            for k, v in self.mlDict.items():
                lf.write(k + ': ' + str(v) + '\n')
                


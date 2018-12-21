''' 
Author Rupert Dodkins, Neelay Fruitwala

A script to automate the identification of resonator attenuations normally performed by PSFit.py. This is accomplished 
using Google's Tensor Flow machine learning package which implements a pattern recognition convolution neural network 
algorithm and classification algorithm on power and frequency sweep data saved in h5 format.

Usage: train using cfg file specifying model parameters and training data (see FlemingLFV1.cfg), with trainModel.py.
For inference, use findPowers.py.

How it works:
For every resonator an "image" of it's power sweep is made, with axes of frequency and attenuation. The image has three 
"color" channels, corresponding to I, Q, and IQ Velocity. This image is used by the CNN to find the optimal attenuation.

mlClassification defines the graph structure, trains the model, and saves it so it can be used for inference.

'''

import numpy as np
import sys, os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import pickle
import random
import time
import math
from mkidreadout.configuration.powersweep.ml.PSFitMLData import *
from mkidreadout.configuration.powersweep.psmldata import *
np.set_printoptions(threshold=np.inf)
import mkidreadout.configuration.powersweep.ml.tools as mlt

#removes visible depreciation warnings from lib.iqsweep
import warnings
warnings.filterwarnings("ignore")


class mlClassification():
    def __init__(self,  mlDict):
        '''
        Implements the machine learning pattern recognition algorithm on IQ velocity data as well as other tests to 
        choose the optimum attenuation for each resonator
        '''
        self.mlDict = mlDict
        #self.uAttDist = +2 # rule of thumb attenuation steps to reach the underpowed peak
        self.nClass = self.mlDict['nAttens']
        self.trainFile = os.path.join(self.mlDict['modelDir'], self.mlDict['trainPklFile'])
        self.trainFrac = self.mlDict['trainFrac']
        self.testFrac = 1-self.trainFrac

    def makeTrainData(self):                
        for i,rawTrainFile in enumerate(self.mlDict['rawTrainFiles']):
            rawTrainFile = os.path.join(self.mlDict['trainFileDir'], rawTrainFile)
            rawTrainLabelFile = os.path.join(self.mlDict['trainFileDir'], self.mlDict['rawTrainLabels'][i])
            if rawTrainFile.split('.')[1]=='h5':
                rawTrainData = PSFitMLData(h5File = rawTrainFile, PSFile=rawTrainLabelFile, useResID=True)
                rawTrainData.loadTrainData()
            elif rawTrainFile.split('.')[1]=='npz':
                rawTrainData = MLData(rawTrainFile, rawTrainLabelFile)
            else:
                raise Exception('Could not open ' + rawTrainFile)
            
            
            good_res = np.arange(len(rawTrainData.resIDs))

            iAttens = np.zeros((len(good_res),self.nClass))

            if self.mlDict['trimAttens']:
                good_res = np.delete(good_res, np.where(rawTrainData.opt_iAttens < self.mlDict['attenWinBelow'])[0])

            # rawTrainData.res_indicies = np.zeros((rawTrainData.res_nums,rawTrainData.nClass))
            # for c in range(rawTrainData.nClass):
            #     for i, rn in enumerate(good_res):
            #         rawTrainData.res_indicies[i,c] = rawTrainData.get_peak_idx(rn,iAttens[i,c])
            
            rawTrainData.res_nums = len(good_res)
            
            rawTrainData.iq_vels=rawTrainData.iq_vels[good_res]
            rawTrainData.freqs=rawTrainData.freqs[good_res]
            rawTrainData.Is = rawTrainData.Is[good_res]
            rawTrainData.Qs = rawTrainData.Qs[good_res]
            rawTrainData.resIDs = rawTrainData.resIDs[good_res]
            rawTrainData.attens = rawTrainData.attens[good_res]
            rawTrainData.opt_iAttens = rawTrainData.opt_iAttens[good_res]
            wsAttenInd = np.where(rawTrainData.attens[0]==self.mlDict['wsAtten'])[0][0]
            print wsAttenInd
            
            #class_steps = 300

            trainImages, trainLabels, testImages, testLabels = [], [], [], []

            # num_rotations = 3
            # angle = np.arange(0,2*math.pi,2*math.pi/num_rotations)
            train_ind = np.array(map(int,np.linspace(0,rawTrainData.res_nums-1,rawTrainData.res_nums*self.trainFrac)))
            print type(train_ind), len(train_ind)
            test_ind=[]
            np.array([test_ind.append(el) for el in range(rawTrainData.res_nums) if el not in train_ind])
            print type(test_ind), len(test_ind)
            
            print train_ind[:10], test_ind[:10]


            #TODO: fix append if this is too slow
            for rn in train_ind:#range(int(self.trainFrac*rawTrainData.res_nums)):
                # for t in range(num_rotations):
                #     image = self.makeResImage(res_num = rn, iAtten= iAttens[rn,c], angle=angle[t],showFrames=False, 
                #                                 test_if_noisy=test_if_noisy, xCenter=self.res_indicies[rn,c])
                image, _, _ = mlt.makeResImage(res_num = rn, center_loop=self.mlDict['center_loop'], phase_normalise=False ,showFrames=False, dataObj=rawTrainData, mlDict=self.mlDict, wsAttenInd=wsAttenInd) 
                if image is not None:
                    trainImages.append(image)
                    oneHot = np.zeros(self.mlDict['nAttens'])
                    oneHot[rawTrainData.opt_iAttens[rn]] = 1
                    trainLabels.append(oneHot)

            print rawTrainData.res_nums


            for rn in test_ind:#range(int(self.trainFrac*rawTrainData.res_nums), int(self.trainFrac*rawTrainData.res_nums + self.testFrac*rawTrainData.res_nums)):
                image, _, _ = mlt.makeResImage(res_num = rn, center_loop=self.mlDict['center_loop'], phase_normalise=False, dataObj=rawTrainData, mlDict=self.mlDict, wsAttenInd=wsAttenInd)
                if image is not None:
                    testImages.append(image)
                    oneHot = np.zeros(self.mlDict['nAttens'])
                    oneHot[rawTrainData.opt_iAttens[rn]] = 1
                    testLabels.append(oneHot)

            with open(self.trainFile, 'ab') as tf:
                pickle.dump([trainImages, trainLabels], tf)
                pickle.dump([testImages, testLabels], tf)

    def initializeAndTrainModel(self, debug=False, saveGraph=False):
        print self.trainFile
        if not os.path.isfile(self.trainFile):
            print 'Could not find train file. Making new training images from initialFile'
            self.makeTrainData()

        trainImages, trainLabels, testImages, testLabels = loadPkl(self.trainFile)
        trainImages = np.asarray(trainImages)
        trainLabels = np.asarray(trainLabels)
        testImages = np.asarray(testImages)
        testLabels = np.asarray(testLabels)
        
        print np.sum(trainLabels,axis=0), np.sum(testLabels,axis=0)   

        print np.sum(trainLabels,axis=0), np.sum(testLabels,axis=0)
        print 'Number of training images:', np.shape(trainImages), ' Number of test images:', np.shape(testImages)

        print np.shape(trainLabels)
        print np.sum(trainLabels,axis=0)

        nColors = 2
        if self.mlDict['useIQV']:
            nColors += 1
        if self.mlDict['useMag']:
            nColors += 1

        
        self.x = tf.placeholder(tf.float32, [None, self.mlDict['nAttens'], self.mlDict['xWidth'], nColors], name='inputImage')
        x_image = tf.reshape(self.x, [-1, self.mlDict['nAttens'], self.mlDict['xWidth'], nColors])

        attenWin = 1 + self.mlDict['attenWinBelow'] + self.mlDict['attenWinAbove']
        numImg = tf.shape(x_image)[0]

        def weight_variable(shape):
            #initial = tf.Variable(tf.zeros(shape))
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def conv3d(x, W):
            return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')
        
        def max_pool_nx1(x,n):
            return tf.nn.max_pool(x, ksize=[1, 1, n, 1], strides=[1, 1, n, 1], padding='SAME')

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

        with tf.name_scope('Layer1'):
            num_filt1 = self.mlDict['num_filt1']
            n_pool1 = self.mlDict['n_pool1']
            self.num_filt1 = num_filt1
            W_conv1 = weight_variable([attenWin, self.mlDict['conv_win1'], nColors, num_filt1])
            b_conv1 = bias_variable([num_filt1])
            variable_summaries(W_conv1)
            variable_summaries(b_conv1)
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name='h_conv1')
            h_pool1 = max_pool_nx1(h_conv1,n_pool1)
            h_pool1_dropout = tf.nn.dropout(h_pool1, self.keep_prob)
            xWidth1 = int(math.ceil(self.mlDict['xWidth']/float(n_pool1)))
            cWidth1 = num_filt1
            tf.summary.histogram('h_conv1', h_conv1)
            tf.summary.histogram('h_pool1', h_pool1)
        
        with tf.name_scope('Layer2'):
            num_filt2 = self.mlDict['num_filt2']
            n_pool2 = self.mlDict['n_pool2']
            self.num_filt2 = num_filt2
            W_conv2 = weight_variable([1, self.mlDict['conv_win2'], cWidth1, num_filt2])
            b_conv2 = bias_variable([num_filt2])
            variable_summaries(W_conv2)
            variable_summaries(b_conv2)
            h_conv2 = tf.nn.relu(conv2d(h_pool1_dropout, W_conv2) + b_conv2, name='h_conv2')
            with tf.control_dependencies([tf.assert_equal(tf.shape(h_pool1),(numImg,self.mlDict['nAttens'],xWidth1,cWidth1),message='hpool1 assertion')]):
                h_pool2 = max_pool_nx1(h_conv2, n_pool2)
                h_pool2_dropout = tf.nn.dropout(h_pool2, self.keep_prob)
                xWidth2 = int(math.ceil(xWidth1/float(n_pool2)))
                cWidth2 = num_filt2
            tf.summary.histogram('h_conv2', h_conv2)
            tf.summary.histogram('h_pool2', h_pool2)

        with tf.name_scope('Layer3'):
            num_filt3 = self.mlDict['num_filt3']
            n_pool3 = self.mlDict['n_pool3']
            self.num_filt3 = num_filt3
            W_conv3 = weight_variable([1, self.mlDict['conv_win3'], cWidth2, num_filt3])
            b_conv3 = bias_variable([num_filt3])
            variable_summaries(W_conv3)
            variable_summaries(b_conv3)
            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3, name='h_conv3')
            with tf.control_dependencies([tf.assert_equal(tf.shape(h_pool2),(numImg,self.mlDict['nAttens'],xWidth2,cWidth2),message='hpool2 assertion')]):
                h_pool3 = max_pool_nx1(h_conv3, n_pool3)
                h_pool3_dropout = tf.nn.dropout(h_pool3, self.keep_prob)
                xWidth3 = int(math.ceil(xWidth2/float(n_pool3)))
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
            h_pool3_flat = tf.reshape(h_pool3_dropout,[-1,self.mlDict['nAttens'],cWidth3*xWidth3,1])        
            W_final = weight_variable([1, cWidth3*xWidth3, 1, 1])
            b_final = bias_variable([1])     
            variable_summaries(W_final)
            variable_summaries(b_final)
            
            with tf.control_dependencies([tf.assert_equal(tf.shape(h_pool3),(numImg,self.mlDict['nAttens'],xWidth3,cWidth3))]):
                h_conv_final = tf.nn.conv2d(h_pool3_flat, W_final, strides=[1, 1, 1, 1], padding='VALID') + b_final
            
            with tf.control_dependencies([tf.assert_equal(tf.shape(h_conv_final),(numImg,self.mlDict['nAttens'],1,1))]):
                h_conv_final = tf.reshape(h_conv_final, [-1,self.mlDict['nAttens']])
                tf.summary.histogram('h_conv_final', h_conv_final)
        
        self.y=tf.nn.softmax(h_conv_final, name="outputLabel") #h_fc1_drop   

        y_ = tf.placeholder(tf.float32, [None, self.mlDict['nAttens']]) # true class lables identified by user 
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(self.y+ 1e-10), reduction_indices=[1])) # find optimum solution by minimizing error
        tf.summary.scalar('cross_entropy', cross_entropy)
        #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y,y_))
        #yWeighted = tf.mul(self.y, tf.to_float(tf.range(tf.shape(self.y)[1])))
        #yInd = tf.reduce_sum(yWeighted, reduction_indices=1)
        #y_Ind = tf.to_float(tf.argmax(y_, 1))
        
        #squared_loss = tf.reduce_mean(tf.to_float(tf.square(yInd-y_Ind)))

        train_step = tf.train.AdamOptimizer(self.mlDict['learning_rate']).minimize(cross_entropy) # the best result is when the wrongness is minimal

        init = tf.initialize_all_variables()

        saver = tf.train.Saver()
        
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
        trainReps = self.mlDict['trainReps']
        batches = self.mlDict['batches']
        if np.shape(trainLabels)[0]< batches:
            batches = np.shape(trainLabels)[0]/2
       

        ce_log = []
        acc_log=[]
        print 'Performing', trainReps, 'training repeats, using batches of', batches
        for i in range(trainReps):  #perform the training step using random batches of images and according labels
            batch_xs, batch_ys = next_batch(trainImages, trainLabels, batches) 
            sys.stdout.write('\rbatch: %i ' % (i))
            sys.stdout.flush()

            if i % 500 == 0:
                ce_log.append(self.sess.run(cross_entropy, feed_dict={self.x: batch_xs, y_: batch_ys, self.keep_prob: 1}))
                summary, acc = self.sess.run([merged, accuracy], feed_dict={self.x: testImages, y_: testLabels, self.keep_prob: 1})
                acc_log.append(acc*100)
                test_writer.add_summary(summary, i)
                if saveGraph:
                    graph_writer.add_summary(summary, i)

                print acc*100
                summary, _ = self.sess.run([merged, train_step], feed_dict={self.x: batch_xs, y_: batch_ys, self.keep_prob: self.mlDict['keep_prob']}) #calculate train_step using feed_dict
                train_writer.add_summary(summary, i)

            elif i % 100 == 0:
                acc = self.sess.run(accuracy, feed_dict={self.x: testImages, y_: testLabels, self.keep_prob: 1})
                print acc*100
                acc_log.append(acc*100)
                self.sess.run(train_step, feed_dict={self.x: batch_xs, y_: batch_ys, self.keep_prob: self.mlDict['keep_prob']}) #calculate train_step using feed_dict

            else:  
                self.sess.run(train_step, feed_dict={self.x: batch_xs, y_: batch_ys, self.keep_prob: self.mlDict['keep_prob']}) #calculate train_step using feed_dict
        
        print "--- %s seconds ---" % (time.time() - start_time)

        modelSavePath = os.path.join(self.mlDict['modelDir'], self.mlDict['modelName'])
        print 'Saving model in', modelSavePath
        save_path = saver.save(self.sess, modelSavePath)
        ys_true = self.sess.run(tf.argmax(y_,1), feed_dict={self.x: testImages, y_: testLabels, self.keep_prob: 1})
        y_probs = self.sess.run(self.y, feed_dict={self.x: testImages, y_: testLabels, self.keep_prob: 1})
        ys_guess = np.argmax(y_probs, 1)
        right = []
        within1dB = []
        for i,y in enumerate(ys_true):
            #print i, y, ys_guess[i] 
            if ys_guess[i] == y: # or ys_guess[i] == y-1 or ys_guess[i] == y+1:
                #print i, 'guessed right'
                right.append(i)
            if np.abs(ys_guess[i]-y) <= 1:
                within1dB.append(i)

        print len(right), len(ys_true), float(len(right))/len(ys_true)

        score = self.sess.run(accuracy, feed_dict={self.x: testImages, y_: testLabels, self.keep_prob: 1}) * 100
        print 'Accuracy of model in testing: ', score, '%'
        if score < 85: print 'Consider making more training images'
        print 'Testing accuracy within 1 dB: ', float(len(within1dB))/len(ys_true)*100, '%' 

        score = self.sess.run(accuracy, feed_dict={self.x: trainImages, y_: trainLabels, self.keep_prob: 1}) * 100
        print 'Accuracy of model in training: ', score, '%'
        
        np.savez(modelSavePath+'_confusion.npz', ys_true=ys_true, ys_guess=ys_guess, y_probs=y_probs, ce=ce_log, acc=acc_log)

        del trainImages, trainLabels, testImages, testLabels

    def plotW_fc2(self, weights):
        s = np.shape(weights)
        print s
        fig2 = plt.figure(figsize=(8.0, 5.0))
        #for f in range(s[0]):
        for nc in range(self.nClass):
            fig2.add_subplot(2,2,nc+1)
            plt.plot(weights[:,nc])
            plt.title('class %i' % nc)
        plt.show()
        #plt.close()

    # def plotWeights(self):
    #     '''creates a 2d map showing the positive and negative weights for each class'''
    #     weights = self.sess.run(self.W)
    #     weights = np.reshape(weights,(self.mlDict['xWidth'],self.mlDict['xWidth'],self.nClass))
    #     weights = np.flipud(weights)
    #     for nc in range(self.nClass):
    #         plt.imshow(weights[:,:, nc])
    #         plt.title('class %i' % nc)
    #         plt.show()
    #         plt.close()

    def plotActivations(self, layer, layername='lol', step=0):
        '''creates a 2d map showing the positive and negative weights for each class'''
        #weights = self.sess.run(self.W_fc2)
        #weights = np.reshape(weights,(math.sqrt(self.fc_filters),math.sqrt(self.fc_filters),self.nClass))
        #weights = np.flipud(weights)
        shape = np.shape(layer)
        print 'layer shape', shape
        
        fig2 = plt.figure(figsize=(8.0, 5.0))
        for nc in range(np.shape(layer)[3]):
            fig2.add_subplot(4,np.shape(layer)[3]/4+1,nc+1)
            plt.imshow(layer[:,0,:, nc]**2)#i*int(shape[0]/3)
            #fig2.title('class %i' % nc)
        #plt.savefig('actv_layer_%s_%i_s%i'%(layername,i,step))
        plt.show()
        #plt.close()



    def plotWeights(self, weights):
        '''creates a 2d map showing the positive and negative weights for each class'''
        import math
        #weights = self.sess.run(self.W_fc2)
        #weights = np.reshape(weights,(math.sqrt(self.fc_filters),math.sqrt(self.fc_filters),self.nClass))
        #weights = np.flipud(weights)
        print np.shape(weights)
        for nc in range(np.shape(weights)[3]):
            plt.subplot(2,np.shape(weights)[3]/2+1,nc+1)
            #plt.imshow(weights[:,0,:, nc])
            plt.imshow(weights[:,:,0, nc])
            #plt.title('class %i' % nc)
        plt.show()
        #plt.close()
    

    
def next_batch(trainImages, trainLabels, batch_size):
    '''selects a random batch of batch_size from trainImages and trainLabels'''
    perm = random.sample(range(len(trainImages)), batch_size)
    trainImagesBatch = trainImages[perm]
    trainLabelsBatch = trainLabels[perm]
    #print 'next_batch trImshape', np.shape(trainImages)
    return trainImagesBatch, trainLabelsBatch


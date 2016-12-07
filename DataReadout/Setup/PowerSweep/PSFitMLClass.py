''' 
Author Rupert Dodkins

A script to automate the identification of resonator attenuations normally performed by PSFit.py. This is accomplished 
using Google's Tensor Flow machine learning package which implements a pattern recognition convolution neural network 
algorithm and classification algorithm on power and frequency sweep data saved in h5 format.

Usage:  python PSFit_ml.py 20160712/ps_r115_FL1_1_20160712-225809.h5 20160720/ps_r118_FL1_b_pos_20160720-231653.h5

Inputs:
20160712/ps_r115_FL1_1.txt:                    list of resonator frequencies and correct attenuations
20160712/ps_r115_FL1_1_20160712-225809.h5:     corresponding powersweep file
20160720/ps_r118_FL1_b_pos_20160720-231653.h5: powersweep file the user wishes to infer attenuations for

Intermediaries:
SDR/Setup/ps_peaks_train_w<x>_s<y>.pkl:        images and corresponding labels used to train the algorithm

Outputs: 
20160712/ps_r115_FL1_1.pkl:                        frequencies, IQ velocities, Is, Qs, attenuations formatted for quick use
20160720/ps_r118_FL1_b_pos.txt:                    final list of frequencies and attenuations

How it works:
For every resonator and attenuation an input "image" is made of I,Q and v_iq. This 3 row x spectral-width image is input into 
the trained neural network and a clasifier algorithm assigns a probability that the loop being either saturated or unsaturated.
The non-saturation probability profile of each resonator should look roughly like a sigmoid function with the transition 
happening on the just before the attenuation loop becomes non-satuarated

This list of attenuation values and frequencies are either fed PSFit.py to checked manually or dumped to 
ps_r118_FL1_b_pos.txt

The machine learning algorithm requires a series of images to train and test the algorithm with. If they exist the image 
data will be loaded from a train pkl file

Alternatively, if the user does not have a train pkl file but does have a powersweep file and corresponding list of 
resonator attenuations this should be used as the initial file and training data will be made. 

These new image data will be saved as pkl files (or appened to existing pkl files) and reloaded

The machine is then trained and its ability to predict the type of image is validated

The weights used to make predictions for each class can be displayed using the plotWeights function as well as the activations

'''
import os,sys,inspect
from PSFit import *
from iqsweep import *
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm
import tensorflow as tf
import pickle
import random
import time
import math
from scipy import interpolate
from PSFitMLData import *
np.set_printoptions(threshold=np.inf)

#removes visible depreciation warnings from lib.iqsweep
import warnings
warnings.filterwarnings("ignore")


class mlClassification():
    def __init__(self,  initialFile=None, inferenceFile=None):
        '''
        Implements the machine learning pattern recognition algorithm on IQ velocity data as well as other tests to 
        choose the optimum attenuation for each resonator
        '''
        self.nClass = 2
        self.xWidth = 40#np.shape(res1_freqs[1])
        self.scalexWidth = 1

        self.initialFile = initialFile
        self.baseFile = ('.').join(initialFile.split('.')[:-1])
        self.PSFile = self.baseFile[:-16] + '.txt'#os.environ['MKID_DATA_DIR']+'20160712/ps_FL1_1.txt' # power sweep fit, .txt 
        self.mldir = './machine_learning_metadata/' 

        self.trainFile = 'ps_peaks_train_iqv_mag_c%ix.pkl' % (self.nClass) 
        self.trainFrac = 0.8
        self.testFrac=1 - self.trainFrac
        
        if not inferenceFile is None:
            print 'Inference File:', inferenceFile
            self.inferenceData = PSFitMLData(h5File = inferenceFile, useAllAttens = False)

    def makeResImage(self, res_num, iAtten, angle=0,  phase_normalise=False, showFrames=False, dataObj=None):
        '''Creates a table with 3 rows: I, Q, and vel_iq for makeTrainData()

        inputs 
        res_num: index of resonator in question
        iAtten: index of attenuation in question
        angle: angle of rotation about the origin (radians)
        showFrames: pops up a window of the frame plotted using matplotlib.plot
        '''     
        if dataObj is None:
            if self.inferenceData is None:
                raise ValueError('Initialize dataObj first!')
            dataObj = self.inferenceData
        
        xWidth= self.xWidth 
        scalexWidth = self.scalexWidth

        xCenter = self.get_peak_idx(res_num,iAtten,dataObj)
        start = int(xCenter - xWidth/2)
        end = int(xCenter + xWidth/2)

        if start < 0:
            start_diff = abs(start)
            start = 0
            iq_vels = dataObj.iq_vels[res_num, iAtten, start:end]
            iq_vels = np.lib.pad(iq_vels, (start_diff,0), 'constant', constant_values=(0))
            Is = dataObj.Is[res_num,iAtten,start:end]
            Is = np.lib.pad(Is, (start_diff,0), 'constant', constant_values=(Is[0]))
            Qs = dataObj.Qs[res_num,iAtten,start:end]
            Qs = np.lib.pad(Qs, (start_diff,0), 'constant', constant_values=(Qs[0]))
        elif end >= np.shape(dataObj.freqs)[1]:
            iq_vels = dataObj.iq_vels[res_num, iAtten, start:end]
            iq_vels = np.lib.pad(iq_vels, (0,end-np.shape(dataObj.freqs)[1]+1), 'constant', constant_values=(0))
            Is = dataObj.Is[res_num,iAtten,start:end]
            Is = np.lib.pad(Is, (0,end-np.shape(dataObj.freqs)[1]), 'constant', constant_values=(Is[-1]))
            Qs = dataObj.Qs[res_num,iAtten,start:end]
            Qs = np.lib.pad(Qs, (0,end-np.shape(dataObj.freqs)[1]), 'constant', constant_values=(Qs[-1]))
        else:
            iq_vels = dataObj.iq_vels[res_num, iAtten, start:end]
            Is = dataObj.Is[res_num,iAtten,start:end]
            Qs = dataObj.Qs[res_num,iAtten,start:end]
        #iq_vels = np.round(iq_vels * xWidth / max(dataObj.iq_vels[res_num, iAtten, :]) )
        iq_vels = iq_vels / np.amax(dataObj.iq_vels[res_num, :, :])


        # interpolate iq_vels onto a finer grid
        if scalexWidth!=None:
            x = np.arange(0, xWidth+1)
            iq_vels = np.append(iq_vels, iq_vels[-1])
            f = interpolate.interp1d(x, iq_vels)
            xnew = np.arange(0, xWidth, scalexWidth)
            iq_vels = f(xnew)/ scalexWidth

            Is = np.append(Is, Is[-1])
            f = interpolate.interp1d(x, Is)
            Is = f(xnew)/ scalexWidth
            
            Qs = np.append(Qs, Qs[-1])
            f = interpolate.interp1d(x, Qs)
            Qs = f(xnew)/ scalexWidth

        xWidth = int(xWidth/scalexWidth)

        res_mag = math.sqrt(np.amax(dataObj.Is[res_num, :, :]**2 + dataObj.Qs[res_num, :, :]**2))
        Is = Is / res_mag
        Qs = Qs / res_mag

        # Is = Is /np.amax(dataObj.iq_vels[res_num, :, :])
        # Qs = Qs /np.amax(dataObj.iq_vels[res_num, :, :])

        # Is = Is /np.amax(dataObj.Is[res_num, :, :])
        # Qs = Qs /np.amax(dataObj.Qs[res_num, :, :])

        if phase_normalise:
            #mags = Qs**2 + Is**2
            #mags = map(lambda x: math.sqrt(x), mags)#map(lambda x,y:x+y, a,b)

            #peak_idx = self.get_peak_idx(res_num,iAtten)
            peak_idx =argmax(iq_vels)
            #min_idx = argmin(mags)

            phase_orig = math.atan2(Qs[peak_idx],Is[peak_idx])
            #phase_orig = math.atan2(Qs[min_idx],Is[min_idx])

            angle = -phase_orig

        rotMatrix = numpy.array([[numpy.cos(angle), -numpy.sin(angle)], 
                                 [numpy.sin(angle),  numpy.cos(angle)]])

        Is,Qs = np.dot(rotMatrix,[Is,Qs])

        if showFrames:
            fig = plt.figure(frameon=False,figsize=(15.0, 5.0))
            fig.add_subplot(131)
            plt.plot(iq_vels)
            plt.ylim(0,1)
            fig.add_subplot(132)
            plt.plot(Is)
            plt.plot(Qs)
            fig.add_subplot(133)
            plt.plot(Is,Qs)
            plt.show()
            plt.close()

        image = np.zeros((3,len(Is)))
        image[0,:] = Is
        image[1,:] = Qs
        image[2,:] = iq_vels

        return image

    def get_peak_idx(self,res_num,iAtten,dataObj=None):
        if dataObj is None:
            if self.inferenceData is None:
                raise ValueError('Initialize dataObj first!')
            dataObj = self.inferenceData
        return argmax(dataObj.iq_vels[res_num,iAtten,:])

    def makeTrainData(self, cherry_pick=False):                
        '''
        Creates the training (and testing) images and labels to be passed to mlClass() which is saved in a .pkl and read using loadPkl() 

        Cheery pick argument allows the user to check the data and choose more appropriate examples of sat and non sat loops
        '''
        
        rawTrainData = PSFitMLData(h5File = self.initialFile, useResID=False)
        rawTrainData.loadTrainData()
        
        good_res = np.arange(len(rawTrainData.resIDs))

        # attDist = np.arange(-2,1,2)
        attDist = [-1,0]

        iAttens = np.zeros((len(good_res),self.nClass))
        print 'iAttens shape', np.shape(iAttens)
        print 'opt_iAttens shape', np.shape(rawTrainData.opt_iAttens)

        # for ia in range(len(rawTrainData.opt_iAttens)):
        #     attDist = int(np.random.normal(2, 1, 1))
        #     if attDist <1: attDist = 2
        #     iAttens[ia,0] = rawTrainData.opt_iAttens[ia] - attDist
        #     iAttens[ia,1] = rawTrainData.opt_iAttens[ia] + attDist
            
        # iAttens[:,1] = rawTrainData.opt_iAttens 

        for i in range(self.nClass):
            iAttens[:,i] =rawTrainData.opt_iAttens + attDist[i] 

        rawTrainData.res_nums = len(good_res)

        # remove resonators in training where the attenuation lies outside the full range
        lb_rej = np.where(iAttens[:,0]<0)[0]
        if len(lb_rej) != 0:
            iAttens = np.delete(iAttens,lb_rej,axis=0) # when index is below zero
            print len(iAttens)
            good_res = np.delete(good_res,lb_rej)
            rawTrainData.res_nums = rawTrainData.res_nums-len(lb_rej)
        
        ub_rej = np.where(iAttens[:,1]>len(rawTrainData.attens))[0]
        if len(ub_rej) != 0:
            iAttens = np.delete(iAttens,ub_rej,axis=0) 
            print len(iAttens)
            good_res = np.delete(good_res,ub_rej)
            rawTrainData.res_nums = rawTrainData.res_nums-len(ub_rej)
       
        rawTrainData.iq_vels=rawTrainData.iq_vels[good_res]
        rawTrainData.freqs=rawTrainData.freqs[good_res]
        rawTrainData.Is = rawTrainData.Is[good_res]
        rawTrainData.Qs = rawTrainData.Qs[good_res]
        rawTrainData.resIDs = rawTrainData.resIDs[good_res]
        rawTrainData.attens = rawTrainData.attens[good_res]
        
        trainImages, trainLabels, testImages, testLabels = [], [], [], []

        # select resonators uniformally distributed across the range for training and testing
        train_ind = np.array(map(int,np.linspace(0,rawTrainData.res_nums-1,rawTrainData.res_nums*self.trainFrac)))
        test_ind=[]
        np.array([test_ind.append(el) for el in range(rawTrainData.res_nums) if el not in train_ind])

        if not cherry_pick:
            for rn in train_ind:
                for c in range(self.nClass):
                    image = self.makeResImage(res_num = rn, iAtten= iAttens[rn,c], phase_normalise=True ,showFrames=False, dataObj=rawTrainData) 
                    trainImages.append(image)
                    one_hot = np.zeros(self.nClass)
                    one_hot[c] = 1
                    trainLabels.append(one_hot)

            for rn in test_ind:
                for c in range(self.nClass):
                    image = self.makeResImage(res_num = rn, iAtten= iAttens[rn,c], phase_normalise=True, dataObj=rawTrainData)
                    testImages.append(image)
                    one_hot = np.zeros(self.nClass)
                    one_hot[c] = 1
                    testLabels.append(one_hot)
        
        else:
            self.selectTrainData(train_ind,test_ind,iAttens)

        append = None
        if os.path.isfile(self.mldir+self.trainFile): 
            append = raw_input('Do you want to append this training data to previous data [y/n]')
        if (append  == 'n'):
            self.trainFile = self.trainFile.split('-')[0]+time.strftime("-%Y-%m-%d-%H-%M-%S")
        if (append  == 'y') or (os.path.isfile(self.trainFile)== False):
            print 'saving %s to %s' % (self.mldir+self.trainFile, os.path.dirname(os.path.abspath(self.trainFile)) )
            with open(self.mldir+self.trainFile, 'ab') as tf:
                pickle.dump([trainImages, trainLabels], tf)
                pickle.dump([testImages, testLabels], tf)

    def selectTrainData(self,train_ind,test_ind,iAttens):
        '''
        A function to allow the user to monitor the data that goes into the training and test set
        during makeTrainData().
        Each time a resonator appears the user can do one of six options...

        Inputs:
        y: the resonator matches the class allocation and should be included
        n: res is not saturated yet (or not non-sat yet if c==1), -1 from iAtten (or +1) and display
        o: the same as n except the next res will not be displayed and only One step can be made
        r: do not include this resonator in the training (the sat loop will still be included if already confirmed) 
        b: remove the last loop and label in the group and reassess
        q: stop adding to the group (train or test) before all the res have been checked
        '''

        trainImages, trainLabels, testImages, testLabels = [], [], [], []
        catagory = ['saturated', 'non-sat']
        group = [train_ind,test_ind]
        include = 'n'

        for t in range(2):
            print 'Should I include this resonator in the training data? [y/n (r/o/b/q)]'
            resonators = group[t]
            print resonators
            ir = 0
            while ir < len(resonators):
                rn = resonators[ir]
                print iAttens[:10], trainLabels[:10], group[t][:10]
                for c in range(self.nClass):
                    if include =='r': 
                        include = 'n'                            
                        break
                    if include == 'b':
                        include = 'n'
                        break

                    include ='n'
                    while include == 'n':
                        print rn, c, catagory[c], iAttens[rn,c]
                        image = self.makeResImage(res_num = rn, iAtten= iAttens[rn,c], showFrames=True)   
                        include = raw_input()

                        if include == 'q':
                            return
                        if include == 'r':
                            # iAttens[rn,:] = [-1,-1]
                            if group == 'test':
                                self.test_ind= np.delete(self.test_ind,ir)
                            break
                        if include =='n':
                            if c==0: iAttens[rn,c] -= 1
                            else: iAttens[rn,c] += 1

                        if include == 'o':
                            if c==0: iAttens[rn,c] -= 1
                            else: iAttens[rn,c] += 1
                            include = 'y'
                    
                        if include == 'b':
                            ir -= 2
                            print 'here'
                            if t == 0:
                                trainImages = trainImages[:-1]
                                trainLabels = trainLabels[:-1]
                            else:
                                testImages = testImages[:-1]
                                testLabels = testLabels[:-1]            
                            break

                    if include == 'y':
                        one_hot = np.zeros(self.nClass)
                        one_hot[c] = 1
                        if t == 0:
                            trainImages.append(image)
                            trainLabels.append(one_hot)
                        else:
                            testImages.append(image)
                            testLabels.append(one_hot)
                
                ir += 1
        return trainImages, trainLabels, testImages, testLabels

    def mlClass(self, learning_rate = -3.5, showFrames =False, accuracy_plot='post', 
                plot_missed=False, plot_weights='', plot_activations=''):       
        '''Code adapted from the tensor flow MNIST CNN tutorial.
        
        Using the training inputs and labels the machine learning class (mlClass) applies a probability to an input image (I,Q,v_iq) belong to the saturated class

        The training and test matricies are loaded from file (those made earlier if chosen to not be appended to file 
        will not be used)

        Inputs
        learning rate: 10**learning_rate is the input for AdamOptimizer
        accuracy plot: shows accuracy and cross entropy with training iterations for train and test data. 
                        post - plots the graph ex post facto of the training
                        real - plots after each 100 iterations of the model training
                        '' - off
        plot weights: plot the filter layers of the CNN (same arguments apply as accuracy plot)
        plot activations: plot the 'image' layers after the activation function (same arguments apply as accuracy plot)
        plot missed: in order to identify the loops from the training data the algorithm incorrectly predicted 
        '''
        
        if not os.path.isfile(self.mldir+self.trainFile):
            print 'Could not find train file. Making new training images from initialFile'
            self.makeTrainData()

        trainImages, trainLabels, testImages, testLabels = loadPkl(self.mldir+self.trainFile)

       
        print 'Number of training images:', np.shape(trainImages), ' Number of test images:', np.shape(testImages)
   
        if self.scalexWidth != 1:
            self.xWidth = int(self.xWidth/self.scalexWidth)
        if np.shape(trainImages)[2]!=self.xWidth:
            print 'Please make new training images of the correct size'
            exit()
              
        self.x = tf.placeholder(tf.float32, [None, 3, self.xWidth])
        
        self.x_image = tf.reshape(self.x, [-1,3,self.xWidth,1])
        
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 3, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

        #  CNN network
        self.num_filt1 = 3
        self.W_conv1 = weight_variable([3, 5, 1, self.num_filt1])
        b_conv1 = bias_variable([self.num_filt1])
        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + b_conv1)

        self.h_pool1 = max_pool_2x2(self.h_conv1)

        num_filt2 = 5
        self.W_conv2 = weight_variable([3, 5, self.num_filt1, num_filt2])
        b_conv2 = bias_variable([num_filt2])
        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + b_conv2)        

        self.h_pool2 = max_pool_2x2(self.h_conv2)
        

        # fully connected network
        self.fc_filters = 5**2
        num_fc_filt = int(math.ceil(math.ceil(np.shape(trainImages)[2]/2)/2)  * np.shape(trainImages)[1] * num_filt2)

        W_fc1 = weight_variable([num_fc_filt, self.fc_filters]) 
        b_fc1 = bias_variable([self.fc_filters])
        h_pool2_flat = tf.reshape(self.h_pool2, [-1, num_fc_filt])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        self.W_fc2 = weight_variable([self.fc_filters, self.nClass])
        b_fc2 = bias_variable([self.nClass])    

        self.y=tf.nn.softmax(tf.matmul(h_fc1_drop, self.W_fc2) + b_fc2) 
        y_ = tf.placeholder(tf.float32, [None, self.nClass]) # true class lables identified by user 
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(self.y+ 1e-10), reduction_indices=[1])) # find optimum solution by minimizing error

        train_step = tf.train.AdamOptimizer(10**learning_rate).minimize(cross_entropy) # the best result is when the wrongness is minimal

        init = tf.initialize_all_variables()

        saver = tf.train.Saver()
        
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1)) #which ones did it get right?
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print self.trainFile
        modelName = ('.').join(self.trainFile.split('.')[:-1]) + '.ckpt'
        print modelName

        if os.path.isfile("%s%s" % (self.mldir,modelName)):
            self.sess = tf.Session()
            self.sess.run(init)           

            # Restore variables from disk.
            saver.restore(self.sess, "%s%s" % (self.mldir,modelName) )
        else:
            self.sess = tf.Session()
            self.sess.run(init) # need to do this everytime you want to access a tf variable (for example the true class labels calculation or plotweights)

            
            def accuracyPlot():
                x.append(i)
                # y.append(temp_y)
                # plt.plot(x, y)

                fig.add_subplot(121)
                plt.plot(x,train_ce, label='train')
                plt.plot(x,test_ce, label='test')
                # plt.legend(loc='upper right')
                fig.add_subplot(122)
                plt.plot(x,train_acc, label='train')
                plt.plot(x,test_acc, label='test')
                # plt.legend(loc='lower right')
                plt.show()
                plt.pause(0.0001) #Note this correction

            start_time = time.time()
            trainReps = 3000
            batches = 50
            if np.shape(trainLabels)[0]< batches:
                batches = np.shape(trainLabels)[0]/2

            score = 0
            entropy = 1
            train_ce =[]
            train_acc=[]
            test_ce = []
            test_acc=[]
            loop = 0

            if accuracy_plot == 'real time':
                plt.ion() ## Note this correction
                fig = plt.figure(frameon=False,figsize=(15.0, 5.0))
                x=list()

            print 'Performing', trainReps, 'training repeats, using batches of', batches
            for i in range(trainReps):  #perform the training step using random batches of images and according labels
                batch_xs, batch_ys = next_batch(trainImages, trainLabels, batches) 

                if i % 100 == 0:
                    entropy = self.sess.run(cross_entropy, feed_dict={self.x: batch_xs, y_: batch_ys, self.keep_prob:1})
                    train_ce.append(entropy)
                    train_score = self.sess.run(accuracy, feed_dict={self.x: batch_xs, y_: batch_ys,self.keep_prob:1}) * 100
                    train_acc.append(train_score)
                    print i, entropy, train_score,
                
                # if i % 1000 ==0:
                    entropy = self.sess.run(cross_entropy, feed_dict={self.x: testImages, y_: testLabels,self.keep_prob:1})
                    test_ce.append(entropy)
                    test_score = self.sess.run(accuracy, feed_dict={self.x: testImages, y_: testLabels, self.keep_prob:1}) * 100
                    test_acc.append(test_score)
                    print entropy, test_score
                    
                    if accuracy_plot == 'real time':
                        accuracy_plot()

                    if plot_weights == 'real time':
                        self.plotWeights()

                    if plot_activations == 'real time':
                        self.plotActivations()
                
                self.sess.run(train_step, feed_dict={self.x: batch_xs, y_: batch_ys, self.keep_prob:0.1}) #calculate train_step using feed_dict
                if entropy < 0.1:
                    break
                # if test_score > 85:
                #     break

            score = self.sess.run(accuracy, feed_dict={self.x: testImages, y_: testLabels, self.keep_prob:1}) * 100
            print 'Accuracy of model in testing: ', score, '%'
            # loop += 1
            # if loop > 50: 
            #     print 'Consider changing the training images or lowering the learning rate'
            #     break
      
            print "--- %s seconds ---" % (time.time() - start_time)
            if accuracy_plot == 'post':
                fig = plt.figure(frameon=False,figsize=(15.0, 5.0))
                fig.add_subplot(121)
                plt.plot(train_ce, label='train')
                plt.plot(test_ce, label='test')
                plt.legend(loc='upper right')
                fig.add_subplot(122)
                plt.plot(train_acc, label='train')
                plt.plot(test_acc, label='test')
                plt.legend(loc='lower right')
                plt.show()

            print "%s%s" % (self.mldir,modelName)
            save_path = saver.save(self.sess, "%s%s" % (self.mldir,modelName))
            print("Model saved in file: %s" % save_path)

        ys_true = self.sess.run(tf.argmax(y_,1), feed_dict={self.x: testImages, y_: testLabels})
        ys_guess = self.sess.run(tf.argmax(self.y,1), feed_dict={self.x: testImages, y_: testLabels, self.keep_prob:1})

        print 'true class labels: ', ys_true
        print 'class estimates:   ', ys_guess

        print np.sum(self.sess.run(self.y, feed_dict={self.x: testImages, y_: testLabels, self.keep_prob:1}),axis=0)
        missed = []
        
        if plot_missed:
            for i,y in enumerate(ys_true):
                if ys_guess[i] != y:
                    missed.append(i)

            res_per_win = 8
            for f in range(int(np.ceil(len(missed)/res_per_win))+1):
                _, axarr = plt.subplots(2,res_per_win,figsize=(16, 4))
                for ir, rm in enumerate(missed[f*res_per_win: (f+1)*res_per_win]):
                    if ys_guess[rm] == 0: title = 'FP' 
                    else: title = 'FN' 
                    axarr[0,ir].set_title(title)
                    axarr[0,ir].plot(testImages[rm][2,:])
                    axarr[1,ir].plot(testImages[rm][0,:],testImages[rm][1,:], '-o')
                    
                plt.show()
                plt.close()
        

        score = self.sess.run(accuracy, feed_dict={self.x: testImages, y_: testLabels, self.keep_prob:1}) * 100
        print 'Accuracy of model in testing: ', score, '%'
        if score < 85: print 'Consider making more training images'

        if plot_weights == 'post':
            self.plotWeights()

        if plot_activations == 'post':
            self.plotActivations()

    def plotWeights(self):
        '''creates a 2d map showing the positive and negative weights for each class'''
        weights = [self.sess.run(self.W_conv1), self.sess.run(self.W_conv2)]
        f, axarr = plt.subplots(2,self.num_filt1,figsize=(20.0, 4))
        for i, w in enumerate(weights):
            # print np.shape(w)
            for filt in range(self.num_filt1):
                # plt.subplot(4,3,(i+1)*(row+1))
                axarr[i,filt].imshow(w[:,:,0,filt], cmap=cm.coolwarm, interpolation='none',aspect='auto')
                # plt.plot(weights[0,:,0, nc])
                plt.title(' %i' % i)
        plt.show()
        plt.close()

    def plotActivations(self):
        '''creates a 2d map showing the positive and negative weights for each class'''
        _,_,testImages,_ = loadPkl(self.mldir+self.trainFile)

        activations = [self.sess.run(self.x_image,feed_dict={self.x: testImages}), 
                    self.sess.run(self.h_conv1, feed_dict={self.x: testImages}),
                    self.sess.run(self.h_pool1, feed_dict={self.x: testImages}),
                    self.sess.run(self.h_conv2, feed_dict={self.x: testImages}),
                    self.sess.run(self.h_pool2, feed_dict={self.x: testImages})]
        for r in range(len(testImages)):
            f, axarr = plt.subplots(len(activations)+2,3,figsize=(8.0, 8.1))
            axarr[0,1].plot(activations[0][r,0,:,0],activations[0][r,1,:,0])
            for row in range(3):
                axarr[1,row].plot(activations[0][r,row,:,0])
                for i, a in enumerate(activations):
                    axarr[i+2,row].imshow(np.rot90(a[r,row,:,:]), cmap=cm.coolwarm,interpolation='none',aspect='auto')
            
            plt.show()
            plt.close()
    
    def checkLoopAtten(self, res_num, iAtten, showFrames=False):
        '''A function to analytically check the properties of an IQ loop: saturatation and smoothness.

        To check for saturation if the ratio between the 1st and 2nd largest edge is > max_ratio_threshold.
        
        Another metric which is more of a proxy is if the angle on either side of the sides connected to the 
        longest edge is < min_theta or > max_theta the loop is considered saturated. 

        A True result means that the loop is unsaturated.

        Inputs:
        res_num: index of resonator in question
        iAtten: index of attenuation in question
        showLoop: pops up a window of the frame plotted 

        Output:
        Theta 1 & 2: used as a proxy for saturation
        Max ratio: the ratio of highest and 2nd highest v_iq - a more reliable indicator of saturation
        vels: angles every 3 points make around the loop. The closer each to ~ 160 deg the smoother the loop
        '''
        vindx = (-self.inferenceData.iq_vels[res_num,iAtten,:]).argsort()[:1]
        if vindx == 0:
            max_neighbor = self.inferenceData.iq_vels[res_num, iAtten,1]
        elif vindx == len(self.inferenceData.iq_vels[res_num,iAtten,:])-1:
            max_neighbor = self.inferenceData.iq_vels[res_num,iAtten,vindx-1]
        else:
            max_neighbor = maximum(self.inferenceData.iq_vels[res_num,iAtten,vindx-1],self.inferenceData.iq_vels[res_num, iAtten,vindx+1])

        max_theta_vel  = math.atan2(self.inferenceData.Qs[res_num,iAtten,vindx[0]-1] - self.inferenceData.Qs[res_num,iAtten,vindx[0]], 
                                    self.inferenceData.Is[res_num,iAtten,vindx[0]-1] - self.inferenceData.Is[res_num,iAtten,vindx[0]])
        low_theta_vel = math.atan2(self.inferenceData.Qs[res_num,iAtten,vindx[0]-2] - self.inferenceData.Qs[res_num,iAtten,vindx[0]-1], 
                                   self.inferenceData.Is[res_num,iAtten,vindx[0]-2] - self.inferenceData.Is[res_num,iAtten,vindx[0]-1])
        upp_theta_vel = math.atan2(self.inferenceData.Qs[res_num,iAtten,vindx[0]] - self.inferenceData.Qs[res_num,iAtten,vindx[0]+1], 
                                   self.inferenceData.Is[res_num,iAtten,vindx[0]] - self.inferenceData.Is[res_num,iAtten,vindx[0]+1])

        theta1 = (math.pi + max_theta_vel - low_theta_vel)/math.pi * 180
        theta2 = (math.pi + upp_theta_vel - max_theta_vel)/math.pi * 180

        theta1 = abs(theta1)
        if theta1 > 360:
            theta1 = theta1-360
        theta2= abs(theta2)
        if theta2 > 360:
            theta2 = theta2-360
        

        max_ratio = (self.inferenceData.iq_vels[res_num,iAtten,vindx[0]]/ max_neighbor)[0]

        if showFrames:
            plt.plot(self.inferenceData.Is[res_num,iAtten,:],self.inferenceData.Qs[res_num,iAtten,:], 'g.-')
            plt.show()
        
        vels = np.zeros((len(self.inferenceData.Is[res_num,iAtten,:])-2))
        # for i,_ in enumerate(vels[1:-1]):
        for i,_ in enumerate(vels, start=1):
            low_theta_vel = math.atan2(self.inferenceData.Qs[res_num,iAtten,i-1] - self.inferenceData.Qs[res_num,iAtten,i], 
                                       self.inferenceData.Is[res_num,iAtten,i-1] - self.inferenceData.Is[res_num,iAtten,i])
            if low_theta_vel < 0: 
                low_theta_vel = 2*math.pi+low_theta_vel
            upp_theta_vel = math.atan2(self.inferenceData.Qs[res_num,iAtten,i+1] - self.inferenceData.Qs[res_num,iAtten,i], 
                                       self.inferenceData.Is[res_num,iAtten,i+1] - self.inferenceData.Is[res_num,iAtten,i])
            if upp_theta_vel < 0: 
                upp_theta_vel = 2*math.pi+upp_theta_vel
            vels[i-1] = abs(upp_theta_vel- low_theta_vel)/math.pi * 180

        return [theta1, theta2, max_ratio, vels]
   
    
    def checkResAtten(self, res_num, plotAngles=False, showResData=False, min_theta = 115, max_theta = 220, max_ratio_threshold = 2.5):
        '''
        Outputs useful properties about each resonator using checkLoopAtten.
        Figures out if a resonator is bad using the distribution of angles around the loop
        Analytically finds the attenuation values when the resonator is saturated using the max ratio metric and adjacent angles to max v_iq line metric 

        Inputs:
        min/max_theta: limits outside of which the loop is considered saturated
        max_ratio_threshold: maximum largest/ 2nd largest IQ velocity allowed before loop is considered saturated
        showFrames: plots all the useful information on one plot for the resonator

        Oututs:
        Angles non sat: array of bools (true is non sat)
        Ratio non sat: array of bools (true is non sat)
        Ratio: ratio in v_iq between 1st and next highest adjacent max
        Running ratio: Ratio but smoothed using a running average
        Bad res: using the distribution of angles bad resonators are identified (true is bad res)
        Angles mean center: the mean of the angles around the center of the distribution (should be ~ 160)
        Angles std center: the standard dev of the angles. In the center they should follow a gauss dist and the tighter the better 
        '''
        max_ratio_threshold = linspace(0,max_ratio_threshold*7,int(len(self.inferenceData.attens)))
        # max_ratio = self.inferenceData.iq_vels[res_num,iAtten,vindx[0]]/ self.inferenceData.iq_vels[res_num,iAtten,vindx[1]]

        max_theta = linspace(max_theta,max_theta*1.2,int(len(self.inferenceData.attens)))
        min_theta = linspace(min_theta,min_theta/1.2,int(len(self.inferenceData.attens)))

        angles = np.zeros((len(self.inferenceData.attens),2))
        ratio = np.zeros(len(self.inferenceData.attens))

        angles_nonsat = np.ones(len(self.inferenceData.attens))
        ratio_nonsat = np.zeros(len(self.inferenceData.attens))


        running_ratio = np.zeros((len(self.inferenceData.attens)))  

        vels = np.zeros((np.shape(self.inferenceData.iq_vels[0])[0], np.shape(self.inferenceData.iq_vels[0])[1]-1))

        for ia, _ in enumerate(self.inferenceData.attens):
            loop_sat_cube = self.checkLoopAtten(res_num=res_num,iAtten=ia, showFrames=False)
            angles[ia,0], angles[ia,1], ratio[ia], vels[ia] = loop_sat_cube
            ratio_nonsat[ia] = ratio[ia] < max_ratio_threshold[ia]
            
        angles_running =  np.ones((len(self.inferenceData.attens),2))*angles[0,:]
        for ia in range(1,len(self.inferenceData.attens)):
            angles_running[ia,0] = (angles[ia,0] + angles_running[ia-1,0])/2
            angles_running[ia,1] = (angles[ia,1] + angles_running[ia-1,1])/2
            # running_ratio[-ia] = np.sum(ratio[-ia-1: -1])/ia
            running_ratio[-ia-1] = (running_ratio[-ia] + ratio[-ia-1])/2

        # for ia in range(1,len(self.inferenceData.attens)-2):
        #     diff_rr[ia] = sum(running_ratio[ia-1:ia])-sum(running_ratio[ia+1:ia])

        for ia in range(len(self.inferenceData.attens)/2):
            angles_nonsat[ia] = (max_theta[ia] > angles_running[ia,0] > min_theta[ia]) and (max_theta[ia] > angles_running[ia,1] > min_theta[ia])

        angles_mean_center = np.mean(vels[:,35:115], axis =1)
        angles_std_center = np.std(vels[:,35:115], axis=1)
        angles_mean = np.mean(vels,axis=1)
        angles_std = np.std(vels,axis=1)

        delim = np.shape(vels)[1]/3

        y, x = np.histogram(vels[:,50:100])
        x = x[:-1]
        
        angles_dist = np.zeros((len(self.inferenceData.attens),len(y)))
        # angles_mean_correct=np.zeros((len(self.inferenceData.attens)))
        # angles_std_correct=np.zeros((len(self.inferenceData.attens)))

        for ia, _ in enumerate(self.inferenceData.attens):
            angles_dist[ia],_ = np.histogram(vels[ia,:])
            tail = np.linspace(angles_dist[ia,0],angles_dist[ia,-1],len(angles_dist[ia]))
            angles_dist[ia] = abs(angles_dist[ia] - tail)
            # angles_mean_correct[ia] = mean(x,angles_dist[ia])
            # angles_std_correct[ia] = std(x,angles_dist[ia],angles_mean_correct[ia])

        tail = np.linspace(y[0],y[-1],len(y))
        y = y - tail
        mid_x = x[4:8]
        mid_y = y[4:8]
        
        # def mean(x, y):
        #     return sum(x*y) /sum(y)
        # def std(x,y,mean):
        #     return np.sqrt(sum(y * (x - mean)**2) / sum(y))
        # def Gauss(x, a, x0, sigma):
        #     return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
        # from scipy.optimize import curve_fit
        # # correction for weighted arithmetic mean
        # mean = mean(x,y)
        # # mean = 153
        # sigma = std(x,y,mean)
        # varience = 5
        # print mean, sigma
        # popt,pcov = curve_fit(Gauss, mid_x, mid_y, p0=[max(y), mean, sigma])
        # chi2 = sum((mid_y- Gauss(mid_x, *popt))**2 / varience**2 )
        # dof = len(mid_y)-3

        if plotAngles:
            plt.title(res_num)
            plt.plot(x, y, 'b:', label='data')
            # plt.plot(mid_x, Gauss(mid_x, *popt), 'r--', label='fit')
            # plt.plot(mid_x,mid_y -  Gauss(mid_x, *popt), label='residual')
            # plt.legend()
            plt.xlabel('Angle')
            plt.show()
            # for ia,_ in enumerate(self.inferenceData.attens):
            #     plt.plot(x, angles_dist[ia], 'b:', label='data')
            #     plt.show()

        # if chi2 < 100
        if np.all(angles_std_center > 80):
            print 'yes', angles_std_center
        if max(mid_y)<0:
                bad_res = True
        else: bad_res = False

        if showResData:
            fig, ax1 = plt.subplots()

            # ax1.plot(self.inferenceData.attens,angles[:,0],'b')
            # ax1.plot(self.inferenceData.attens,angles[:,1],'g')
            ax1.plot(self.inferenceData.attens,angles_running,'g')
            ax1.plot(self.inferenceData.attens,vels, 'bo',alpha=0.2)
            # ax1.plot(self.inferenceData.attens,angles_dist, 'bo',alpha=0.2)
            ax1.plot(self.inferenceData.attens,min_theta, 'k--')
            ax1.plot(self.inferenceData.attens,max_theta, 'k--')
            ax1.plot(self.inferenceData.attens,angles_mean)
            ax1.plot(self.inferenceData.attens,angles_std, 'r')
            ax1.plot(self.inferenceData.attens,angles_mean_center, 'b--')
            ax1.plot(self.inferenceData.attens,angles_std_center, 'r--')
            ax1.set_xlabel('Atten index')
            ax1.set_ylabel('Angles')
            ax1.set_title(res_num)
            for tl in ax1.get_yticklabels():
                tl.set_color('b')

            # ax2 = ax1.twinx()
            # ax2.plot(self.inferenceData.attens,ratio, 'r', label='ratio')
            # ax2.plot(self.inferenceData.attens,max_ratio_threshold, 'k--', label='thresh')
            # ax2.set_ylabel('Ratios')
            # for tl in ax2.get_yticklabels():
            #     tl.set_color('r')
            # plt.legend()

            # ax3 = ax1.twinx()
            # ax3.plot(self.inferenceData.attens,angles_nonsat, 'purple', label='angles_nonsat')
            # ax3.plot(self.inferenceData.attens,ratio_nonsat, 'crimson', label='ratio_nonsat')
            # fig.subplots_adjust(right=0.75)
            # ax3.spines['right'].set_position(('axes', 1.2))
           
 
            ax4 = ax1.twinx()
            
            from matplotlib import cm
            import matplotlib.colors
            # ax4.autoscale(False)
            ax4.imshow(angles_dist.T,interpolation='none',cmap=cm.coolwarm,alpha=0.5, origin='lower',extent=[self.inferenceData.attens[0],self.inferenceData.attens[-1],0,len(x)], aspect='auto')
            # ax4.legend()
            # plt.colorbar(cmap=cm.afmhot)
            # ax4.set_ylim(0,10)
            plt.show() 

        return [angles_nonsat,ratio_nonsat, ratio, running_ratio, bad_res, angles_mean_center, angles_std_center]

    def findAtten(self, inferenceFile, showFrames = False, plot_res= False, searchAllRes=True, res_nums=50):
        '''The trained machine learning class (mlClass) finds the optimum attenuation for each resonator using I,Q magnitudes and IQ velocity

        Inputs
        inferenceFile: widesweep data file to be used
        searchAllRes: if only a few resonator attenuations need to be identified set to False
        res_nums: if searchAllRes is False, the number of resonators the atteunation value will be estimated for
        plot_res: plot the values obtained the each resonator across all attenuations

        Outputs
        List of attenuation estimates for all resonators
        '''

        try:
            self.sess
        except AttributeError:
            print 'You have to train the model first'
            exit()

        if self.scalexWidth!= 1:
            self.xWidth=self.xWidth*self.scalexWidth #reset ready for get_PS_data

        total_res_nums = np.shape(self.inferenceData.freqs)[0]
        if searchAllRes:
            res_nums = total_res_nums

        resonators = range(res_nums)

        self.inferenceLabels = np.zeros((res_nums,len(self.inferenceData.attens),self.nClass))
        self.angles_proxys = np.zeros((res_nums))
        self.ang_means = np.zeros((res_nums,len(self.inferenceData.attens)))
        self.ang_stds = np.zeros((res_nums,len(self.inferenceData.attens)))
        self.max_ratios = np.zeros((res_nums,len(self.inferenceData.attens)))
        self.running_ratios = np.zeros((res_nums,len(self.inferenceData.attens)))
        self.ratio_guess = np.zeros((res_nums))

        print 'Using trained algorithm on images on each resonator'
        self.bad_res = []
        for i,rn in enumerate(resonators): 
            sys.stdout.write("\r%d of %i" % (i+1,res_nums) )
            sys.stdout.flush()

            angles_nonsat, ratio_nonsat, ratio, running_ratio, bad_res, ang_mean, ang_std = self.checkResAtten(res_num=rn)
            angles_proxy = np.argmax(ang_mean/ang_std)

            self.angles_proxys[rn] = angles_proxy
            self.ang_means[rn] = ang_mean
            self.ang_stds[rn] = ang_std
            self.max_ratios[rn] = ratio
            self.running_ratios[rn] = running_ratio

            # ratio_guess = np.where(running_ratio/max(running_ratio)<0.4)[0][0]
            ratio_guess = np.where(running_ratio<2.5)[0][0]
            
            if ratio_guess<len(self.inferenceData.attens)-1:
                while (running_ratio[ratio_guess] - running_ratio[ratio_guess+1] > 0.1) and (ratio_guess<len(self.inferenceData.attens)-2):
                    ratio_guess +=1

            if type(ratio_guess) ==np.int64:
                self.ratio_guess[rn] = ratio_guess

            if not bad_res:
                for ia in range(len(self.inferenceData.attens)-1):                
                    # first check the loop for saturation           
            
                    nonsaturated_loop = ratio_nonsat[ia] #and angles_nonsat[ia] 

                    if nonsaturated_loop:
                        # each image is formatted into a single element of a list so sess.run can receive a single values dictionary 
                        image = self.makeResImage(res_num = rn, iAtten= ia, showFrames=False)
                        inferenceImage=[]
                        inferenceImage.append(image)            # inferenceImage is just reformatted image
                        self.inferenceLabels[rn,ia,:] = self.sess.run(self.y, feed_dict={self.x: inferenceImage, self.keep_prob:1} )
                        del inferenceImage
                        del image
                       
                        self.inferenceLabels[rn,0,0]= self.inferenceLabels[rn,1,0] # since 0th term is skipped (and therefore 0)
                    else:
                        self.inferenceLabels[rn,ia,:] = [1,0]
            else:
                self.bad_res.append(rn)
                self.inferenceLabels[rn,:] = [1,0] # removed later anyhow
                # self.inferenceLabels = np.delete(self.inferenceLabels,rn,0)

        # self.inferenceLabels = np.delete(self.inferenceLabels,self.bad_res,0)

        print '\n'

        # res_nums = res_nums - len(self.bad_res)

        # self.max_2nd_vels = np.zeros((res_nums,len(self.inferenceData.attens)))
        # for r in range(res_nums):
        #     for iAtten in range(len(self.inferenceData.attens)):
        #         vindx = (-self.inferenceData.iq_vels[r,iAtten,:]).argsort()[:2]
        #         self.max_2nd_vels[r,iAtten] = self.inferenceData.iq_vels[r,iAtten,vindx[1]]

        self.inferenceData.opt_attens=numpy.zeros((res_nums))
        self.inferenceData.opt_freqs=numpy.zeros((res_nums))

        self.atten_guess=numpy.zeros((res_nums))

        bad=0
        self.inferenceLabels[:,-1,1] = 1
        for r in range(res_nums):

            if np.all(self.inferenceLabels[r,:,1]<0.5): self.atten_guess[r]=20 
            else:
                jump_guesses = []
                ideal_jgs = []
                for ia in range(int(self.ratio_guess[r]),len(self.inferenceData.attens)):
                # for ia in range(1,len(self.inferenceData.attens)):
                    jc1 = self.inferenceLabels[r,ia-1,1]<0.5    #jump condition 1
                    jc2 = self.inferenceLabels[r,ia,1]>0.5

                    if jc1 and jc2:
                        jump_guesses.append(ia)

                trail_vals = []
                if len(jump_guesses) >1:
                    for i, jg in enumerate(jump_guesses):
                        if i != len(jump_guesses)-1:
                            # print len(jump_guesses), i, jg, np.where(self.inferenceLabels[r,jg:jump_guesses[i+1],1]>0.5)
                            trail_vals.append(len(np.where(self.inferenceLabels[r,jg:jump_guesses[i+1],1]>0.5)[0]))
                        else:
                            # print len(jump_guesses), i, jg, np.where(self.inferenceLabels[r,jg:,1]>0.5)
                            trail_vals.append(len(np.where(self.inferenceLabels[r,jg:,1]>0.5)[0]))
                    # print argmax(trail_vals), jump_guesses[argmax(trail_vals)]
                    self.atten_guess[r] = jump_guesses[argmax(trail_vals)] # choose the jump with the most trailing non-sat prob >0.5 vals
                
                elif len(jump_guesses) ==1:
                    self.atten_guess[r] = jump_guesses[0]
                else:
                    self.atten_guess[r] = np.where(self.inferenceLabels[r,:,1]>0.5)[0][0]

            if self.atten_guess[r] < self.ratio_guess[r]:
                self.atten_guess[r] = self.ratio_guess[r]

            self.inferenceData.opt_attens[r] = self.inferenceData.attens[self.atten_guess[r]]
            self.inferenceData.opt_freqs[r] = self.inferenceData.freqs[r,self.get_peak_idx(r,self.atten_guess[r])]
    
            if plot_res:        
                fig, ax1 = plt.subplots()
                ax1.set_title(r)
                # plt.plot(self.inferenceLabels[self.test_res[wg],:,0], label='sat')
                ax1.plot(self.inferenceLabels[r,:,1], color='b',label='non-sat')
                # ax1.plot(self.max_2nd_vels[self.test_res[wg],:]/max(self.max_2nd_vels[self.test_res[wg],:]),color='g', label='2nd vel')
                ax1.plot(self.max_ratios[r,:]/max(self.max_ratios[r]),color='k', label='max vel')
                ax1.plot(self.running_ratios[r,:]/max(self.running_ratios[r,:]), color='g', label='running' )
                # ax1.plot(self.diff_rr[r,:]/max(self.diff_rr[r,:]), color='b',label='diff')
                ax1.axvline(self.atten_guess[r], color='k', linestyle='--', label='machine')
                ax1.axvline(self.ratio_guess[r], color='g', linestyle='--', label='ratio')
                ax1.axvline(self.angles_proxys[r], color='r', linestyle='--', label='angles')
                
                ax1.set_xlabel('Atten index')
                ax1.set_ylabel('Scores and 2nd vel')
                for tl in ax1.get_yticklabels():
                    tl.set_color('b')

                ax2 = ax1.twinx()
                ax2.plot(self.ang_means[r], color='r', label='ang means')
                ax2.plot(self.ang_stds[r], color='r', label='std means')
                for tl in ax2.get_yticklabels():
                    tl.set_color('r')
                # plt.legend()
                plt.show()     
            
    def evaluateModel(self, showFrames =False, retrain=False, plot_missed=False, res_nums=50):
        '''
        The loopTrain() function evaluates true performance by running findAtten on the training dataset. The predictions 
        on the correct attenuation value for is resonator can then compared with what the human chose. Then you can see the 
        models accuracy and if their are any trends in the resonators it's missing
        '''
        print 'running model on test input data'
        self.findAtten(inferenceFile=self.initialFile, searchAllRes=False, res_nums=res_nums)

        rawTrainData = PSFitMLData(h5File = self.initialFile, useResID=False)
        rawTrainData.loadTrainData() # run again to get 

        rawTrainData.good_res=rawTrainData.good_res[:np.where(rawTrainData.good_res<=res_nums)[0][-1]]
        
        bad_good_res = [] #res that are bad in good_res
        for item in self.bad_res:
            try:
                bad_good_res.append(list(rawTrainData.good_res).index(item) )
            except: pass
        rawTrainData.opt_iAttens =np.delete(rawTrainData.opt_iAttens, bad_good_res)

        # rawTrainData.good_res = np.setdiff1d(rawTrainData.good_res,self.bad_res) # remove bad_res from good_res
        rawTrainData.good_res = np.delete(rawTrainData.good_res, bad_good_res)

        bad_res_attens = np.delete(self.atten_guess, rawTrainData.good_res)
        good_res_atten_guess = self.atten_guess[rawTrainData.good_res]
        good_res_low_stds = self.angles_proxys[rawTrainData.good_res]
        good_res_ratio_guess = self.ratio_guess[rawTrainData.good_res]

        correct_guesses = []
        wrong_guesses=[]
        within_5=np.zeros((len(good_res_atten_guess)))
        within_3=np.zeros((len(good_res_atten_guess)))
        within_1=np.zeros((len(good_res_atten_guess)))
        within_0=np.zeros((len(good_res_atten_guess)))
        for ig, ag in enumerate(good_res_atten_guess):
            if abs(good_res_atten_guess[ig]-rawTrainData.opt_iAttens[ig]) <=5: within_5[ig] = 1
            if abs(good_res_atten_guess[ig]-rawTrainData.opt_iAttens[ig]) <=3: within_3[ig] = 1
            if abs(good_res_atten_guess[ig]-rawTrainData.opt_iAttens[ig]) <=1: within_1[ig] = 1
            if abs(good_res_atten_guess[ig]-rawTrainData.opt_iAttens[ig]) ==0: 
                within_0[ig] = 1
                correct_guesses.append(ig)
            if abs(good_res_atten_guess[ig]-rawTrainData.opt_iAttens[ig]) >=1: 
                wrong_guesses.append(ig)
                # print good_res_atten_guess[ig], rawTrainData.opt_iAttens[ig], ig
      
        print 'within 5', sum(within_5)/len(good_res_atten_guess)
        print 'within 3', sum(within_3)/len(good_res_atten_guess)
        print 'within 1', sum(within_1)/len(good_res_atten_guess)
        print 'within 0', sum(within_0)/len(good_res_atten_guess)

        within_5=np.zeros((len(good_res_low_stds)))
        within_3=np.zeros((len(good_res_low_stds)))
        within_1=np.zeros((len(good_res_low_stds)))
        within_0=np.zeros((len(good_res_low_stds)))

        for ig, ag in enumerate(good_res_atten_guess):
            if abs(good_res_low_stds[ig]-rawTrainData.opt_iAttens[ig]) <=5: within_5[ig] = 1
            if abs(good_res_low_stds[ig]-rawTrainData.opt_iAttens[ig]) <=3: within_3[ig] = 1
            if abs(good_res_low_stds[ig]-rawTrainData.opt_iAttens[ig]) <=1: within_1[ig] = 1
            if abs(good_res_low_stds[ig]-rawTrainData.opt_iAttens[ig]) ==0: within_0[ig] = 1
      
        print '\nwithin 5', sum(within_5)/len(good_res_low_stds)
        print 'within 3', sum(within_3)/len(good_res_low_stds)
        print 'within 1', sum(within_1)/len(good_res_low_stds)
        print 'within 0', sum(within_0)/len(good_res_low_stds)
        
        within_5=np.zeros((len(good_res_ratio_guess)))
        within_3=np.zeros((len(good_res_ratio_guess)))
        within_1=np.zeros((len(good_res_ratio_guess)))
        within_0=np.zeros((len(good_res_ratio_guess)))

        for ig, ag in enumerate(good_res_ratio_guess):
            if abs(good_res_ratio_guess[ig]-rawTrainData.opt_iAttens[ig]) <=5: within_5[ig] = 1
            if abs(good_res_ratio_guess[ig]-rawTrainData.opt_iAttens[ig]) <=3: within_3[ig] = 1
            if abs(good_res_ratio_guess[ig]-rawTrainData.opt_iAttens[ig]) <=1: within_1[ig] = 1
            if abs(good_res_ratio_guess[ig]-rawTrainData.opt_iAttens[ig]) ==0: within_0[ig] = 1
      
        print '\nwithin 5', sum(within_5)/len(good_res_ratio_guess)
        print 'within 3', sum(within_3)/len(good_res_ratio_guess)
        print 'within 1', sum(within_1)/len(good_res_ratio_guess)
        print 'within 0', sum(within_0)/len(good_res_ratio_guess)

        plot_missed = False
        if plot_missed:
            for i, wg in enumerate(wrong_guesses):
                # print wg, rawTrainData.good_res[wg], self.atten_guess[rawTrainData.good_res[wg]], '\t', rawTrainData.opt_iAttens[wg], '\t', self.low_stds[rawTrainData.good_res[wg]] 
                fig, ax1 = plt.subplots()
                ax1.set_title(rawTrainData.good_res[wg])
                ax1.plot(self.inferenceLabels[rawTrainData.good_res[wg],:,1], color='b',label='non-sat')
                # ax1.plot(self.max_2nd_vels[rawTrainData.good_res[wg],:]/max(self.max_2nd_vels[rawTrainData.good_res[wg],:]),color='g', label='2nd vel')
                ax1.plot(self.max_ratios[rawTrainData.good_res[wg],:]/max(self.max_ratios[rawTrainData.good_res[wg]]),color='k', label='max vel')
                ax1.plot(self.running_ratios[rawTrainData.good_res[wg],:]/max(self.running_ratios[rawTrainData.good_res[wg],:]),color='g', label='running' )
                ax1.axvline(self.atten_guess[rawTrainData.good_res[wg]], color='b', linestyle='--', label='machine')
                ax1.axvline(self.ratio_guess[rawTrainData.good_res[wg]], color='g', linestyle='--', label='ratio')
                ax1.axvline(self.angles_proxys[rawTrainData.good_res[wg]], color='r', linestyle='--', label='angles')
                ax1.axvline(rawTrainData.opt_iAttens[wg], color='k', linestyle='--', label='human')
                ax1.set_xlabel('Atten index')
                ax1.set_ylabel('Scores and 2nd vel')
                for tl in ax1.get_yticklabels():
                    tl.set_color('b')
                plt.legend()

                ax2 = ax1.twinx()
                ax2.plot(self.ang_means[rawTrainData.good_res[wg]], color='r', label='ang means')
                ax2.plot(self.ang_stds[rawTrainData.good_res[wg]], color='r', label='std means')
                for tl in ax2.get_yticklabels():
                    tl.set_color('r')

                plt.show()   

        cs_5 = np.cumsum(within_5/len(good_res_atten_guess))
        cs_3 = np.cumsum(within_3/len(good_res_atten_guess))
        cs_1 = np.cumsum(within_1/len(good_res_atten_guess))
        cs_0 = np.cumsum(within_0/len(good_res_atten_guess))

        rawTrainData.opt_iAttens = rawTrainData.opt_iAttens[:len(good_res_atten_guess)]
        guesses_map = np.zeros((np.shape(rawTrainData.attens)[1],np.shape(rawTrainData.attens)[1]))
        for ia,ao in enumerate(rawTrainData.opt_iAttens):   
            ag = good_res_atten_guess[ia]
            guesses_map[ag,ao] += 1

        from matplotlib import cm
        import matplotlib.colors
        plt.imshow(guesses_map,interpolation='none',cmap=cm.coolwarm)#,norm = matplotlib.colors.LogNorm())
        plt.xlabel('actual')
        plt.ylabel('estimate')

        plt.colorbar(cmap=cm.afmhot)
        plt.show()

        plt.plot(np.sum(guesses_map, axis=0), label='actual')
        plt.plot(np.sum(guesses_map, axis=1), label='estimate')
        plt.legend(loc="upper left")
        plt.show()

        showFrames=True
        if showFrames:
            # plt.plot(np.arange(len(good_res_atten_guess))/float(len(good_res_atten_guess)), 'r--', label='max')
            # plt.fill_between(range(len(cs_0)), cs_5, alpha=0.15, label='within 5')
            # plt.fill_between(range(len(cs_0)), cs_3, alpha=0.15,label='within 3')
            # plt.fill_between(range(len(cs_0)), cs_1, alpha=0.15,label='within 1')
            # plt.fill_between(range(len(cs_0)), cs_0, alpha=0.15, facecolor='blue', label='within 0')
            
            plt.plot(np.arange(len(good_res_atten_guess))/float(len(good_res_atten_guess))-cs_5, label='within 5')
            plt.plot(np.arange(len(good_res_atten_guess))/float(len(good_res_atten_guess))-cs_3, label='within 3')
            plt.plot(np.arange(len(good_res_atten_guess))/float(len(good_res_atten_guess))-cs_1, label='within 1')
            plt.plot(np.arange(len(good_res_atten_guess))/float(len(good_res_atten_guess))-cs_0, label='within 0')
            plt.legend(loc="upper left")
            plt.show()

def next_batch(trainImages, trainLabels, batch_size):
    '''selects a random batch of batch_size from trainImages and trainLabels'''
    perm = random.sample(range(len(trainImages)), batch_size)
    trainImages = np.array(trainImages)[perm,:]
    trainLabels = np.array(trainLabels)[perm,:]
    return trainImages, trainLabels

def loadPkl(filename):
    '''load the train and test data to train and test mlClass

    pkl file hirerachy is as follows:
        -The file is split in two, one side for train data and one side for test data -These halfs are further divdided into image data and labels
        -makeTrainData creates image data of size: xWidth * xWidth * res_nums and the label data of size: res_nums
        -each time makeTrainData is run a new image cube and label array is created and appended to the old data

    so the final size of the file is (xWidth * xWidth * res_nums * "no of train runs") + (res_nums * "no of train runs") + [the equivalent test data structure]

    A more simple way would be to separate the train and test data after they were read but this did not occur to the 
    me before most of the code was written

    Input
    pkl filename to be read.

    Outputs
    image cube and label array
    '''
    file =[]
    with open(filename, 'rb') as f:
        while 1:
            try:
                file.append(pickle.load(f))
            except EOFError:
                break
    
    trainImages = file[0][0]
    trainLabels = file[0][1]
    testImages = file[1][0]
    testLabels = file[1][1]

    print np.shape(file)[0]/2 -1
    if np.shape(file)[0]/2 > 1:
        for i in range(1, np.shape(file)[0]/2):
            trainImages = np.append(trainImages, file[2*i][0], axis=0)
            trainLabels = np.append(trainLabels, file[2*i][1], axis=0)
            testImages = np.append(testImages, file[2*i+1][0], axis=0)
            testLabels = np.append(testLabels, file[2*i+1][1], axis=0)

    print np.shape(trainLabels)

    print "loaded dataset from ", filename
    return trainImages, trainLabels, testImages, testLabels

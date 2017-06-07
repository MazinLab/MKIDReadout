''' 
90'%' accuracy sometimes and 50'%' exact attenuation choice

4 hidden layers. 2 pools
3 classes (like the original)

To do:
Could try just overpower and normal seeing as that's all you care about and then use the 2nd max to find optimal
Have it plot missed resonators to dientify why it's missing them
Read up on CNN if you have to
Make it do loop test automatically without having to do 2012 2012 in command line

Author Rupert Dodkins

A script to automate the identification of resonator attenuations normally performed by PSFit.py. This is accomplished 
using Google's Tensor Flow machine learning package which implements a pattern recognition algorithm on the IQ velocity
spectrum. The code implements a 2D image classification algorithm similar to the MNIST test. This code creates a 2D 
image from a 1D variable by populating a matrix of zeros with ones at the y location of each datapoint

Usage:  python PSFit_ml.py 20160712/ps_r115_FL1_1_20160712-225809.h5 20160720/ps_r118_FL1_b_pos_20160720-231653.h5

Inputs:
20160712/ps_r115_FL1_1.txt:                    list of resonator frequencies and correct attenuations
20160712/ps_r115_FL1_1_20160712-225809.h5:     corresponding powersweep file
20160720/ps_r118_FL1_b_pos_20160720-231653.h5: powersweep file the user wishes to infer attenuations for

Intermediaries:
SDR/Setup/ps_peaks_train_w<x>_s<y>.pkl:        images and corresponding labels used to train the algorithm

Outputs: 
20160712/ps_r115_FL1_1.pkl:                        frequencies, IQ velocities, Is, Qs, attenuations formatted for quick use
20160720/ps_r118_FL1_b_pos_20160720-231653-ml.txt: to be used with PSFit.py (temporary)
20160720/ps_r118_FL1_b_pos.txt:                    final list of frequencies and attenuations

How it works:
For each resonator and attenuation the script first assesses if the IQ loop appears saturated. If the unstaurated IQ 
velocity spectrum for that attenuation is compared with the pattern recognition machine. A list of attenuations for each 
resonator, where the loop is not saturated and the IQ velocity peak looks the correct shape, and the attenuation value 
is chosen which has the highest 2nd largest IQ velocity. This identifier was chosen because the optimum attenuation 
value has a high max IQ velocity and a low ratio of max IQ velocity to 2nd max IQ velocity which is equivalent to 
choosing the highest 2nd max IQ velocity.

This list of attenuation values and frequencies are either fed PSFit.py to checked manually or dumped to 
ps_r118_FL1_b_pos.txt

The machine learning algorithm requires a series of images to train and test the algorithm with. If they exist the image 
data will be loaded from a train pkl file

Alternatively, if the user does not have a train pkl file but does have a powersweep file and corresponding list of 
resonator attenuations this should be used as the initial file and training data will be made. The 3 classes are an 
overpowered peak (saturated), peak with the correct amount of power, or an underpowered peak. 

These new image data will be saved as pkl files (or appened to existing pkl files) and reloaded

The machine is then trained and its ability to predict the type of image is validated

The weights used to make predictions for each class can be displayed using the plotWeights function

to do:
change training txt file freq comparison function so its able to match all frequencies

'''
import os,sys,inspect
from PSFit import *
from iqsweep import *
import numpy as np
import sys, os
import matplotlib.pyplot as plt
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
        self.xWidth = 99#np.shape(res1_freqs[1])
        self.scalexWidth = 1
        self.oAttDist = -1 # rule of thumb attenuation steps to reach the overpowered peak
        #self.uAttDist = +2 # rule of thumb attenuation steps to reach the underpowed peak
        self.nAttens = 31
        self.nClass = self.nAttens

        self.initialFile = initialFile
        self.baseFile = ('.').join(initialFile.split('.')[:-1])
        self.PSFile = self.baseFile[:-16] + '.txt'#os.environ['MKID_DATA_DIR']+'20160712/ps_FL1_1.txt' # power sweep fit, .txt 
        self.mldir = './machine_learning_metadata/' 

        self.trainFile = 'ps_peaks_train_faceless_hf_img_cube_normed.pkl'
        print self.trainFile
        self.trainFrac = 0.9
        self.testFrac=1 - self.trainFrac
        self.attenWinBelow = 2 
        self.attenWinAbove = 2 
        
        if not inferenceFile is None:
            print 'Inference File:', inferenceFile
            self.inferenceData = PSFitMLData(h5File = inferenceFile, useAllAttens = False, useResID=True)

    def makeResImage(self, res_num, angle=0, center_loop=False,  phase_normalise=False, showFrames=False, test_if_noisy=False, dataObj=None):
        '''Creates a table with 2 rows, I and Q for makeTrainData(mag_data=True)

        inputs 
        res_num: index of resonator in question
        iAtten: index of attenuation in question
        self.scalexWidth: typical values: 1/2, 1/4, 1/8
                          uses interpolation to put data from an xWidth x xWidth grid to a 
                          (xWidth/scalexWidth) x (xWidth/scalexWidth) grid. This allows the 
                          user to probe the spectrum using a smaller window while utilizing 
                          the higher resolution training data
        angle: angle of rotation about the origin (radians)
        showFrames: pops up a window of the frame plotted using matplotlib.plot
        '''     
        if dataObj is None:
            if self.inferenceData is None:
                raise ValueError('Initialize dataObj first!')
            dataObj = self.inferenceData
        
        xWidth= self.xWidth 
        scalexWidth = self.scalexWidth

        #xCenter = self.get_peak_idx(res_num,iAtten,dataObj)
        start = 0
        end = self.xWidth

        # plt.plot(self.Is[res_num,iAtten], self.Qs[res_num,iAtten])
        # plt.show()
        # for spectra where the peak is close enough to the edge that some points falls across the bounadry, pad zeros

        
        iq_vels = dataObj.iq_vels[res_num, :, start:end]
        Is = dataObj.Is[res_num,:,start:end]
        Qs = dataObj.Qs[res_num,:,start:end]
        
        if center_loop:
            Is = np.transpose(np.transpose(Is) - np.mean(Is,1))
            print 'Is shape', np.shape(Is)
            print 'mean shape', np.shape(np.mean(Qs,1))
            Qs = np.transpose(np.transpose(Qs) - np.mean(Qs,1))
        #iq_vels = np.round(iq_vels * xWidth / max(dataObj.iq_vels[res_num, iAtten, :]) )
        iq_vels = np.transpose(np.transpose(iq_vels) / np.amax(iq_vels, axis=1))


                # interpolate iq_vels onto a finer grid


        # if test_if_noisy:
        #     peak_iqv = mean(iq_vels[int(xWidth/4): int(3*xWidth/4)])
        #     nonpeak_indicies=np.delete(np.arange(xWidth),np.arange(int(xWidth/4),int(3*xWidth/4)))
        #     nonpeak_iqv = iq_vels[nonpeak_indicies]
        #     nonpeak_iqv = mean(nonpeak_iqv[np.where(nonpeak_iqv!=0)]) # since it spans a larger area
        #     noise_condition = 1.5#0.7 

        #     if (peak_iqv/nonpeak_iqv < noise_condition):
        #         return None 

        res_mag = np.sqrt(np.amax(Is**2 + Qs**2, axis=1))
        Is = np.transpose(np.transpose(Is) / res_mag)
        Qs = np.transpose(np.transpose(Qs) / res_mag)

        # Is = Is /np.amax(dataObj.iq_vels[res_num, :, :])
        # Qs = Qs /np.amax(dataObj.iq_vels[res_num, :, :])

        # Is = Is /np.amax(dataObj.Is[res_num, :, :])
        # Qs = Qs /np.amax(dataObj.Qs[res_num, :, :])

        # print Is[::5]
        # print Qs[::5]

        if phase_normalise: #need to fix for imgcube
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

        image = np.zeros((np.shape(Is)[0],1+self.attenWinAbove+self.attenWinBelow,np.shape(Is)[1],3))
        singleFrameImage = np.zeros((np.shape(Is)[0],np.shape(Is)[1],3))
        singleFrameImage[:,:,0] = Is
        singleFrameImage[:,:,1] = Qs
        singleFrameImage[:,:,2] = iq_vels

        #for offs in range(self.attenWinBelow):
        #    offsImage = np.roll(singleFrameImage, offs, axis=0)
        #    offsImage[0:offs] = singleFrameImage[0]
        #    image[:,-offs,:,:] = offsImage

        #for offs in range(1,self.attenWinAbove):
        #    offsImage = np.roll(singleFrameImage, -offs, axis=0)
        #    offsImage[-offs:] = singleFrameImage[-1]
        #    image[:,offs,:,:] = offsImage
        
        #image = image.flatten()
        # image = np.append(Is,Qs,axis=0)

        #print np.shape(image)

        return singleFrameImage

    def get_peak_idx(self,res_num,iAtten,dataObj=None,smooth=False):
        if dataObj is None:
            if self.inferenceData is None:
                raise ValueError('Initialize dataObj first!')
            dataObj = self.inferenceData
            iq_vels = dataObj.iq_vels[res_num,iAtten,:]
            if smooth:
                iq_vels = np.correlate(iq_vels, np.ones(15), mode='same')
        return argmax(iq_vels)

    def makeTrainData(self, trimAttens=False):                
        '''creates 1d arrays using makeWindowImage of each class with associated labels and saves to pkl file

        0: saturated peak, too much power
        1: goldilocks, not too narrow or short
        2: underpowered peak, too little power
        
        or if plotting IQ mags (mag_data==True)

        outputs
        train file.pkl. contains...
            trainImages: table of size- xWidth * xCenters*trainFrac
            trainLabels: 1d array of size- xCenters*trainFrac
            testImages: table of size- xWidth * xCenters*testFrac
            testLabels: 1d array of size- xCenters*testFrac

        creates the training data to be passed to function mlClass. For each resonator and attenuation a 2 x num_freqs table is created 
        with associated labels and saves to pkl file

        -3              :
        -2:             :
        -1: slightly saturated too much power  
        0:  goldilocks, not too narrow or short
        1:  underpowered peak, too little power
        2:              :
        3:              :

        '''
        
        rawTrainData = PSFitMLData(h5File = self.initialFile, useResID=True)
        rawTrainData.loadTrainData()
        
        a=0 # index to remove values from all_freqs
        b = 0  # backtrack on g when good freqs can't be used
            # g index for the good freqs
        bad_opt_res = []
        
        good_res = np.arange(len(rawTrainData.resIDs))

        iAttens = np.zeros((len(good_res),self.nClass))
        print 'iAttens shape', np.shape(iAttens)
        print 'opt_iAttens shape', np.shape(rawTrainData.opt_iAttens)
        # for i in range(self.nClass-1):
        #     iAttens[:,i] = np.delete(self.opt_iAttens,bad_opt_res) + attDist[i]        
        # print len(self.attens)

        # iAttens[:,2] = np.ones((len(good_res)))*len(self.attens)-1#self.opt_iAttens[:self.res_nums] + self.uAttDist

        #self.opt_iAttens = np.delete(self.opt_iAttens,bad_opt_res)



        if trimAttens:
            good_res = np.delete(good_res, np.where(rawTrainData.opt_iAttens < self.attenWinBelow)[0])

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


        for rn in train_ind:#range(int(self.trainFrac*rawTrainData.res_nums)):
            # for t in range(num_rotations):
            #     image = self.makeResImage(res_num = rn, iAtten= iAttens[rn,c], angle=angle[t],showFrames=False, 
            #                                 test_if_noisy=test_if_noisy, xCenter=self.res_indicies[rn,c])
            image = self.makeResImage(res_num = rn, phase_normalise=False ,showFrames=False, dataObj=rawTrainData) 
            if image!=None:
                trainImages.append(image)
                oneHot = np.zeros(self.nAttens)
                oneHot[rawTrainData.opt_iAttens[rn]] = 1
                trainLabels.append(oneHot)

        print rawTrainData.res_nums


        for rn in test_ind:#range(int(self.trainFrac*rawTrainData.res_nums), int(self.trainFrac*rawTrainData.res_nums + self.testFrac*rawTrainData.res_nums)):
            image = self.makeResImage(res_num = rn, phase_normalise=False, dataObj=rawTrainData)
            if image!=None:
                testImages.append(image)
                oneHot = np.zeros(self.nAttens)
                oneHot[rawTrainData.opt_iAttens[rn]] = 1
                testLabels.append(oneHot)
        


        append = None
        if os.path.isfile(self.trainFile): 
            append = raw_input('Do you want to append this training data to previous data [y/n]')
        if (append  == 'n'):
            self.trainFile = self.trainFile.split('-')[0]+time.strftime("-%Y-%m-%d-%H-%M-%S")
        if (append  == 'y') or (os.path.isfile(self.trainFile)== False):
            print 'saving %s to %s' % (self.trainFile, os.path.dirname(os.path.abspath(self.trainFile)) )
            with open(self.trainFile, 'ab') as tf:
                pickle.dump([trainImages, trainLabels], tf)
                pickle.dump([testImages, testLabels], tf)

    def mlClass(self):       
        '''Code adapted from the tensor flow MNIST tutorial 1.
        
        Using training images and labels the machine learning class (mlClass) "learns" how to classify IQ velocity peaks. 
        Using similar data the ability of mlClass to classify peaks is tested

        The training and test matricies are loaded from file (those made earlier if chosen to not be appended to file 
        will not be used)
        '''
        print self.trainFile
        if not os.path.isfile(self.trainFile):
            print 'Could not find train file. Making new training images from initialFile'
            self.makeTrainData()

        trainImages, trainLabels, testImages, testLabels = loadPkl(self.trainFile)
        
        print np.sum(trainLabels,axis=0), np.sum(testLabels,axis=0)   

        print np.sum(trainLabels,axis=0), np.sum(testLabels,axis=0)
        print 'Number of training images:', np.shape(trainImages), ' Number of test images:', np.shape(testImages)

        print np.shape(trainLabels)
        print np.sum(trainLabels,axis=0)

        # print len(trainImages)
        # for i in range(len(trainImages)):
        #     if i % 50 ==0:
        #         print trainLabels[i]
        #         print np.shape(trainImages[i][:])
        #         plt.plot(trainImages[i][:40])
        #         plt.plot(trainImages[i][40:])
        #         plt.show()
      
        #if self.scalexWidth != 1:
        #    self.xWidth = int(self.xWidth/self.scalexWidth)
        #if np.shape(trainImages)[1]/3!=self.xWidth:
        #    print 'Please make new training images of the correct size'
        #    exit()
          
        #self.nClass = np.shape(trainLabels)[1]

        #self.x = tf.placeholder(tf.float32, [None, self.xWidth]) # correspond to the images
        
        self.x = tf.placeholder(tf.float32, [None, self.nAttens, self.xWidth, 3])
        attenWin = 1 + self.attenWinBelow + self.attenWinAbove
        #print type(self.x[0][0])
        #print self.x[0][0]
        #print self.xWidth
        #exit()

        #x_image = tf.reshape(self.x, [-1,1,self.xWidth,1])
        #x_image = tf.reshape(self.x, [-1,3,self.xWidth,1])
        x_image = tf.reshape(self.x, [-1, self.nAttens, self.xWidth, 3])
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
        
        self.keep_prob = tf.placeholder(tf.float32)

        num_filt1 = 3
        n_pool1 = 3
        self.num_filt1 = num_filt1
        W_conv1 = weight_variable([attenWin, 5, 3, num_filt1])
        b_conv1 = bias_variable([num_filt1])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_nx1(h_conv1,n_pool1)
        h_pool1_dropout = tf.nn.dropout(h_pool1, self.keep_prob)
        xWidth1 = int(math.ceil(self.xWidth/float(n_pool1)))
        cWidth1 = num_filt1
        
        num_filt2 = 3 
        n_pool2 = 1 
        self.num_filt2 = num_filt2
        W_conv2 = weight_variable([1, 5, cWidth1, num_filt2])
        b_conv2 = bias_variable([num_filt2])
        h_conv2 = tf.nn.relu(conv2d(h_pool1_dropout, W_conv2) + b_conv2)
        with tf.control_dependencies([tf.assert_equal(tf.shape(h_pool1),(numImg,self.nAttens,xWidth1,cWidth1),message='hpool1 assertion')]):
            h_pool2 = max_pool_nx1(h_conv2, n_pool2)
            h_pool2_dropout = tf.nn.dropout(h_pool2, self.keep_prob)
            xWidth2 = int(math.ceil(xWidth1/float(n_pool2)))
            cWidth2 = num_filt2

        num_filt3 = 2
        n_pool3 = 1 
        self.num_filt3 = num_filt3
        W_conv3 = weight_variable([1, 4, cWidth2, num_filt3])
        b_conv3 = bias_variable([num_filt3])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        with tf.control_dependencies([tf.assert_equal(tf.shape(h_pool2),(numImg,self.nAttens,xWidth2,cWidth2),message='hpool2 assertion')]):
            h_pool3 = max_pool_nx1(h_conv3, n_pool3)
            h_pool3_dropout = tf.nn.dropout(h_pool3, self.keep_prob)
            xWidth3 = int(math.ceil(xWidth2/float(n_pool3)))
            cWidth3 = num_filt3

        # num_filt4 = 2
        # n_pool4 = 2
        # self.num_filt4 = num_filt4
        # W_conv4 = weight_variable([1, 3, cWidth3, num_filt4])
        # b_conv4 = bias_variable([num_filt4])
        # h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
        # with tf.control_dependencies([tf.assert_equal(tf.shape(h_pool3),(numImg,self.nAttens,xWidth3,cWidth3),message='hpool3 assertion')]):
        #     h_pool4 = max_pool_nx1(h_conv4, n_pool4)
        #     h_pool4_dropout = tf.nn.dropout(h_pool4, self.keep_prob)
        #     xWidth4 = int(math.ceil(xWidth3/float(n_pool4)))
        #     cWidth4 = num_filt4
        
        h_pool3_flat = tf.reshape(h_pool3_dropout,[-1,self.nAttens,cWidth3*xWidth3,1])        
        W_final = weight_variable([1, cWidth3*xWidth3, 1, 1])
        b_final = bias_variable([1])     
        
        with tf.control_dependencies([tf.assert_equal(tf.shape(h_pool3),(numImg,self.nAttens,xWidth3,cWidth3))]):
            h_conv_final = tf.nn.conv2d(h_pool3_flat, W_final, strides=[1, 1, 1, 1], padding='VALID') + b_final
        
        with tf.control_dependencies([tf.assert_equal(tf.shape(h_conv_final),(numImg,self.nAttens,1,1))]):
            h_conv_final = tf.reshape(h_conv_final, [-1,self.nAttens])
        
        self.y=tf.nn.softmax(h_conv_final) #h_fc1_drop   

        y_ = tf.placeholder(tf.float32, [None, self.nAttens]) # true class lables identified by user 
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(self.y+ 1e-10), reduction_indices=[1])) # find optimum solution by minimizing error
        #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y,y_))
        yWeighted = tf.mul(self.y, tf.to_float(tf.range(tf.shape(self.y)[1])))
        yInd = tf.reduce_sum(yWeighted, reduction_indices=1)
        y_Ind = tf.to_float(tf.argmax(y_, 1))
        
        squared_loss = tf.reduce_mean(tf.to_float(tf.square(yInd-y_Ind)))

        train_step = tf.train.AdamOptimizer(10**-3).minimize(cross_entropy) # the best result is when the wrongness is minimal

        init = tf.initialize_all_variables()

        saver = tf.train.Saver()
        
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1)) #which ones did it get right?
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        modelName = ('.').join(self.trainFile.split('.')[:-1]) + '_3layer_img_cube.ckpt'
        print modelName
        if os.path.isfile("%s%s" % (self.mldir,modelName)):
            #with tf.Session() as sess:
            self.sess = tf.Session()
            self.sess.run(init)           #do I need this? 
            # Restore variables from disk.
            saver.restore(self.sess, "%s%s" % (self.mldir,modelName) )
        else:
            self.sess = tf.Session()
            self.sess.run(init) # need to do this everytime you want to access a tf variable (for example the true class labels calculation or plotweights)

            # for i in range(len(trainImages)):
            #     if i % 30 ==0:
            #         print trainLabels[i]
            #         print np.shape(trainImages[i][:])
            #         plt.plot(self.sess.run(self.x)[i][0],feed_dict={self.x: trainImages})
            #         plt.plot(self.sess.run(self.x)[i][1],feed_dict={self.x: trainImages})
            #         plt.show()
            
            start_time = time.time()
            trainReps = 10000
            batches = 50
            if np.shape(trainLabels)[0]< batches:
                batches = np.shape(trainLabels)[0]/2
            


            # print self.sess.run(tf.shape(W_conv1), feed_dict={self.x: testImages, y_: testLabels})    
            # print self.sess.run(tf.shape(h_conv1), feed_dict={self.x: testImages})    
            # #print self.sess.run(tf.shape(h_pool1), feed_dict={self.x: testImages})
            # print '\n'
            # print self.sess.run(tf.shape(W_conv2))    
            # print self.sess.run(tf.shape(h_conv2), feed_dict={self.x: testImages})    
            # print self.sess.run(tf.shape(h_pool2), feed_dict={self.x: testImages})
            # print '\n'
            # print self.sess.run(tf.shape(W_conv3))    
            # print self.sess.run(tf.shape(h_conv3), feed_dict={self.x: testImages})    
            # #print self.sess.run(tf.shape(h_pool3), feed_dict={self.x: testImages})
            # print '\n'
            # print self.sess.run(tf.shape(W_conv4))    
            # print self.sess.run(tf.shape(h_conv4), feed_dict={self.x: testImages})    
            # print self.sess.run(tf.shape(h_pool4), feed_dict={self.x: testImages})
            # print '\n'
            # print self.sess.run(tf.shape(W_conv5))    
            # print self.sess.run(tf.shape(h_conv5), feed_dict={self.x: testImages})    
            # #print self.sess.run(tf.shape(h_pool3), feed_dict={self.x: testImages})
            # print '\n'
            # print self.sess.run(tf.shape(W_conv6))    
            # print self.sess.run(tf.shape(h_conv6), feed_dict={self.x: testImages})    
            # print self.sess.run(tf.shape(h_pool6), feed_dict={self.x: testImages})
            # print '\n'
            # print self.sess.run(tf.shape(W_fc1))    
            # print self.sess.run(tf.shape(h_pool6_flat), feed_dict={self.x: testImages})    
            # print self.sess.run(tf.shape(h_fc1), feed_dict={self.x: testImages})
            # print '\n'

            # print '\n'
            # print self.sess.run(tf.shape(self.W_fc2)) 



            ce_log = []
            acc_log=[]
            print 'Performing', trainReps, 'training repeats, using batches of', batches
            for i in range(trainReps):  #perform the training step using random batches of images and according labels
                batch_xs, batch_ys = next_batch(trainImages, trainLabels, batches) 
                #print 'batch_xs shape', np.shape(batch_xs) 
                #print np.shape(batch_xs), np.shape(batch_ys)
                sys.stdout.write('\rbatch: %i' % (i))
                sys.stdout.flush()

                if i % 100 == 0:
                    #print'Plotting final Weights'
                    #self.plotW_fc2(self.sess.run(self.W_fc3))
                    #print 'Plotting W_conv1'
                    #self.plotWeights(self.sess.run(W_conv1))
                    #print 'Plotting h_conv1 activations'
                    #self.plotActivations(self.sess.run(h_conv1, feed_dict={self.x: batch_xs}), 'h_conv1', i)
                    #print 'Plotting W_conv2'
                    #self.plotWeights(self.sess.run(W_conv2))
                    #print 'Plotting W_conv2 activations'
                    #self.plotActivations(self.sess.run(h_conv2, feed_dict={self.x: testImages}), 'h_conv2', i)
                    #print self.sess.run(W_conv1, feed_dict={self.x: testImages, y_: testLabels})
                    #print "max W vales: %g %g %g %g"%(self.sess.run(tf.reduce_max(tf.abs(W_conv1))),self.sess.run(tf.reduce_max(tf.abs(W_conv2))),self.sess.run(tf.reduce_max(tf.abs(W_fc1))),self.sess.run(tf.reduce_max(tf.abs(self.W_fc2))))
                    #print "max b vales: %g %g %g %g"%(self.sess.run(tf.reduce_max(tf.abs(b_conv1))),self.sess.run(tf.reduce_max(tf.abs(b_conv2))),self.sess.run(tf.reduce_max(tf.abs(b_fc1))),self.sess.run(tf.reduce_max(tf.abs(b_fc2))))
                    ce_log.append(self.sess.run(cross_entropy, feed_dict={self.x: batch_xs, y_: batch_ys, self.keep_prob: 1}))
                    # print batch_ys[10],#, feed_dict={y_: batch_ys}),
                    # print self.sess.run(self.y, feed_dict={self.x: batch_xs})[10]
                    acc_log.append(self.sess.run(accuracy, feed_dict={self.x: testImages, y_: testLabels, self.keep_prob: 1}) * 100)

                # if i % 1000 ==0:
                    # saver.save(self.sess, "/tmp/model.ckpt",global_step=i)#self.plotWeights(self.sess.run(W_fc2))
                    print ' ', self.sess.run(cross_entropy, feed_dict={self.x: batch_xs, y_: batch_ys, self.keep_prob: 1}),
                    # print batch_ys[10],#, feed_dict={y_: batch_ys}),
                    # print self.sess.run(self.y, feed_dict={self.x: batch_xs})[10]
                    print self.sess.run(accuracy, feed_dict={self.x: testImages, y_: testLabels, self.keep_prob: 1}) * 100
                self.sess.run(train_step, feed_dict={self.x: batch_xs, y_: batch_ys, self.keep_prob: 1}) #calculate train_step using feed_dict
        
            print "--- %s seconds ---" % (time.time() - start_time)
            #print ce_log, acc_log
            fig = plt.figure(frameon=False,figsize=(15.0, 5.0))
            fig.add_subplot(121)
            plt.plot(ce_log)
            fig.add_subplot(122)
            plt.plot(acc_log)
            plt.show()

            print "%s%s" % (self.mldir,modelName)
            save_path = saver.save(self.sess, "%s%s" % (self.mldir,modelName))
            print 'true class labels: ', self.sess.run(tf.argmax(y_,1), 
                                                       feed_dict={self.x: testImages, y_: testLabels, self.keep_prob: 1})
            print 'class estimates:   ', self.sess.run(tf.argmax(self.y,1), 
                                                       feed_dict={self.x: testImages, y_: testLabels, self.keep_prob: 1}) #1st 25 printed
        #print self.sess.run(self.y, feed_dict={self.x: testImages, y_: testLabels})[:100]  # print the scores for each class
            ys_true = self.sess.run(tf.argmax(y_,1), feed_dict={self.x: testImages, y_: testLabels, self.keep_prob: 1})
            ys_guess = self.sess.run(tf.argmax(self.y,1), feed_dict={self.x: testImages, y_: testLabels, self.keep_prob: 1})
            print np.sum(self.sess.run(self.y, feed_dict={self.x: testImages, y_: testLabels, self.keep_prob: 1}),axis=0)
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
            #acc_log.append(score)
            train_log = np.array([['score ', score], ['keep_prob', 1], ['num_filt1', self.num_filt1], ['num_filt2', self.num_filt2], ['num_filt3', self.num_filt3],
                ['num_filt4', 0], ['n_pool1', n_pool1], ['n_pool2', n_pool2], ['n_pool3', n_pool3], ['n_pool4', 0]])
            print train_log
            np.savetxt(modelName[:-4]+'.log', train_log, ('%10s', '%10s'))
            plt.hist(ys_guess[right])
            plt.title('correct guesses')
            plt.show()
            plt.hist(ys_guess)
            plt.title('all guesses')
            plt.show()
            plt.hist(ys_true)
            plt.title('correct attens')
            plt.show()

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
    #     weights = np.reshape(weights,(self.xWidth,self.xWidth,self.nClass))
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
    
    def checkLoopAtten(self, res_num, iAtten, showLoop=False, min_theta = 135, max_theta = 200, max_ratio_threshold = 1.5):
        '''function to check if the IQ loop at a certain attenuation is saturated. 3 checks are made.
        if the angle on either side of the sides connected to the longest edge is < min_theta or > max_theta
        the loop is saturated. Or if the ratio between the 1st and 2nd largest edge is > max_ratio_threshold.

        A True result means that the loop is unsaturated.

        Inputs:
        res_num: index of resonator in question
        iAtten: index of attenuation in question
        showLoop: pops up a window of the frame plotted using matplotlib.plot
        min/max_theta: limits outside of which the loop is considered saturated
        max_ratio_threshold: maximum largest/ 2nd largest IQ velocity allowed before loop is considered saturated

        Output:
        Boolean. True if unsaturated
        '''
        
        vindx = (-self.inferenceData.iq_vels[res_num,iAtten,:]).argsort()[:3]
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

        max_ratio = self.inferenceData.iq_vels[res_num,iAtten,vindx[0]]/ self.inferenceData.iq_vels[res_num,iAtten,vindx[1]]
        
        if showLoop:
            plt.plot(self.inferenceData.Is[res_num,iAtten,:],self.inferenceData.Qs[res_num,iAtten,:])
            plt.show()
        

        # return True
        # return bool(max_ratio < max_ratio_threshold)
        # if max_ratio < max_ratio_threshold == True:
        #     return True
        # if (max_theta >theta1 > min_theta) * (max_theta > theta2 > min_theta) == True:
        #     return 'noisy'
        # else:
        #     return False

        # return [(max_theta >theta1 > min_theta)*(max_theta > theta2 > min_theta) , max_ratio < max_ratio_threshold]
        return bool((max_theta >theta1 > min_theta) * 
                    (max_theta > theta2 > min_theta) * 
                    (max_ratio < max_ratio_threshold))
        # if res_num==6:
        #     print res_num, max_ratio, self.inferenceData.iq_vels[res_num,iAtten,vindx[0]], self.inferenceData.iq_vels[res_num,iAtten,vindx[1]]
        #     plt.plot(self.inferenceData.Is[res_num,iAtten,:],self.inferenceData.Qs[res_num,iAtten,:])
        #     plt.show()


    def findAtten(self, res_nums =20, searchAllRes=True, showFrames = True, usePSFit=True):
        '''The trained machine learning class (mlClass) finds the optimum attenuation for each resonator using peak shapes in IQ velocity

        Inputs
        inferenceFile: widesweep data file to be used
        searchAllRes: if only a few resonator attenuations need to be identified set to False
        res_nums: if searchAllRes is False, the number of resonators the atteunation value will be estimated for
        usePSFit: if true once all the resonator attenuations have been estimated these values are fed into PSFit which opens
                  the window where the user can manually check all the peaks have been found and make corrections if neccessary

        Outputs
        Goodfile: either immediately after the peaks have been located or through WideAna if useWideAna =True
        mlFile: temporary file read in to PSFit.py containing an attenuation estimate for each resonator
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
        span = range(res_nums)
        
        self.inferenceData.opt_attens=numpy.zeros((res_nums))
        self.inferenceData.opt_freqs=numpy.zeros((res_nums))

        print 'inferenceAttens', self.inferenceData.attens

        self.inferenceLabels = np.zeros((res_nums, self.nAttens))
        print 'Using trained algorithm on images on each resonator'
        skip = []
        for i,rn in enumerate(span): 
            sys.stdout.write("\r%d of %i" % (i+1,res_nums) )
            sys.stdout.flush()
            image = self.makeResImage(res_num = rn, phase_normalise=False,showFrames=False, dataObj=self.inferenceData)
            inferenceImage=[]
            inferenceImage.append(image)            # inferenceImage is just reformatted image
            self.inferenceLabels[rn,:] = self.sess.run(self.y, feed_dict={self.x: inferenceImage, self.keep_prob: 1})
            iAtt = np.argmax(self.inferenceLabels[rn,:])
            self.inferenceData.opt_attens[rn] = self.inferenceData.attens[iAtt]
            self.inferenceData.opt_freqs[rn] = self.inferenceData.freqs[rn,self.get_peak_idx(rn,iAtt,smooth=True)]
            del inferenceImage
            del image
                

    
def next_batch(trainImages, trainLabels, batch_size):
    '''selects a random batch of batch_size from trainImages and trainLabels'''
    perm = random.sample(range(len(trainImages)), batch_size)
    trainImages = np.array(trainImages)[perm]
    trainLabels = np.array(trainLabels)[perm]
    #print 'next_batch trImshape', np.shape(trainImages)
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

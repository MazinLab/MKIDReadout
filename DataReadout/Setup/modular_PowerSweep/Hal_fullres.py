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
# from PSFit import *
from iqsweep import *
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.INFO)
import pickle
import random
import time
import math
from scipy import interpolate
from PSFitMLData import *
import PSFitMLTools as mlt
from ml_params import mldir, trainFile, max_nClass, trainReps, batches
# from PSFitMLData_origPSFile import *
np.set_printoptions(threshold=np.inf)
from tensorflow.contrib.tensorboard.plugins import projector

#removes visible depreciation warnings from lib.iqsweep
import warnings
warnings.filterwarnings("ignore")

class mlClassification():
    def __init__(self, subdir=None):
        '''
        Implements the machine learning pattern recognition algorithm on IQ velocity data as well as other tests to 
        choose the optimum attenuation for each resonator
        '''

        # self.inferenceFile = inferenceFile
        # self.baseFile = ('.').join(inferenceFile.split('.')[:-1])
        # self.PSFile = self.baseFile[:-16] + '.txt'#os.environ['MKID_DATA_DIR']+'20160712/ps_FL1_1.txt' # power sweep fit, .txt 
        if subdir != None:
            self.mldir = mldir+subdir
        else:
            self.mldir = mldir
        self.trainFile = trainFile
        self.batches = batches
        self.trainReps = trainReps
       
    def train(self, **kwargs):       
        # accuracy_plot='post', plot_weights='', plot_activations=''
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
        assert os.path.isfile(self.mldir + self.trainFile)

        batches = kwargs.pop('batches')
        trainReps = kwargs.pop('trainReps')
        plot_confusion = kwargs.pop('plot_confusion', True)
        plot_missed = kwargs.pop('plot_missed', False)
        max_learning_rate = kwargs.pop('max_learning_rate')
        min_learning_rate = kwargs.pop('min_learning_rate')
        decay_speed = kwargs.pop('decay_speed')
        fully_connected = kwargs.pop('fully_connected', False)
        plot_activations = kwargs.pop('plot_activations', False)
        plot_weights = kwargs.pop('plot_weights', False)
        recursive = kwargs.pop('recursive', False)

        trainImages, trainLabels, testImages, testLabels = loadPkl(self.mldir+self.trainFile)
        trainImages = np.asarray(trainImages)
        trainLabels = np.asarray(trainLabels)
        # hist, bins = np.histogram(np.argmax(trainLabels, axis =1), range(max_nClass+1))
        # plt.plot(hist)
        # hist, bins = np.histogram(np.argmax(testLabels, axis =1), range(max_nClass+1))
        # plt.plot(hist)
        # plt.show()

        # max_iq = np.amax(trainImages[:,:,:,2], axis=2)
        # def PCA():
        #     LOG_DIR = '/tmp/emb_logs/'
        #     metadata = os.path.join(LOG_DIR, 'metadata.tsv')

        #     # mnist = input_data.read_data_sets('MNIST_data')

        #     #Variables
        #     images = tf.Variable(max_iq, name='images')

        #     with open(metadata, 'wb') as metadata_file:
        #         for row in trainLabels:
        #             row = np.argmax(row)
        #             metadata_file.write('%d\n' % row)

        #     with tf.Session() as sess:
        #         saver = tf.train.Saver([images])

        #         sess.run(images.initializer)
        #         saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))

        #         config = projector.ProjectorConfig()
        #         # One can add multiple embeddings.
        #         embedding = config.embeddings.add()
        #         embedding.tensor_name = images.name
        #         # Link this tensor to its metadata file (e.g. labels).
        #         embedding.metadata_path = metadata
        #         # Saves a config file that TensorBoard will read during startup.
        #         projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

        # PCA()
        # exit()
        self.xWidth = np.shape(trainImages)[2]
        self.nClass = np.shape(trainImages)[1]
        print self.xWidth, self.nClass
        # exit()

        print 'Number of training images:', np.shape(trainImages), ' Number of test images:', np.shape(testImages)
   
        # if self.scalexWidth != 1:
        #     self.xWidth = int(self.xWidth/self.scalexWidth)
        if np.shape(trainImages)[2]!=self.xWidth:
            print 'Please make new training images of the correct size'
            exit()
              
        # res_per_win = 4
        # for f in range(int(np.ceil(len(trainLabels)/res_per_win))+1):
        
        #     _, axarr = plt.subplots(2*res_per_win,self.nClass, figsize=(16.0, 8.1))
        #     for r in range(res_per_win):
        #         print argmax(trainLabels[f+r],0)
        #         for ia in range(self.nClass):
        #             # print f, r, missed[f+r]
        #             axarr[2*r,0].set_ylabel(trainLabels[f+r])
        #             # axarr[2*r,ia].axis('off')
        #             # axarr[(2*r)+1,ia].axis('off')
        #             if ia != argmax(trainLabels[r+f*res_per_win],0): axarr[2*r,ia].axis('off')
        #             if ia != argmax(trainLabels[r+f*res_per_win],0): axarr[(2*r)+1,ia].axis('off')
        #             # print np.shape(testImages[f+r,ia,:,2])
        #             axarr[2*r,ia].plot(trainImages[r+f*res_per_win][ia,:,2])
        #             axarr[(2*r)+1,ia].plot(trainImages[r+f*res_per_win][ia,:,0],trainImages[r+f*res_per_win][ia,:,1], '-o')

        #     plt.show()
        #     plt.close()

        self.x = tf.placeholder(tf.float32, [None, max_nClass, self.xWidth, 3])
        
        x_image = tf.reshape(self.x, [-1, self.nClass, self.xWidth, 3])
        is_test = tf.placeholder(tf.bool)
        lr = tf.placeholder(tf.float32)

        def weight_variable(shape):
            #initial = tf.Variable(tf.zeros(shape))
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        def max_pool_nx1(x,n):
            return tf.nn.max_pool(x, ksize=[1, 1, n, 1], strides=[1, 1, n, 1], padding='SAME')
        
        def batchnorm_layer(Ylogits, is_test):
            scale = tf.Variable(tf.ones([Ylogits.get_shape()[-1]]), trainable=True)
            beta = tf.Variable(tf.zeros([Ylogits.get_shape()[-1]]), trainable=True)
            ema = tf.train.ExponentialMovingAverage(0.99)
            # if convolutional:
            mean, var= tf.nn.moments(Ylogits, [0,1,2])
            # else:
                # mean, variance= tf.nn.moments(Ylogits, [0])
            # update_moving_averages=exp_moving_avg.apply([mean, variance])
            def mean_var_with_update():
                ema_apply_op = ema.apply([mean, var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(mean), tf.identity(var)
            # m = mean
            # v = variance

            # mean, var = tf.cond(is_test, mean_var_with_update, lambda: (ema.average(mean), ema.average(var)))   
            print is_test  
            # v = tf.cond(is_test, lambda: exp_moving_avg(variance), lambda: variance)
            Ybn = tf.nn.batch_normalization(Ylogits, mean, var, beta, scale, variance_epsilon=1e-5)
            return Ybn


        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)

        def batch_norm_wrapper(inputs, is_training=False, decay = 0.99):
            scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
            beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
            epsilon = 1e-5
            pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
            pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

            if is_training:
                batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
                train_mean = tf.assign(pop_mean,
                                       pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(pop_var,
                                      pop_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(inputs,
                        batch_mean, batch_var, beta, scale, epsilon)
            else:
                return tf.nn.batch_normalization(inputs,
                    pop_mean, pop_var, beta, scale, epsilon)

        # def make_CNN_layer(x_input, n_pool, att_win, freq_win, out_layers, in_layers, is_test, xWidth):
        #     W_conv = weight_variable([att_win, freq_win, in_layers, out_layers])
        #     h_conv = conv2d(x_input, W_conv) #+ b_conv1
        #     # Scale = tf.Variable(tf.ones([x_inputs.get_shape()[-1]]))
        #     # Offset = tf.Variable(tf.zeros([x_inputs.get_shape()[-1]]))
        #     # h_norm = batch_norm_wrapper(h_conv, is_training)
        #     h_norm = batchnorm_layer(h_conv, is_test)#, is_test)
        #     h_actv = tf.nn.relu(h_norm)
        #     h_pool = max_pool_nx1(h_actv,n_pool)
        #     xWidth = int(math.ceil(xWidth/float(n_pool)))
        #     return h_pool, xWidth, out_layers


        #with tf.name_scope('layer1'):

        num_filt1 = 3
        n_pool1 = 3
        self.num_filt1 = num_filt1
     #   with tf.name_scope('weights'):
        W_conv1 = weight_variable([7, 5, 3,  num_filt1])
        variable_summaries(W_conv1)
     #   with tf.name_scope('convs'):
        h_conv1 = conv2d(x_image, W_conv1) #+ b_conv1
       # tf.summary.histogram('pre_norm',h_conv1)
        # h_norm1 = batchnorm_layer(h_conv1, False, beta1, scale1, 2000, True)
        h_norm1 = batch_norm_wrapper(h_conv1, True)
     #   tf.summary.histogram('norm',h_norm1)
        h_actv1 = tf.nn.relu(h_norm1)
     #   tf.summary.histogram('activations',h_actv1)
        h_pool1 = max_pool_nx1(h_actv1,n_pool1)
        xWidth1 = int(math.ceil(self.xWidth/float(n_pool1)))

        num_filt2 = 5
        n_pool2 = 1
        W_conv2 = weight_variable([1, 5, num_filt1, num_filt2])
        # b_conv2 = bias_variable([num_filt2])
        h_conv2 = conv2d(h_pool1, W_conv2)# + b_conv2
        h_norm2 = batch_norm_wrapper(h_conv2, True)
        h_actv2 = tf.nn.relu(h_norm2)
        h_pool2 = max_pool_nx1(h_actv2, n_pool2)
        xWidth2 = int(math.ceil(xWidth1/float(n_pool2)))

        num_filt3 = 6
        n_pool3 = 1
        W_conv3 = weight_variable([1, 4, num_filt2, num_filt3])
        # b_conv3 = bias_variable([num_filt3])
        
        h_conv3 = conv2d(h_pool2, W_conv3)
        h_norm3 = batch_norm_wrapper(h_conv3, True)
        h_actv3 = tf.nn.relu(h_norm3)
        h_pool3 = max_pool_nx1(h_actv3, n_pool3)
        xWidth3 = int(math.ceil(xWidth2/float(n_pool3)))
        cWidth3 = num_filt3

        # def build_graph(is_training=False):

        # 3 x CNN layers
        # actv1, xWidth, out_layers = make_CNN_layer(x_input = x_image, 
        #                                     n_pool =3,
        #                                     att_win = 7,
        #                                     freq_win = 5,
        #                                     in_layers =3,
        #                                     out_layers =3,
        #                                     xWidth = self.xWidth,
        #                                     is_test= is_test)
        #                                     # is_training =True)

        # actv2, xWidth, out_layers = make_CNN_layer(x_input = actv1, 
        #                                     n_pool =1,
        #                                     att_win = 1,
        #                                     freq_win = 5,
        #                                     in_layers =out_layers,
        #                                     out_layers =5,
        #                                     xWidth = xWidth, 
        #                                     is_test= is_test)
        #                                     # is_training =True)

        # h_pool3, xWidth, out_layers = make_CNN_layer(x_input = actv2, 
        #                                         n_pool =1,
        #                                         att_win = 1,
        #                                         freq_win = 4,
        #                                         in_layers =out_layers,
        #                                         out_layers =6,
        #                                         xWidth = xWidth,
        #                                         is_test= is_test)
        #                                         # is_training =True)

        num_fc_filt = int(math.ceil(math.ceil(np.shape(trainImages)[2]/2)/2)  * np.shape(trainImages)[1] * 6)

        self.keep_prob = tf.placeholder(tf.float32)
        if fully_connected:
            N = 200
            YY = tf.reshape(h_pool3, shape=[-1, self.nClass*num_filt3*xWidth3])
            # YY = tf.reshape(h_pool3, shape=[-1, 2652])

            W4 = tf.Variable(tf.truncated_normal([1530, N], stddev=0.1))
            Y4l = tf.matmul(YY, W4)
            # Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4)
            Y4r = tf.nn.relu(Y4l)
            # Y4 = tf.nn.dropout(Y4r, pkeep)
            W5 = tf.Variable(tf.truncated_normal([N, max_nClass], stddev=0.1))
            B5 = tf.Variable(tf.constant(0.1, tf.float32, [max_nClass]))
            Ylogits = tf.matmul(Y4r, W5) + B5
            self.y = tf.nn.softmax(Ylogits)

        else:
            h_pool3_flat = tf.reshape(h_pool3,[-1,self.nClass,num_filt3*xWidth3,1])
            W_final = weight_variable([1, num_filt3*xWidth3, 1, 1])
            # b_final = bias_variable([1])
            h_conv_final = tf.nn.conv2d(h_pool3_flat, W_final, strides=[1, 1, 1, 1], padding='VALID')# + b_final
            # h_conv_final = batch_norm_wrapper(h_conv_final, True)
            h_conv_final = tf.reshape(h_conv_final, [-1,self.nClass])
            h_conv_final = tf.nn.dropout(h_conv_final, self.keep_prob)

            self.y=tf.nn.softmax(h_conv_final) #h_fc1_drop

        y_ = tf.placeholder(tf.float32, [None, self.nClass]) # true class lables identified by user 

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(self.y+ 1e-10), reduction_indices=[1])) # find optimum solution by minimizing error
        tf.summary.scalar('cross_entropy', cross_entropy)

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy) # the best result is when the wrongness is minimal

        # saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1)) #which ones did it get right?
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

            # return (x, y_), train_step, accuracy, y, saver

        # (x, y_), train_step, accuracy, _, saver = build_graph(is_training=True)

        saver = tf.train.Saver()

        print self.trainFile
        modelName = ('.').join(self.trainFile.split('.')[:-1]) + '.meta'
        # modelName = 'my-model.meta'
        print modelName

        # init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()

        # if os.path.isfile("%s" % ('my-model.meta')):
        if os.path.isfile("%s%s" % (self.mldir,modelName)):
            self.sess = tf.Session()
            self.sess.run(init)           

            # Restore variables from disk.
            # saver =tf.train.import_meta_graph(self.mldir+modelName)

            saver.restore(self.sess, tf.train.latest_checkpoint(self.mldir) )
        
        else:
            self.sess = tf.Session()
            # print is_test, self.keep_prob
            self.sess.run(init) 

            # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
            merged = tf.summary.merge_all()
            # train_writer = tf.summary.FileWriter(mdd + '/train',
            #                                       self.sess.graph)
            # test_writer = tf.summary.FileWriter(mdd + '/test')
            # tf.global_variables_initializer().run()


            start_time = time.time()

            # self.trainReps = 2000
            # self.batches = 50

            # if np.shape(trainLabels)[0]< batches:
            #     batches = np.shape(trainLabels)[0]/2


            print "--- %s seconds ---" % (time.time() - start_time)

            train_ce, train_acc, test_ce, test_acc = [], [], [], []

            def training_step(i, batch_xs, batch_ys):
                learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i / decay_speed)


                # if i % 10 == 0:  # Record summaries and test-set accuracy
                #     summary, acc = self.sess.run([merged, accuracy], feed_dict={self.x: batch_xs, y_: batch_ys, lr: learning_rate, self.keep_prob:1, is_test: True})
                #     test_writer.add_summary(summary, i)

                if i % 10 == 0:
                    train_entropy, train_score = self.sess.run([cross_entropy, accuracy],
                                                         feed_dict={self.x: batch_xs, y_: batch_ys, lr: learning_rate,
                                                                    self.keep_prob: 1, is_test: True})
                    train_ce.append(train_entropy)
                    train_acc.append(train_score)
                if i % 100 == 0:
                    test_entropy, test_score = self.sess.run([cross_entropy, accuracy],
                                                        feed_dict={self.x: testImages, y_: testLabels,
                                                                   lr: learning_rate,
                                                                   self.keep_prob: 1, is_test: False})

                    test_ce.append(test_entropy)
                    test_acc.append(test_score)
                    print i, learning_rate, train_entropy, train_score, test_entropy, test_score

                    # saver.save(self.sess, os.path.join(LOG_DIR, "model.ckpt"), i)

                # self.sess.run(train_step, feed_dict={self.x: batch_xs, y_: batch_ys, self.keep_prob:1, is_test: True}) #calculate train_step using feed_dict
                summary, _ = self.sess.run([merged, train_step],
                                           feed_dict={self.x: batch_xs, y_: batch_ys, lr: learning_rate,
                                                      self.keep_prob: 1,
                                                      is_test: True})
                # train_writer.add_summary(summary, i)

            print 'Performing', trainReps, 'training repeats, using batches of', batches
            for i in range(trainReps):  # perform the training step using random batches of images and according labels
                batch_xs, batch_ys = next_batch(trainImages, trainLabels, self.batches)
                training_step(i, batch_xs, batch_ys)

            # train_writer.close()
            # test_writer.close()

            ys_true_train = self.sess.run(tf.argmax(y_, 1), feed_dict={self.x: trainImages, y_: trainLabels})
            ys_guess_train = self.sess.run(tf.argmax(self.y, 1),
                                           feed_dict={self.x: trainImages, y_: trainLabels, self.keep_prob: 1})
            print 'true train class labels: ', ys_true_train
            print 'class train estimates:   ', ys_guess_train

            missed = []
            for i, y in enumerate(ys_true_train):
                if ys_guess_train[i] != y:
                    missed.append(i)

            missed = np.asarray(missed)
            print missed

            if recursive:
                # def recurrent(trainImages, trainLabels):

                trainImages = trainImages[missed]
                trainLabels = trainLabels[missed]

                print 'Performing', trainReps, 'training repeats, using batches of', batches
                for i in range(
                        trainReps+1):  # perform the training step using random batches of images and according labels
                    batch_xs, batch_ys = next_batch(trainImages, trainLabels, self.batches)
                    training_step(i, batch_xs, batch_ys)

            score = self.sess.run(accuracy, feed_dict={self.x: testImages, y_: testLabels,  lr: min_learning_rate,self.keep_prob:1, is_test: False}) * 100
            print 'Accuracy of model in testing: ', score, '%'

            mlt.plot_accuracy(train_ce, test_ce, train_acc, test_acc) 

            print "%s%s" % (self.mldir,modelName)
            # save_path = saver.save(self.sess, "%s%s" % (self.mldir,modelName))
            save_path = saver.save(self.sess, self.mldir+modelName[:-5])
            print("Model saved in file: %s" % save_path)

        ys_true = self.sess.run(tf.argmax(y_,1), feed_dict={self.x: testImages, y_: testLabels})
        ys_guess = self.sess.run(tf.argmax(self.y,1), feed_dict={self.x: testImages, y_: testLabels, self.keep_prob:1})

        print 'true class labels: ', ys_true
        print 'class estimates:   ', ys_guess

        print np.sum(self.sess.run(self.y, feed_dict={self.x: testImages, y_: testLabels, self.keep_prob:1}),axis=0)

        print 'within 1dB:', float(len(np.where(abs(ys_true-ys_guess) <= 1)[0]))/len(ys_guess)
        
        if plot_confusion:
            mlt.plot_confusion(ys_true, ys_guess)
        if plot_missed:
            mlt.plot_missed(ys_true, ys_guess, testImages)

        # recurrent(trainImages, trainLabels)
        score = self.sess.run(accuracy, feed_dict={self.x: testImages, y_: testLabels, self.keep_prob:1}) * 100
        print 'Accuracy of model in testing: ', score, '%'
        if score < 85: print 'Consider making more training images'

        plot_weights = ''
        if plot_weights == 'post':
            weights = [self.sess.run(W_conv1), self.sess.run(W_conv2), self.sess.run(W_conv3)]    
            # weights = self.sess.run(W_conv1)
            mlt.plotWeights(weights)

        # plot_activations =''
        print plot_activations
        if plot_activations == 'post':
            activations = [self.sess.run(x_image,feed_dict={self.x: testImages}),
                self.sess.run(h_conv1, feed_dict={self.x: testImages}),
                self.sess.run(h_norm1, feed_dict={self.x: testImages}),
                self.sess.run(h_actv1, feed_dict={self.x: testImages}),
                self.sess.run(h_pool1, feed_dict={self.x: testImages}),
                self.sess.run(h_actv2, feed_dict={self.x: testImages}),
                self.sess.run(h_actv3, feed_dict={self.x: testImages}),]
            mlt.plotActivations(activations)
   
        # # return sess



    def findPowers(self, inferenceFile, showFrames = False, plot_res= False, searchAllRes=True, res_nums=50):
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

        if not inferenceFile is None:
            print 'Inference File:', inferenceFile
            inferenceData = PSFitMLData(h5File = inferenceFile, useAllAttens = True)

        if searchAllRes:
            res_nums = np.shape(inferenceData.freqs)[0]

        print 'res nums',  res_nums
        resonators = range(res_nums)
        nattens = np.shape(inferenceData.attens)[1]
        # nattens = max_nClass
        #
        self.inferenceLabels = np.zeros((res_nums,max_nClass))
        self.low_stds = np.zeros((res_nums))
        self.ang_means = np.zeros((res_nums,max_nClass))
        self.ang_stds = np.zeros((res_nums,max_nClass))
        self.max_ratios = np.zeros((res_nums,max_nClass))
        self.running_ratios = np.zeros((res_nums,max_nClass))
        self.ratio_guesses = np.zeros((res_nums))
        #
        # self.atten_guess_mode=np.zeros((res_nums))
        # self.atten_guess_mean=np.zeros((res_nums))
        # self.atten_guess_med=np.zeros((res_nums))

        print 'Using trained algorithm on images on each resonator'
        self.bad_res = []
        
        # angles = np.linspace(0, 2*math.pi, 8)
        angles = [0]
        # perturbations = [-2,-1,0,1,2]
        perturbations = [0]
        # scalings = [1,1.1,0.9]
        scalings = [1]
        # print angles

        from scipy.stats import mode
        for i,rn in enumerate(resonators): 

            sys.stdout.write("\r%d of %i" % (i+1,res_nums) )
            sys.stdout.flush()

            # indx=[]
            # for ia in range(nattens):
            #     indx.append(argmax(iq_vels[rn,ia,:]))
            # plt.plot(attens,freqs[rn,indx]/1e9)
            # plt.show()


            # angles_nonsat, ratio_nonsat, ratio, running_ratio, bad_res, ang_mean, ang_std = mlt.checkResAtten(inferenceData, res_num=rn)
            # # noisy_res = 0
            # low_std = np.argmax(ang_mean/ang_std)
            # # low_std = np.argmax(ang_mean)
            # # low_std = np.argmin(ang_std)

            # self.low_stds[rn] = low_std
            # self.ang_means[rn,:inferenceData.nClass] = ang_mean
            # self.ang_stds[rn, :inferenceData.nClass] = ang_std
            # self.max_ratios[rn, :inferenceData.nClass] = ratio
            # self.running_ratios[rn, :inferenceData.nClass] = running_ratio
            # # self.ratio_guess = np.where(running_ratio/max(running_ratio)<0.4)[0][0]
            # ratio_guess = np.where(running_ratio<2.5)[0][0]
            
            # # print ratio_guess
            # if ratio_guess<nattens-1:
            #     while (running_ratio[ratio_guess] - running_ratio[ratio_guess+1] > 0.1) and (ratio_guess<nattens-2):
            #         ratio_guess +=1
            #         # print ratio_guess, nattens-2
            # if type(ratio_guess) ==np.int64:
            #     self.ratio_guesses[rn] = ratio_guess

            bad_res = False
            if not bad_res:
                # for ia in range(nattens-1):                
                # first check the loop for saturation           
               
                # nonsaturated_loop = angles_nonsat[ia] and ratio_nonsat[ia]
                # nonsaturated_loop = True
                # if nonsaturated_loop:
                # each image is formatted into a single element of a list so sess.run can receive a single values dictionary 
                perspectives = []
                # for angle in angles:
                #     for pert in perturbations:
                        # for scale in scalings:
                image = mlt.makeResImage(inferenceData, res_num = rn,showFrames=False)
                inferenceImage=[]
                inferenceImage.append(image)            # inferenceImage is just reformatted image
                self.inferenceLabels[rn,:] = self.sess.run(self.y, feed_dict={self.x: inferenceImage,self.keep_prob:1} )
                # perspectives.append( np.argmax(self.inferenceLabels[rn,:]) )
                # # perspectives = np.asarray(perspectives)
                # self.atten_guess_mode[rn] = mode(perspectives).mode[0]
                # self.atten_guess_mean[rn] = int(np.mean(perspectives))
                # self.atten_guess_med[rn] = np.median(perspectives)

                del inferenceImage
                del image

               
                # inferenceLabels[rn,0,0]= inferenceLabels[rn,1,0] # since 0th term is skipped (and therefore 0)
            # else:
            #     inferenceLabels[rn,:] = [1,0]
            else:
                self.bad_res.append(rn)
                self.inferenceLabels[rn,:] = np.zeros((nattens)) # removed later anyway

                # print np.shape(inferenceLabels), rn
                # inferenceLabels = np.delete(inferenceLabels,rn,0)
                # inferenceLabels = inferenceLabels[:-1] 
        
        # print inferenceLabels[:,19,0]
        # inferenceLabels = np.delete(inferenceLabels,bad_res,0)
        # print inferenceLabels[:,19,0]
        #
        #     if np.all(inferenceLabels[rn,:,1] ==0): # if all loops appear saturated for resonator then set attenuation to highest
        #         #best_guess = argmax(inferenceLabels[rn,:,1])
        #         #print best_guess
        #         # print inferenceLabels[rn,:,1]
        #         # print rn
        #
        #         best_guess = 20#int(np.random.normal(nattens*2/5, 3, 1))
        #         if best_guess > nattens: best_guess = nattens
        #         inferenceLabels[rn,best_guess,:] = [0,1]  # or omit from list
        #
        #     if noisy_res >= 15:#nattens:
        #         inferenceLabels[rn,:] = [0,0,0]
        #         inferenceLabels[rn,5] = [0,1,0]
        #         skip.append(rn)
        print '\n'

        # # res_nums = res_nums - len(bad_res)
        # self.max_2nd_vels = np.zeros((res_nums,nattens))
        # for r in range(res_nums):
        #     for iAtten in range(nattens):
        #         vindx = (-inferenceData.iq_vels[r,iAtten,:]).argsort()[:2]
        #         self.max_2nd_vels[r,iAtten] = inferenceData.iq_vels[r,iAtten,vindx[1]]

        self.atten_guess=np.zeros((res_nums))
        # # choose attenuation where there is the maximum in the 2nd highest IQ velocity
        # bad=0
        # inferenceLabels[:,-1,1] = 1
        for r in range(res_nums):
            # print inferenceLabels[r]
            # print ratio_guess[r]
            # print inferenceLabels[r][int(ratio_guess[r]):]
            # print argmax(inferenceLabels[r][int(ratio_guess[r]):])
            # print argmax(inferenceLabels[r][int(ratio_guess[r]):])+int(ratio_guess[r])
            # print r, self.inferenceLabels[r], self.ratio_guesses[r], self.inferenceLabels[r][int(self.ratio_guesses[r]):-1]

            # if self.ratio_guesses[r] >= nattens -1:
            #     self.atten_guess[r] = nattens
            # else:
            # self.atten_guess[r] = np.argmax(self.inferenceLabels[r][int(self.ratio_guesses[r]):-1])+ int(self.ratio_guesses[r])
            self.atten_guess[r] = np.argmax(self.inferenceLabels[r])
            # self.atten_guess[r] = argmax(self.inferenceLabels[r])
            # print inferenceLabels[r]

def next_batch(trainImages, trainLabels, batch_size):
    '''selects a random batch of batch_size from trainImages and trainLabels'''
    perm = random.sample(range(len(trainImages)), batch_size)
    trainImages = np.array(trainImages)[perm,:]
    trainLabels = np.array(trainLabels)[perm,:]
    return trainImages, trainLabels

    
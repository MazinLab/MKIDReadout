'''
Class for loading and/or saving ML data for PS fitting.  Class contains one data set, could be either inference or training.  Typical
use would have the save functionality used exclusively for inference.

'''
import os, sys, inspect
# from PSFit import *
from iqsweep import *
import numpy as np
import sys, os
import matplotlib.pyplot as plt
# import tensorflow as tf
import pickle
import random
import time
import math
from scipy import interpolate
import PSFitMLTools as mlt

np.set_printoptions(threshold=np.inf)
from ml_params import mldir, trainFile, testFrac, max_nClass, trainBinFile, res_per_class

# removes visible depreciation warnings from lib.iqsweep
import warnings

warnings.filterwarnings("ignore")


class PSFitting():
    '''Class has been lifted from PSFit.py and modified to incorporate the machine learning algorithms from
    WideAna_ml.py
    '''

    def __init__(self, initialFile=None):
        self.initialFile = initialFile
        self.resnum = 0

    # def loadres(self, useResID=False):
    def loadres(self):
        '''
        Outputs
        Freqs:          the span of frequencies for a given resonator
        iq_vels:        the IQ velocities for all attenuations for a given resonator
        Is:             the I component of the frequencies for a given resonator
        Qs:             the Q component of the frequencies for a given resonator
        attens:         the span of attenuations. The same for all resonators
        '''

        self.Res1 = IQsweep()
        self.Res1.LoadPowers(self.initialFile, 'r0', self.freq[self.resnum])

        self.resfreq = self.freq[self.resnum]
        self.NAttens = len(self.Res1.atten1s)
        self.res1_iq_vels = np.zeros((self.NAttens, self.Res1.fsteps - 1))
        self.res1_iq_amps = np.zeros((self.NAttens, self.Res1.fsteps))
        for iAtt in range(self.NAttens):
            for i in range(1, self.Res1.fsteps - 1):
                self.res1_iq_vels[iAtt, i] = np.sqrt((self.Res1.Qs[iAtt][i] - self.Res1.Qs[iAtt][i - 1]) ** 2 + (
                self.Res1.Is[iAtt][i] - self.Res1.Is[iAtt][i - 1]) ** 2)
                self.res1_iq_amps[iAtt, :] = np.sqrt((self.Res1.Qs[iAtt]) ** 2 + (self.Res1.Is[iAtt]) ** 2)
        # Sort the IQ velocities for each attenuation, to pick out the maximums

        sorted_vels = np.sort(self.res1_iq_vels, axis=1)
        # Last column is maximum values for each atten (row)
        self.res1_max_vels = sorted_vels[:, -1]
        # Second to last column has second highest value
        self.res1_max2_vels = sorted_vels[:, -2]
        # Also get indices for maximum of each atten, and second highest
        sort_indices = np.argsort(self.res1_iq_vels, axis=1)
        max_indices = sort_indices[:, -1]
        max2_indices = sort_indices[:, -2]
        max_neighbor = max_indices.copy()

        # for each attenuation find the ratio of the maximum velocity to the second highest velocity
        self.res1_max_ratio = self.res1_max_vels.copy()
        max_neighbors = np.zeros(self.NAttens)
        max2_neighbors = np.zeros(self.NAttens)
        self.res1_max2_ratio = self.res1_max2_vels.copy()
        for iAtt in range(self.NAttens):
            if max_indices[iAtt] == 0:
                max_neighbor = self.res1_iq_vels[iAtt, max_indices[iAtt] + 1]
            elif max_indices[iAtt] == len(self.res1_iq_vels[iAtt, :]) - 1:
                max_neighbor = self.res1_iq_vels[iAtt, max_indices[iAtt] - 1]
            else:
                max_neighbor = np.maximum(self.res1_iq_vels[iAtt, max_indices[iAtt] - 1],
                                          self.res1_iq_vels[iAtt, max_indices[iAtt] + 1])
            max_neighbors[iAtt] = max_neighbor
            self.res1_max_ratio[iAtt] = self.res1_max_vels[iAtt] / max_neighbor
            if max2_indices[iAtt] == 0:
                max2_neighbor = self.res1_iq_vels[iAtt, max2_indices[iAtt] + 1]
            elif max2_indices[iAtt] == len(self.res1_iq_vels[iAtt, :]) - 1:
                max2_neighbor = self.res1_iq_vels[iAtt, max2_indices[iAtt] - 1]
            else:
                max2_neighbor = np.maximum(self.res1_iq_vels[iAtt, max2_indices[iAtt] - 1],
                                           self.res1_iq_vels[iAtt, max2_indices[iAtt] + 1])
            max2_neighbors[iAtt] = max2_neighbor
            self.res1_max2_ratio[iAtt] = self.res1_max2_vels[iAtt] / max2_neighbor
        # normalize the new arrays
        self.res1_max_vels /= np.max(self.res1_max_vels)
        self.res1_max_vels *= np.max(self.res1_max_ratio)
        self.res1_max2_vels /= np.max(self.res1_max2_vels)

        max_ratio_threshold = 2.5  # 1.5
        rule_of_thumb_offset = 1  # 2

        # require ROTO adjacent elements to be all below the MRT
        bool_remove = np.ones(len(self.res1_max_ratio))
        for ri in range(len(self.res1_max_ratio) - rule_of_thumb_offset - 2):
            bool_remove[ri] = bool((self.res1_max_ratio[ri:ri + rule_of_thumb_offset + 1] < max_ratio_threshold).all())
        guess_atten_idx = np.extract(bool_remove, np.arange(len(self.res1_max_ratio)))

        # require the attenuation value to be past the initial peak in MRT
        guess_atten_idx = guess_atten_idx[np.where(guess_atten_idx > np.argmax(self.res1_max_ratio))[0]]

        if np.size(guess_atten_idx) >= 1:
            if guess_atten_idx[0] + rule_of_thumb_offset < len(self.Res1.atten1s):
                guess_atten_idx[0] += rule_of_thumb_offset
                guess_atten_idx = int(guess_atten_idx[0])
        else:
            guess_atten_idx = self.NAttens / 2

        try:
            self.Res1.resID
            useResID = True
        except AttributeError:
            useResID = False
        print useResID

        if useResID:
            return {'freq': self.Res1.freq,
                    'resID': self.Res1.resID,
                    'iq_vels': self.res1_iq_vels,
                    'Is': self.Res1.Is,
                    'Qs': self.Res1.Qs,
                    'attens': self.Res1.atten1s}

        return {'freq': self.Res1.freq,
                'iq_vels': self.res1_iq_vels,
                'Is': self.Res1.Is,
                'Qs': self.Res1.Qs,
                'attens': self.Res1.atten1s}

    def loadps(self):
        hd5file = openFile(self.initialFile, mode='r')
        group = hd5file.getNode('/', 'r0')
        self.freq = np.empty(0, dtype='float32')

        for sweep in group._f_walkNodes('Leaf'):
            k = sweep.read()
            self.scale = k['scale'][0]
            # print "Scale factor is ", self.scale
            self.freq = np.append(self.freq, [k['f0'][0]])
        hd5file.close()


class PSFitMLData():
    # def __init__(self, h5File=None, useAllAttens=True, useResID=True):
    def __init__(self, h5File=None, PSFile=None, useAllAttens=True):
        self.useAllAttens = useAllAttens
        # self.useResID=useResID
        self.h5File = h5File
        if PSFile == None:
            self.PSFile = self.h5File[:-19] + '.txt'  # 'x-reduced.txt'
        else:
            self.PSFile = PSFile
        self.PSPFile = self.h5File[:-19] + '.pkl'
        self.baseFile = self.h5File[:-19]

        self.freqs, self.iq_vels, self.Is, self.Qs, self.attens, self.resIDs = self.get_PS_data()

        self.opt_attens = None
        self.opt_freqs = None
        self.nClass = np.shape(self.attens)[1]
        # self.nClass = max_nClass
        self.xWidth = 50

        # self.trainFile = 'ps_peaks_train_iqv_allres_c%i.pkl' % (self.nClass)
        self.mldir = mldir
        self.trainFile = trainFile
        self.trainBinFile = trainBinFile
        self.testFrac = testFrac
        self.trainFrac = 1 - self.testFrac
        # self.mldir = os.environ['MLDIR'] #'./cache/'
        # self.trainFile = os.environ['TRAINFILE']# 'ps_train.pkl'
        # self.trainFrac = 0.9
        # self.testFrac=1 - self.trainFrac

    def loadRawTrainData(self):
        '''
        Loads in a PS frequency text file to use in a training set.  Populates the variables self.opt_attens and self.opt_freqs.

        '''
        PSFile = np.loadtxt(self.PSFile, skiprows=1)

        print np.shape(PSFile)
        if np.shape(PSFile)[1] > 3:
            print 'This train file is in the old format'
            print 'loading peak location data from %s' % self.PSFile
            # PSFile = np.loadtxt(self.PSFile, skiprows=1)
            opt_freqs = PSFile[:, 0]
            self.opt_attens = PSFile[:, 3] + 1
            # print self.opt_attens[95:105]
            self.opt_attens[99] = 40
            # print self.opt_attens[95:105]
            print 'adding one'
            # exit()
            # print 'psfile shape', np.shape(PSFile)

            all_freqs = np.around(self.freqs, decimals=-4)
            opt_freqs = np.around(opt_freqs, decimals=-4)
            good_res = np.arange(len(self.freqs))
            a, b = 0, 0

            for g in range(len(opt_freqs) - 2):
                # print g, a, opt_freqs[g], [all_freqs[a,0], all_freqs[a,-1]]
                while opt_freqs[g] not in all_freqs[a, :]:
                    good_res = np.delete(good_res,
                                         g + b)  # identify this value of all_freqs as bad by removing from list
                    a += 1  # keep incrementing until opt_freqs matches good_freqs
                a += 1  # as g increments so does a
            iFinTrainRes = np.where(opt_freqs[-1] == np.around(self.freqs[good_res], decimals=-4))[0][0] + 1
            self.good_res = good_res[:iFinTrainRes]

        else:
            print 'loading peak location data from %s' % self.PSFile
            PSFile = np.loadtxt(self.PSFile, skiprows=0)
            opt_freqs = PSFile[:, 1]
            self.opt_attens = PSFile[:, 2]  # + 1
            goodResIDs = PSFile[:, 0]
            # self.opt_attens[92] = 40

            # print goodResIDs
            jumps = np.roll(goodResIDs, 1)[1:] - goodResIDs[1:]
            if np.any(jumps != -1):
                useResID = True
            else:
                useResID = False

            print useResID, 'lol'
            if useResID:
                goodResIDs = PSFile[:, 0]
                self.good_res = np.where(map(lambda x: np.any(x == goodResIDs), self.resIDs))[0]
            else:
                self.good_res = np.array(PSFile[:, 0] - PSFile[0, 0], dtype='int')

        self.res_nums = len(self.good_res)
        self.attens_orig = self.attens

        self.attens = self.attens[self.good_res, :]

        optAttenLocs = np.where(np.transpose(
            np.transpose(np.array(self.attens)) == np.array(self.opt_attens)))  # find optimal atten indices
        print np.shape(optAttenLocs)
        optAttenExists = optAttenLocs[0]
        self.opt_iAttens = optAttenLocs[1]

        # print optAttenLocs
        print self.attens[:11], self.opt_attens[:11]
        attenSkips = optAttenLocs[0] - np.arange(len(optAttenLocs[0]))
        attenSkips = np.where(np.diff(attenSkips))[0] + 1  # indices where opt atten was not found

        print attenSkips
        for resSkip in attenSkips:
            print 'resSkip', resSkip
            print self.opt_attens[resSkip], self.attens[resSkip, 0], self.attens[resSkip, -1]
            if (self.opt_attens[resSkip] < self.attens[resSkip, 0]):
                self.opt_attens[resSkip] = self.attens[resSkip, 0]
                self.opt_iAttens = np.insert(self.opt_iAttens, resSkip, 0)
            elif (self.opt_attens[resSkip] > self.attens[resSkip, -1]):
                self.opt_attens[resSkip] = self.attens[resSkip, -1]
                self.opt_iAttens = np.insert(self.opt_iAttens, resSkip, np.shape(self.attens)[1] - 1)
            else:
                raise ValueError('Atten skip index error')

        # print self.good_res
        self.freqs_orig = np.around(self.freqs, decimals=-4)
        self.opt_freqs_orig = np.around(opt_freqs, decimals=-4)
        self.iq_vels_orig = self.iq_vels
        self.Is_orig = self.Is
        self.Qs_orig = self.Qs
        self.resIDs_orig = self.resIDs

        # for r in range(10):
        #     print r, self.opt_iAttens[r]

        # rn = test_ind[19]
        # print 19, rn, self.opt_iAttens[rn], 'true value 16'
        # exit()
        # for ia, a in enumerate(self.opt_iAttens):
        #     if a >max_nClass:
        #         print ia, a
        #         self.good_res = np.delete(self.good_res, np.where(self.good_res ==ia)[0])

        self.good_res = np.delete(self.good_res, np.where(self.opt_iAttens >= max_nClass)[0])
        # self.attens = np.delete(self.attens, np.where(self.opt_iAttens>=max_nClass))
        self.opt_iAttens = np.delete(self.opt_iAttens, np.where(self.opt_iAttens >= max_nClass)[0])

        print np.shape(self.opt_iAttens)

        # print 'optfreqs len', len(opt_freqs)
        # print 'self.freqs len', len(self.freqs)
        # print self.good_res
        print type(self.good_res), type(self.resIDs)
        self.freqs = np.around(self.freqs[self.good_res], decimals=-4)
        self.opt_freqs = np.around(opt_freqs, decimals=-4)
        self.iq_vels = self.iq_vels[self.good_res]
        self.Is = self.Is[self.good_res]
        self.Qs = self.Qs[self.good_res]
        self.resIDs = np.asarray(self.resIDs)[self.good_res]

    def makeBinTrainData(self):
        '''
        Training data for the binary classifier: good res or not?

        Creates the training (and testing) images and labels to be passed to mlClass() which is saved in a .pkl and read using loadPkl()
        '''

        self.loadRawTrainData()

        good_res = np.arange(len(self.resIDs))

        # self.res_nums = len(good_res)
        self.res_nums = 400

        trainImages, trainLabels, testImages, testLabels = [], [], [], []

        # select resonators uniformally distributed across the range for training and testing
        train_ind = np.array(map(int, np.linspace(0, self.res_nums - 1, self.res_nums * self.trainFrac)))

        test_ind = []
        np.array([test_ind.append(el) for el in range(self.res_nums) if el not in train_ind])

        for rn in train_ind:

            good_res = np.any(self.good_res == rn)
            print rn, good_res
            if good_res != 1:
                angles = np.linspace(0, 2 * math.pi, 32)
                # showFrames=True
            else:
                angles = [0]
                # showFrames = False

            for angle in angles:
                image = mlt.makeBinResImage(self, res_num=rn, angle=angle, showFrames=False)
                trainImages.append(image)
                one_hot = np.zeros(2)
                # 1 if good_res 0 if bad
                # good_res = np.any(self.good_res == rn)
                one_hot[good_res] = 1
                trainLabels.append(one_hot)

        for rn in test_ind:
            good_res = np.any(self.good_res == rn)
            print rn, good_res
            if good_res != 1:
                angles = np.linspace(0, 2 * math.pi, 32)
                # showFrames=True
            else:
                angles = [0]
                # showFrames = False

            for angle in angles:
                image = mlt.makeBinResImage(self, res_num=rn, angle=angle, showFrames=False)
                testImages.append(image)
                one_hot = np.zeros(2)
                # good_res = np.any(self.good_res == rn)
                one_hot[good_res] = 1
                testLabels.append(one_hot)

        append = None
        if os.path.isfile(self.mldir + self.trainBinFile):
            append = raw_input('Do you want to append this training data to previous data [y/n]')
        if (append == 'n'):
            self.trainBinFile = self.trainBinFile.split('-')[0] + time.strftime("-%Y-%m-%d-%H-%M-%S")
        if (append == 'y') or (os.path.isfile(self.trainFile) == False):
            print 'saving %s to %s' % (
            self.mldir + self.trainBinFile, os.path.dirname(os.path.abspath(self.trainBinFile)))
            with open(self.mldir + self.trainBinFile, 'ab') as tf:
                pickle.dump([trainImages, trainLabels], tf)
                pickle.dump([testImages, testLabels], tf)

    def makeTrainData(self, res_per_class):
        '''
        Training data for the main algorithm with a class for each attenuation

        Creates the training (and testing) images and labels to be passed to mlClass() which is saved in a .pkl and read using loadPkl()

        '''

        self.loadRawTrainData()

        good_res = np.arange(len(self.resIDs))

        # for i in range(self.nClass):
        #     iAttens[:,i] =self.opt_iAttens + attDist[i]

        self.res_nums = len(good_res)
        # self.res_nums = 400
        # print self.good_res

        # select resonators uniformally distributed across the range for training and testing
        train_ind = np.array(map(int, np.linspace(0, self.res_nums - 1, self.res_nums * self.trainFrac)))
        test_ind = []
        np.array([test_ind.append(el) for el in range(self.res_nums) if el not in train_ind])

        hist, bins = np.histogram(self.opt_iAttens[train_ind], range(max_nClass + 1))

        print hist, len(hist), bins, len(bins)
        print res_per_class

        # plot the original distribution of classes
        plt.plot(hist)
        plt.show()

        # this bit of code is concerned with evening out the classes with label preserving transformation duplications
        class_prob = np.zeros((max_nClass))
        for c in range(max_nClass):
            diff = res_per_class - hist[c]
            # print c, diff, hist[c]
            try:
                class_prob[c] = float(diff) / hist[c]
            except ZeroDivisionError:
                print 'class %i has no resonators' % c
            print c, diff, class_prob[c]

        def random_angles(class_label):
            angles_array = []
            guaranteed = np.floor(class_prob[class_label]).astype(int)
            for p in range(guaranteed + 1):  # +1 so at least one orientation is made for each res
                angles_array.append(random.uniform(0, 2 * math.pi))

            rand_event_specifier = random.uniform(0, 1)
            if rand_event_specifier < class_prob[class_label] - guaranteed:
                angles_array.append(random.uniform(0, 2 * math.pi))

            return angles_array

        res_angles = []
        for r in range(self.res_nums):
            angles = random_angles(self.opt_iAttens[r])
            res_angles.append(angles)

        trainImages, trainLabels, testImages, testLabels = [], [], [], []

        for rn in train_ind:
            for angle in res_angles[rn]:
                image = mlt.makeResImage(self, res_num=rn, angle=angle, showFrames=False)
                trainImages.append(image)
                one_hot = np.zeros(max_nClass)
                one_hot[self.opt_iAttens[rn]] = 1
                print rn, angle, self.opt_iAttens[rn]
                trainLabels.append(one_hot)

        for rn in test_ind:
            for angle in [0]:
                image = mlt.makeResImage(self, res_num=rn, angle=angle, showFrames=False)
                testImages.append(image)
                one_hot = np.zeros(max_nClass)
                one_hot[self.opt_iAttens[rn]] = 1
                testLabels.append(one_hot)

        # else:
        #     self.selectTrainData(train_ind,test_ind)

        # exit()
        append = None
        if os.path.isfile(self.mldir + self.trainFile):
            append = raw_input('Do you want to append this training data to previous data [y/n]')
        if (append == 'n'):
            self.trainFile = self.trainFile.split('-')[0] + time.strftime("-%Y-%m-%d-%H-%M-%S")
        if (append == 'y') or (os.path.isfile(self.trainFile) == False):
            print 'saving %s to %s' % (self.mldir + self.trainFile, os.path.dirname(os.path.abspath(self.trainFile)))
            with open(self.mldir + self.trainFile, 'ab') as tf:
                pickle.dump([trainImages, trainLabels], tf)
                pickle.dump([testImages, testLabels], tf)

    def selectTrainData(self, train_ind, test_ind):
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
        # catagory = ['saturated', 'non-sat']
        group = [train_ind, test_ind]
        include = 'n'

        for t in range(2):
            print 'Should I include this resonator in the training data? [y/n (r/o/b/q)]'
            resonators = group[t]
            print resonators
            ir = 0
            while ir < len(resonators):
                rn = resonators[ir]
                # print self.opt_iAttens[:10], trainLabels[:10], group[t][:10]
                # for c in range(self.nClass):
                if include == 'r':
                    include = 'n'
                    break
                if include == 'b':
                    include = 'n'
                    break

                include = 'n'
                while include == 'n':
                    # print rn, c, catagory[c], iAttens[rn,c]
                    image = mlt.makeResImage(self, res_num=rn, showFrames=True)
                    include = raw_input()

                    if include == 'q':
                        return
                    if include == 'r':
                        # iAttens[rn,:] = [-1,-1]
                        if group == 'test':
                            self.test_ind = np.delete(self.test_ind, ir)
                        break
                    if include == 'n':
                        if c == 0:
                            iAttens[rn, c] -= 1
                        else:
                            iAttens[rn, c] += 1

                    if include == 'o':
                        if c == 0:
                            iAttens[rn, c] -= 1
                        else:
                            iAttens[rn, c] += 1
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

    def savePSTxtFile(self, flag=''):
        '''
        Saves a frequency file after inference.  self.opt_attens and self.opt_freqs
        should be populated by an external ML algorithm.
        '''
        if self.opt_attens is None or self.opt_freqs is None:
            raise ValueError('Classify Resonators First!')

        PSSaveFile = self.baseFile + flag + '.txt'
        sf = open(PSSaveFile, 'wb')
        print 'saving file', PSSaveFile
        print 'baseFile', self.baseFile
        # sf.write('1\t1\t1\t1 \n')

        for r in range(len(self.opt_attens)):
            # line = "%4i \t %10.9e \t %4i \n" % (self.resIDs[r], self.opt_freqs[r],
            #                              self.opt_attens[r])
            # line = "%4i \t %10.9e \t %4i \n" % (r, self.opt_freqs[r],
            #                              self.opt_attens[r])
            line = "%4i \t %10.9e \t %4i \n" % (self.good_res[r], self.opt_freqs[r],
                                                self.opt_attens[r])
            sf.write(line)
            print line
        sf.close()

    def get_PS_data(self, searchAllRes=True, res_nums=-1):
        '''A function to read and write all resonator information so stop having to run the PSFit function on all resonators
        if running the script more than once. This is used on both the initial and inference file

        Inputs:
        h5File: the power sweep h5 file for the information to be extracted from. Can be initialFile or inferenceFile
        '''
        print 'get_PS_data_all_attens H5 file', self.h5File
        if os.path.isfile(self.PSPFile):
            print 'loading pkl file', self.PSPFile
            file = []
            with open(self.PSPFile, 'rb') as f:
                for v in range(6):
                    file.append(pickle.load(f))

            if searchAllRes:
                res_nums = -1
                resIDs = file[0][:]
                freqs = file[1][:]
                iq_vels = file[2][:]
                Is = file[3][:]
                Qs = file[4][:]
                attens = file[5]

            else:
                resIDs = file[0][:]
                freqs = file[1][:res_nums]
                iq_vels = file[2][:res_nums]
                Is = file[3][:res_nums]
                Qs = file[4][:res_nums]
                attens = file[5]

                # print resIDs
                # print np.roll(resIDs,1)[1:] - resIDs[1:]

        else:
            PSFit = PSFitting(initialFile=self.h5File)
            PSFit.loadps()
            tot_res_nums = len(PSFit.freq)
            if searchAllRes:
                res_nums = tot_res_nums

            res_size = np.shape(PSFit.loadres()['iq_vels'])

            freqs = np.zeros((res_nums, res_size[1] + 1))
            iq_vels = np.zeros((res_nums, res_size[0], res_size[1]))
            Is = np.zeros((res_nums, res_size[0], res_size[1] + 1))
            Qs = np.zeros((res_nums, res_size[0], res_size[1] + 1))
            attens = np.zeros((res_nums, res_size[0]))
            resIDs = np.zeros(res_nums)

            for r in range(res_nums):
                sys.stdout.write("\r%d of %i " % (r + 1, res_nums))
                sys.stdout.flush()

                res = PSFit.loadres()

                try:
                    res['resID']
                    useResID = True
                except KeyError:
                    useResID = False

                print useResID
                # exit()
                if useResID:
                    resIDs[r] = res['resID']
                else:
                    resIDs[r] = r

                freqs[r, :] = res['freq']
                print np.shape(iq_vels), np.shape(res['iq_vels'])
                iq_vels[r, :, :] = res['iq_vels']
                Is[r, :, :] = res['Is']
                Qs[r, :, :] = res['Qs']
                attens[r, :] = res['attens']
                PSFit.resnum += 1

            with open(self.PSPFile, "wb") as f:
                pickle.dump(resIDs, f)
                pickle.dump(freqs, f)
                pickle.dump(iq_vels, f)
                pickle.dump(Is, f)
                pickle.dump(Qs, f)
                pickle.dump(attens, f)

        if not (self.useAllAttens):
            attens = attens[0, :]

        return freqs, iq_vels, Is, Qs, attens, resIDs


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
    file = []
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

    print np.shape(file)[0] / 2 - 1
    if np.shape(file)[0] / 2 > 1:
        for i in range(1, np.shape(file)[0] / 2):
            trainImages = np.append(trainImages, file[2 * i][0], axis=0)
            trainLabels = np.append(trainLabels, file[2 * i][1], axis=0)
            testImages = np.append(testImages, file[2 * i + 1][0], axis=0)
            testLabels = np.append(testLabels, file[2 * i + 1][1], axis=0)

    print np.shape(trainLabels)

    print "loaded dataset from ", filename
    return trainImages, trainLabels, testImages, testLabels
''' 
Class for loading and/or saving ML data for PS fitting.  Class contains one data set, could be either inference or training.  Typical
use would have the save functionality used exclusively for inference.

'''

from mkidreadout.utils.iqsweep import *
import numpy as np
import sys, os
import pickle
np.set_printoptions(threshold=np.inf)
#removes visible depreciation warnings from lib.iqsweep
import warnings
warnings.filterwarnings("ignore")

class PSFitting():
    '''Class has been lifted from PSFit.py and modified to incorporate the machine learning algorithms from 
    WideAna_ml.py  
    '''
    def __init__(self, initialFile=None):
        self.initialFile = initialFile
        self.resnum = 0

    def loadres(self, useResID=False):
        '''
        Outputs
        Freqs:          the span of frequencies for a given resonator
        iq_vels:        the IQ velocities for all attenuations for a given resonator
        Is:             the I component of the frequencies for a given resonator
        Qs:             the Q component of the frequencies for a given resonator
        attens:         the span of attenuations. The same for all resonators
        '''
        
        self.Res1=IQsweep()
        self.Res1.LoadPowers(self.initialFile, 'r0', self.freq[self.resnum])
        self.resfreq = self.freq[self.resnum]
        self.NAttens = len(self.Res1.atten1s)
        self.res1_iq_vels=np.zeros((self.NAttens,self.Res1.fsteps-1))
        self.res1_iq_amps=np.zeros((self.NAttens,self.Res1.fsteps))
        for iAtt in range(self.NAttens):
            for i in range(1,self.Res1.fsteps-1):
                self.res1_iq_vels[iAtt,i]=np.sqrt((self.Res1.Qs[iAtt][i]-self.Res1.Qs[iAtt][i-1])**2+(self.Res1.Is[iAtt][i]-self.Res1.Is[iAtt][i-1])**2)
                self.res1_iq_amps[iAtt,:]=np.sqrt((self.Res1.Qs[iAtt])**2+(self.Res1.Is[iAtt])**2)
        #Sort the IQ velocities for each attenuation, to pick out the maximums
        
        sorted_vels = np.sort(self.res1_iq_vels,axis=1)
        #Last column is maximum values for each atten (row)
        self.res1_max_vels = sorted_vels[:,-1]
        #Second to last column has second highest value
        self.res1_max2_vels = sorted_vels[:,-2]
        #Also get indices for maximum of each atten, and second highest
        sort_indices = np.argsort(self.res1_iq_vels,axis=1)
        max_indices = sort_indices[:,-1]
        max2_indices = sort_indices[:,-2]
        max_neighbor = max_indices.copy()
        
        #for each attenuation find the ratio of the maximum velocity to the second highest velocity
        self.res1_max_ratio = self.res1_max_vels.copy()
        max_neighbors = np.zeros(self.NAttens)
        max2_neighbors = np.zeros(self.NAttens)
        self.res1_max2_ratio = self.res1_max2_vels.copy()
        for iAtt in range(self.NAttens):
            if max_indices[iAtt] == 0:
                max_neighbor = self.res1_iq_vels[iAtt,max_indices[iAtt]+1]
            elif max_indices[iAtt] == len(self.res1_iq_vels[iAtt,:])-1:
                max_neighbor = self.res1_iq_vels[iAtt,max_indices[iAtt]-1]
            else:
                max_neighbor = np.maximum(self.res1_iq_vels[iAtt,max_indices[iAtt]-1],
                                       self.res1_iq_vels[iAtt,max_indices[iAtt]+1])
            max_neighbors[iAtt]=max_neighbor
            self.res1_max_ratio[iAtt] = self.res1_max_vels[iAtt]/max_neighbor
            if max2_indices[iAtt] == 0:
                max2_neighbor = self.res1_iq_vels[iAtt,max2_indices[iAtt]+1]
            elif max2_indices[iAtt] == len(self.res1_iq_vels[iAtt,:])-1:
                max2_neighbor = self.res1_iq_vels[iAtt,max2_indices[iAtt]-1]
            else:
                max2_neighbor = np.maximum(self.res1_iq_vels[iAtt,max2_indices[iAtt]-1],
                                        self.res1_iq_vels[iAtt,max2_indices[iAtt]+1])
            max2_neighbors[iAtt]=max2_neighbor
            self.res1_max2_ratio[iAtt] = self.res1_max2_vels[iAtt]/max2_neighbor
        #normalize the new arrays
        self.res1_max_vels /= np.max(self.res1_max_vels)
        self.res1_max_vels *= np.max(self.res1_max_ratio)
        self.res1_max2_vels /= np.max(self.res1_max2_vels)

        
        max_ratio_threshold = 2.5#1.5
        rule_of_thumb_offset = 1#2

        # require ROTO adjacent elements to be all below the MRT
        bool_remove = np.ones(len(self.res1_max_ratio))
        for ri in range(len(self.res1_max_ratio)-rule_of_thumb_offset-2):
            bool_remove[ri] = bool((self.res1_max_ratio[ri:ri+rule_of_thumb_offset+1]< max_ratio_threshold).all())
        guess_atten_idx = np.extract(bool_remove,np.arange(len(self.res1_max_ratio)))

        # require the attenuation value to be past the initial peak in MRT
        guess_atten_idx = guess_atten_idx[np.where(guess_atten_idx > np.argmax(self.res1_max_ratio) )[0]]

        if np.size(guess_atten_idx) >= 1:
            if guess_atten_idx[0]+rule_of_thumb_offset < len(self.Res1.atten1s):
                guess_atten_idx[0] += rule_of_thumb_offset
                guess_atten_idx =  int(guess_atten_idx[0])
        else:
            guess_atten_idx = self.NAttens/2

        print 'file', self.initialFile
        # print 'atten1s', self.Res1.atten1s

        if useResID:            
            return {'freq': self.Res1.freq,
                    'resID': self.Res1.resID,
                    'iq_vels': self.res1_iq_vels, 
                    'Is': self.Res1.Is,     
                    'Qs': self.Res1.Qs, 
                    'attens':self.Res1.atten1s}
            

        return {'freq': self.Res1.freq, 
                'iq_vels': self.res1_iq_vels, 
                'Is': self.Res1.Is,     
                'Qs': self.Res1.Qs, 
                'attens':self.Res1.atten1s}
    
    def loadps(self):
        hd5file=openFile(self.initialFile,mode='r')
        group = hd5file.getNode('/','r0')
        self.freq=np.empty(0,dtype='float32')
        
        for sweep in group._f_walkNodes('Leaf'):
            k=sweep.read()
            self.scale = k['scale'][0]
            #print "Scale factor is ", self.scale
            self.freq=np.append(self.freq,[k['f0'][0]])
        hd5file.close()

class PSFitMLData():
    def __init__(self, h5File=None, PSFile=None, useAllAttens=True, useResID=False):
        self.useAllAttens=useAllAttens
        self.useResID=useResID
        self.h5File = h5File
        self.PSFile = PSFile
        self.PSPFile = self.h5File[:-3] + '.pkl'
        print 'h5File', self.h5File
        print 'pkl file', self.PSPFile
        self.baseFile = self.h5File[:-19]
        self.freqs, self.iq_vels,self.Is,self.Qs, self.attens, self.resIDs = self.get_PS_data()
        self.opt_attens = None
        self.opt_freqs = None

    def loadTrainData(self):                
        '''
        Loads in a PS frequency text file to use in a training set.  Populates the variables self.opt_attens and self.opt_freqs.

        '''
       
        # if mag_data==True:
        #     #self.trainFile = self.trainFile.split('.')[0]+'_mag.pkl'
        #     print self.trainFile, 'self.trainFile'
        #     self.nClass =3

        if os.path.isfile(self.PSFile):
            print 'loading peak location data from %s' % self.PSFile
            PSFile = np.loadtxt(self.PSFile, skiprows=0)
            opt_freqs = PSFile[:,1]
            opt_attens = PSFile[:,2]
            if self.useResID:
                goodResIDs = PSFile[:,0]
                self.good_res = np.where(map(lambda x: np.any(x==goodResIDs), self.resIDs))[0]
            else:
                self.good_res = np.array(PSFile[:,0]-PSFile[0,0],dtype='int')

            self.res_nums = len(self.good_res)          

            self.attens = self.attens[self.good_res,:]
            optAttenLocs = np.where(np.transpose(np.transpose(np.array(self.attens))==np.array(opt_attens))) #find optimal atten indices
            optAttenExists = optAttenLocs[0]
            self.opt_iAttens = optAttenLocs[1]

            attenSkips = optAttenLocs[0]-np.arange(len(optAttenLocs[0]))
            attenSkips = np.where(np.diff(attenSkips))[0]+1 #indices where opt atten was not found
            for resSkip in attenSkips:
                print 'resSkip', resSkip
                if(opt_attens[resSkip]<self.attens[resSkip,0]):
                    opt_attens[resSkip] = self.attens[resSkip,0]
                    self.opt_iAttens = np.insert(self.opt_iAttens,resSkip,0) 
                elif(opt_attens[resSkip]>self.attens[resSkip,-1]):
                    opt_attens[resSkip] = self.attens[resSkip,-1]
                    self.opt_iAttens = np.insert(self.opt_iAttens,resSkip,np.shape(self.attens)[1]-1)
                else:
                    raise ValueError('Atten skip index error')
        else: 
            print self.PSFile, 'not found' 
            exit()

        print 'optfreqs len', len(opt_freqs)
        print 'self.freqs len', len(self.freqs)
        self.freqs = np.around(self.freqs[self.good_res], decimals=-4)
        self.opt_freqs = np.around(opt_freqs, decimals=-4)
        self.iq_vels = self.iq_vels[self.good_res]
        self.Is = self.Is[self.good_res]
        self.Qs = self.Qs[self.good_res]
        self.resIDs = self.resIDs[self.good_res]

    def savePSTxtFile(self, flag = '', outputFN=None):
        '''
        Saves a frequency file after inference.  self.opt_attens and self.opt_freqs
        should be populated by an external ML algorithm.
        '''
        if self.opt_attens is None or self.opt_freqs is None:
            raise ValueError('Classify Resonators First!')
        
        if outputFN is None: PSSaveFile = self.baseFile + flag + '.txt'
        else: PSSaveFile = outputFN.rsplit('.',1)[0]+flag+'.txt'
        sf = open(PSSaveFile,'wb')
        print 'saving file', PSSaveFile
        print 'baseFile', self.baseFile
        #sf.write('1\t1\t1\t1 \n')
        for r in range(len(self.opt_attens)):
            line = "%4i \t %10.9e \t %4i \n" % (self.resIDs[r], self.opt_freqs[r], 
                                         self.opt_attens[r])
            sf.write(line)
            #print line
        sf.close()        

    def get_PS_data(self, searchAllRes=True, res_nums=50):
        '''A function to read and write all resonator information so stop having to run the PSFit function on all resonators 
        if running the script more than once. This is used on both the initial and inference file

        Inputs:
        h5File: the power sweep h5 file for the information to be extracted from. Can be initialFile or inferenceFile
        '''
        print 'get_PS_data_all_attens H5 file', self.h5File
        print 'resNums', res_nums
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
                print 'freqshape get_PS_data', np.shape(freqs)
                iq_vels = file[2][:]
                Is = file[3][:]
                Qs= file[4][:]
                attens = file[5]

            else:
                resIDs = file[0][:]
                freqs = file[1][:res_nums]
                print 'freqshape get_PS_data', np.shape(freqs)
                iq_vels = file[2][:res_nums]
                Is = file[3][:res_nums]
                Qs= file[4][:res_nums]
                attens = file[5]

        else:
            PSFit = PSFitting(initialFile=self.h5File)
            PSFit.loadps()
            tot_res_nums= len(PSFit.freq)
            print 'totalResNums in getPSdata', tot_res_nums
            if searchAllRes:
                res_nums = tot_res_nums

            res_size = np.shape(PSFit.loadres()['iq_vels'])

            freqs = np.zeros((res_nums, res_size[1]+1))
            iq_vels = np.zeros((res_nums, res_size[0], res_size[1]))
            Is = np.zeros((res_nums, res_size[0], res_size[1]+1))
            Qs = np.zeros((res_nums, res_size[0], res_size[1]+1))
            attens = np.zeros((res_nums, res_size[0]))
            resIDs = np.zeros(res_nums)

            for r in range(res_nums):
                sys.stdout.write("\r%d of %i " % (r+1,res_nums) )
                sys.stdout.flush()
                res = PSFit.loadres(self.useResID)
                if self.useResID:
                    resIDs[r] = res['resID']+0
                else:
                    resIDs[r] = r
                freqs[r,:] =res['freq']
                iq_vels[r,:,:] = res['iq_vels']
                Is[r,:,:] = res['Is']
                Qs[r,:,:] = res['Qs']
                attens[r,:] = res['attens']
                PSFit.resnum += 1

            with open(self.PSPFile, "wb") as f:
                pickle.dump(resIDs, f)
                pickle.dump(freqs, f)
                pickle.dump(iq_vels, f)
                pickle.dump(Is, f)
                pickle.dump(Qs, f)
                pickle.dump(attens, f)

        #print 'prekill attens', attens
        if not(self.useAllAttens):
            attens = attens[0,:]

        print 'h5 data file', self.h5File
        print 'h5 attens', attens
        return  freqs, iq_vels, Is, Qs, attens, resIDs

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

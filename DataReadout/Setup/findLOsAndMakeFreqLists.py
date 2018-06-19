'''
Author: Neelay Fruitwala

Automates finding LOs, generating frequecy files and modifying templarconf.cfg files.
Usage: python findLOsAndMakeFreqFiles.py <setupcfgfile> <templarcfgfile>
    setupcfgfile - Configuration file containing lists of feedline numbers, roach
        numbers, and WideAna generated clickthrough results. An example can be found in
        example_setup.cfg. If it can't find the file it looks in MKID_DATA_DIR.
    templarcfgfile - High templar configuration file to be modified. WARNING: file will
        be OVERWRITTEN. Changes the frequency lists, LOs, longsnap files, and powersweep files
        to point to the write locations (for the boards specified in setupcfgfile). If it can't 
        find the file it looks in MKID_DATA_DIR.

'''
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pdb
import ConfigParser
from createTemplarResList import createTemplarResList
from readDict import readDict

def findLOs(freqs, loRange=0.2, nIters=10000, colParamWeight=1, resBW=0.0002, ifHole=0.003):
    '''
    Finds the optimal LO frequencies for a feedline, given a list of resonator frequencies.
    Does Monte Carlo optimization to minimize the number of out of band tones and sideband 
    collisions.
    
    Parameters
    ----------
        freqs - list of resonator frequencies, in GHz
        loRange - size of LO search band, in GHz
        nIters - number of optimization interations
        colParamWeight - relative weighting between number of collisions and number of omitted
            tones in cost function. 1 usually gives good performance, set to 0 if you don't want
            to optimize for sideband collisions. 
        resBW - bandwidth of resonator channels. Tones are considered collisions if their difference
            is less than this value.
        ifHole - tones within this distance from LO are not counted
    Returns
    -------
        lo1, lo2 - low and high frequency LOs (in GHz)
    '''
    lfRange = np.array([freqs[0]+1, freqs[0]+1+loRange]) #range to search for LF LO; 200 MHz span
    hfRange = np.array([freqs[-1]-1-loRange, freqs[-1]-1])
    
    nCollisionsOpt = len(freqs) #number of sideband collisions
    nFreqsOmittedOpt = len(freqs) #number of frequencies outside LO band
    costOpt = nCollisionsOpt + colParamWeight*nCollisionsOpt
    for i in range(nIters):
        lo1 = np.random.rand(1)[0]*loRange + lfRange[0]
        hflolb = max(hfRange[0], lo1 + 1) #lower bound of hf sampling range; want LOs to be 1 GHz apart
        lo2 = np.random.rand(1)[0]*(hfRange[1]-hflolb) + hflolb

        #find nFreqsOmitted
        freqsIF1 = freqs - lo1
        freqsIF2 = freqs - lo2
        isInLFBand = np.logical_and(np.abs(freqsIF1)<1, np.abs(freqsIF1)>ifHole)
        isInHFBand = np.logical_and(np.abs(freqsIF2)<1, np.abs(freqsIF2)>ifHole)
        isNotValidTone = np.logical_not(np.logical_or(isInLFBand, isInHFBand))
        nFreqsOmitted = np.sum(isNotValidTone)

        #find nCollisions
        freqsIF1 = freqsIF1[np.where(isInLFBand)]
        freqsIF2 = freqsIF2[np.where(isInHFBand)]
        freqsIF1SB = np.sort(np.abs(freqsIF1))
        freqsIF2SB = np.sort(np.abs(freqsIF2))
        nLFColl = np.sum(np.diff(freqsIF1SB)<resBW)
        nHFColl = np.sum(np.diff(freqsIF2SB)<resBW)
        nCollisions = nLFColl + nHFColl

        #pdb.set_trace()

        cost = nFreqsOmitted + colParamWeight*nCollisions
        if cost<costOpt:
            costOpt = cost
            nCollisionsOpt = nCollisions
            nFreqsOmittedOpt = nFreqsOmitted
            lo1Opt = lo1
            lo2Opt = lo2
            #print 'nCollOpt', nCollisionsOpt
            #print 'nFreqsOmittedOpt', nFreqsOmittedOpt
            #print 'los', lo1, lo2

    print 'Optimal nCollisions', nCollisionsOpt
    print 'Optimal nFreqsOmitted', nFreqsOmittedOpt
    print 'LO1', lo1Opt
    print 'LO2', lo2Opt

    return lo1Opt, lo2Opt

def modifyTemplarConfigFile(templarConfFn, flNums, roachNums, freqFiles, los, freqBandFlags):
    '''
    Modifies the specified templar config file with the correct frequency lists and los. All files
    are referenced to the MKID_DATA_DIR environment variable. Also changes powersweep_file, longsnap_file, etc
    to be referenced to the correct feedline and MKID_DATA_DIR

    Parameters
    ----------
        templarConfFn - name of templar config file
        flNums - list of feedline numbers corresponding to roachNums
        roachNums - list of board numbers (last 3 digits of roach IP)
        freqFiles - list of frequency file names
        los - list of LO frequencies (in GHz)
        freqBandFlags - list of flags indicating whether board is LF or HF. 'a' for LF and 'b' for HF
    '''
    
    try:
        tcfp = open(templarConfFn, 'r')
        mdd = os.path.dirname(templarConfFn)
    except IOError:
        mdd = os.environ['MKID_DATA_DIR']
        templarConfFn = os.path.join(mdd, templarConfFn)
        tcfp = open(templarConfFn, 'r')
    templarConf = ConfigParser.ConfigParser()
    templarConf.readfp(tcfp)
    tcfp.close()
    
    for i,roachNum in enumerate(roachNums):
        templarConf.set('Roach '+str(roachNum), 'freqfile', os.path.join(mdd, freqFiles[i]))
        templarConf.set('Roach '+str(roachNum), 'powersweepfile', os.path.join(mdd, 'ps_r'+str(roachNum)+'_FL_'+str(flNums[i])+'_'+freqBandFlags[i]+'.h5'))
        templarConf.set('Roach '+str(roachNum), 'longsnapfile', os.path.join(mdd, 'phasesnaps/snap_'+str(roachNum)+'.npz'))
        templarConf.set('Roach '+str(roachNum), 'lo_freq', '%0.9E'%(los[i]*1.e9))

    tcfp = open(templarConfFn, 'w')
    templarConf.write(tcfp)
    tcfp.close()
 
def loadClickthroughFile(fn):
    '''
    Loads clickthrough results from WideAna output file. Returns list of
    ResIDs, peak locations, and frequencies (in GHz)
    '''
    if not os.path.isfile(fn):
        fn = os.path.join(mdd, fn)
    resIDs, locs, freqs = np.loadtxt(fn, unpack=True)
    return resIDs, locs, freqs



if __name__=='__main__':
    if len(sys.argv)<3:
        print 'Usage: python findLOsAndMakeFreqFiles.py <setupcfgfile> <templarcfgfile>'
    
    setupDict = readDict()
    try: 
        setupDict.readFromFile(sys.argv[1])
        try: mdd = setupDict['MKID_DATA_DIR']
        except KeyError: mdd = os.environ['MKID_DATA_DIR']
    except IOError:
        mdd = os.environ['MKID_DATA_DIR']
        setupDict.readFromFile(os.path.join(mdd, sys.argv[1]))
    #print setupDict
    templarCfgFile = sys.argv[2]
    freqFiles = []
    los = []
    flNums = []
    freqBandFlags = []
    boardNums = []

    for i, fl in enumerate(setupDict['feedline_nums']):
        clickthroughFile = os.path.join(mdd, setupDict['clickthrough_files'][i])
        _, _, freqs = loadClickthroughFile(clickthroughFile)
        lo1, lo2 = findLOs(freqs)
        createTemplarResList(clickthroughFile, lo1, lo2, fl)
        freqFiles.append('freq_FL' + str(fl) + '_a.txt')
        freqFiles.append('freq_FL' + str(fl) + '_b.txt')
        los.append(lo1)
        los.append(lo2)
        flNums.append(fl)
        flNums.append(fl)
        freqBandFlags.append('a')
        freqBandFlags.append('b')
        boardNums.append(setupDict['low_freq_boardnums'][i])
        boardNums.append(setupDict['high_freq_boardnums'][i])

    modifyTemplarConfigFile(templarCfgFile, flNums, boardNums, freqFiles, los, freqBandFlags)


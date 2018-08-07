
import numpy as np
import os
import sys


def createTemplarResList(fn, LO1, LO2, feedline=1, atten=46):
    """
    INPUTS:
        fn - filename comtinaing widesweep's list of resonators on a feedline
        LO1 - lower LO freq     [GHz]
        LO2 - higher LO freq    [GHz]
    """
    resIDs,_,freqs = np.loadtxt(fn,unpack=True)  # this is in GHz
    print len(freqs)
    
    #freqs=freqs - 0.01
    
    freqs = freqs[np.where(freqs > (LO1 - 1.))]
    freqs = freqs[np.where(freqs < (LO2 + 1.))]
    resIDs = resIDs[np.where(freqs > (LO1 - 1.))]
    resIDs = resIDs[np.where(freqs < (LO2 + 1.))]

    print len(freqs)
    
    LO_avg = (LO1+LO2) / 2.
    
    freqs1 = freqs[np.where(freqs<=LO_avg)]     # on first board
    resIDs1 = resIDs[np.where(freqs<=LO_avg)]     # on first board
    freqs2 = freqs[np.where(freqs>LO_avg)]      # on second board
    resIDs2 = resIDs[np.where(freqs>LO_avg)]      # on second board
    freqs1 = freqs1[np.where(freqs1 < LO1 + 1.)]
    resIDs1 = resIDs1[np.where(freqs1 < LO1 + 1.)]
    freqs2 = freqs2[np.where(freqs2 > LO2 - 1.)]
    resIDs2 = resIDs2[np.where(freqs2 > LO2 - 1.)]
    print len(freqs1)
    print len(freqs2)
    
    path = os.path.dirname(fn)+'/'
    data = np.transpose([resIDs1,freqs1*10**9,[atten]*len(freqs1)])
    np.savetxt(path+'freq_FL'+str(feedline)+'_a.txt', data, fmt="%6i %10.9e %4i")
    data = np.transpose([resIDs2,freqs2*10**9,[atten]*len(freqs2)])
    np.savetxt(path+'freq_FL'+str(feedline)+'_b.txt', data, fmt="%6i %10.9e %4i")
    
    
    
    
    '''
    # on single stream firmware can only do 256 resonators at a time instead of 1024
    freqs1_list = []
    num=0
    lastNum=0
    for i in range(4):
        lastNum += num
        num = np.ceil((len(freqs1)-lastNum)/(4-i))
        freqs1_i = freqs1[lastNum:lastNum+num]
        freqs1_list.append(freqs1_i)
        print len(freqs1_i)
    
    freqs2_list = []
    num=0
    lastNum=0
    for i in range(4):
        lastNum += num
        num = np.ceil((len(freqs2)-lastNum)/(4-i))
        freqs2_i = freqs1[lastNum:lastNum+num]
        freqs2_list.append(freqs2_i)
        print len(freqs2_i)
    
    path = os.path.dirname(fn)+'/'
    for i in range(len(freqs1_list)):
        data = np.transpose([freqs1_list[i]*10**9,[atten]*len(freqs1_list[i])])
        #np.savetxt(path+'freqs'+str(feedline)+'_'+str(i)+'.txt', data, fmt="%10.1f %4i")
    
    for i in range(len(freqs2_list)):
        data = np.transpose([freqs2_list[i]*10**9,[atten]*len(freqs2_list[i])])
        np.savetxt(path+'freqs2_FL'+str(feedline)+'_'+str(i)+'.txt', data, fmt="%10.1f %4i")
    '''
'''
if __name__ == '__main__':
    fl = 1
    fn = '/mnt/data0/Darkness/20160719/Ukko_Palomar_FL'+str(fl)+'-1-good.txt'
    LO1 = 5.1364336
    LO2 = 7.3371334
    createTemplarResList(fn, LO1, LO2,feedline=fl,atten=38)
'''

if __name__=='__main__':
    fn = "faceless_FL5-freqs-good.txt"
    dirn = os.environ['MKID_DATA_DIR']
    fn = os.path.join(dirn,fn)
    LO1 = 5.1613579
    LO2 = 7.2725754
    createTemplarResList(fn, LO1, LO2, feedline=5)











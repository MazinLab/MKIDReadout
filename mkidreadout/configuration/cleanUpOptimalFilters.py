'''
Script to resolve channel discrepancies in optimal filters calculated using a different configuration
'''

import numpy as np

#todo make part of an optimal filters result object

filtResListFn = '/mnt/data0/Darkness/20170403/ps_r122_FL4_b_faceless_hf_train.txt'
newResListFn = '/mnt/data0/Darkness/20170407/truncatedAttens/ps_r122_FL4_b_faceless_hf_train_rm_doubles.txt' 
optFiltFn = '/mnt/data0/Darkness/20170403/optimal_filters/122_data/filter_coefficients_20170407.txt'

filtResIDs, _, _ = np.loadtxt(filtResListFn, unpack=True)
newResIDs, _, _ = np.loadtxt(newResListFn, unpack=True)
optFiltCoeffs = np.loadtxt(optFiltFn)

rmChannelList = []

if not np.shape(optFiltCoeffs)[0]==len(filtResIDs):
    raise ValueError('Wrong optimal filter file!')

for i in range(len(filtResIDs)):
    if not np.any(filtResIDs[i]==newResIDs):
        rmChannelList.append(i)

optFiltCoeffs = np.delete(optFiltCoeffs, rmChannelList, axis=0)

print 'nChannels ', np.shape(optFiltCoeffs)[0]

np.savetxt(optFiltFn.split('.')[0]+'_new.txt', np.transpose(optFiltCoeffs))

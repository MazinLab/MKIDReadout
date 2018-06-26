import numpy as np

filterCoeffFile = '/mnt/data0/Darkness/20170403/optimal_filters/122_data/filter_coefficients_20170407_new.txt'
filter = np.transpose(np.loadtxt(filterCoeffFile))
print 'initialfiltershape', np.shape(filter)

filterMagList = np.sum(filter, axis=1)
filter = np.transpose(np.transpose(filter)/filterMagList)

print 'newfiltershape', np.shape(filter)

np.savetxt(filterCoeffFile.split('.')[0]+'_normd.txt', np.transpose(filter))

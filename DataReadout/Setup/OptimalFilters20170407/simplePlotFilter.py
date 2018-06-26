import numpy as np
import sys
import matplotlib.pyplot as plt

if __name__=='__main__':
    boardNum = int(sys.argv[1])
    channel = int(sys.argv[2])
    filterCoeffs = np.transpose(np.loadtxt('/mnt/data0/Darkness/20170403/optimal_filters/'+str(boardNum)+'_data/filter_coefficients_20170407_new_normd.txt'))
    plt.plot(filterCoeffs[channel,:])
    plt.show()

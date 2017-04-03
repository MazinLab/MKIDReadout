from sys import path
import numpy as np
import scraps as scr
import lmfit as lf
# import plot_tools as pt
import PSFitTools as pt
from an_params import saveDir, inferenceFile, cacheDir, cutoff, ml_dir
# import runMLCode as ml
# import PSFitMLData as pd
path.append(ml_dir)
# from PSFitMLData_origPSFile import *
import sys
sys.path.append('/home/rupert/PythonProjects/MkidDigitalReadout/DataReadout/Setup/modular_PowerSweep')
from PSFitMLData import *
import PSFitSc as PSFitSc

if __name__ == "__main__":
    # PowerSweep fit object
    # mlClass = train_power_NN()
    ps = PSFitSc.PSFitSc()

    # get resLists
    ps.resListsFile = cacheDir + 'resLists_' + inferenceFile + '.pkl'
    # resLists = ps.loadresLists()
    # resLists = ps.fitresLists(num_res =20)
    resLists = ps.loadresLists()

    # get analytical powers from resLists
    a_pwrs, a_ipwrs = ps.evaluatePwrsFromDetune(resLists)

    # get the manual clicked powers
    mc_mask, mc_ipwrs, mc_pwrs = pt.loadManualClickFile(ps.inferenceData,cutoff)

    # reduce the a-fits to just those that have click counterparts
    a_pwrs=np.asarray(a_pwrs)[mc_mask]
    a_ipwrs=np.asarray(a_ipwrs)[mc_mask]
    ps.nonlin_params=np.asarray(ps.nonlin_params)[mc_mask]

    # compare the values
    # pt.getMatch(mc_ipwrs, a_ipwrs)
    # pt.plotcumAccuracy(mc_pwrs, a_pwrs)
    # pt.plotPwrGuessCompMap(mc_pwrs, a_pwrs)

    # plot confusion matrix
    pt.plotIndGuessCompMap(mc_ipwrs, a_ipwrs)
    # wrong_guesses = pt.getMissed(a_ipwrs, mc_ipwrs)
    # pt.plotMissed(ps, wrong_guesses, a_ipwrs, mc_ipwrs)

 
    # plt.hist(independent, range(max_nClass+1), label='True', facecolor='blue', alpha=0.65) 
    # plt.hist(dependent, range(max_nClass+1), label='Evaluated', facecolor='green', alpha=0.65)
    # plt.legend(loc="upper left")
    # plt.show()

    # plot the comparisons
    # pt.plotAll(ps,a_ipwrs, mc_ipwrs)
    # for res in range(len(mc_ipwrs)):
    #     ps.plotResListData(resLists[res])
    #     pt.plotRes(ps, a_ipwrs, mc_ipwrs, res)
    #     plt.show()
# fit_pwrs = get_pwrs()
# ml_pwrs = get_pwrs()
# man_pwrs = pd.get_pwrs()

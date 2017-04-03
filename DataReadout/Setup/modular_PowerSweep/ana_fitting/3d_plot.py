from sys import path
import numpy as np
import scraps as scr
# import lmfit as lf
# import plot_tools as pt
# import PSFitTools as pt
from params import saveDir, inferenceFile, cacheDir, cutoff, ml_dir
# import runMLCode as ml
# import PSFitMLData as pd
path.append(ml_dir)
# from PSFitMLData_origPSFile import *
# from PSFitMLData import *
import PSFitSc as PSFitSc

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

if __name__ == "__main__":
    # PowerSweep fit object
    ps = PSFitSc.PSFitSc(saveDir=saveDir, inferenceFile=inferenceFile)

    # get resLists
    # ps.resListsFile = cacheDir + 'resLists_' + inferenceFile + '.pkl'
    # resLists = ps.loadresLists()

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # def cc(arg):
    #     return mcolors.to_rgba(arg, alpha=0.6)
    # resLists = ps.fitresLists(num_res=1)
    resLists = ps.loadresLists()
    # resList = ps.fitresList()
    resList = resLists[0]

    verts = []

    Is=[]
    Qs=[]
    z=0
    maxI, minI, maxQ, minQ, maxZ, minZ = 0, 0, 0, 0, -50, 0

    for res in resList[::2]:
        # ys = np.random.rand(len(xs))
        # ys[0], ys[-1] = 0, 0
        # verts.append(list(zip(res.I, res.Q)))
        Is.append(res.I)
        Qs.append(res.Q)
        print len(res.I)
        print res.pwr

        ax.plot(res.I, res.Q, zs=res.pwr, zdir='y', alpha=0.8)
        print z
        # if z == 24:
        #     xticklabels = ax.get_xticks()
        # ax.plot(res.I, res.Q, zs=-66, zdir='y', alpha=0.8)
        z += 2
        maxI = np.max([maxI, np.max(res.I)])
        minI = np.min([minI, np.min(res.I)])
        maxQ = np.max([maxQ, np.max(res.Q)])
        minQ = np.min([minQ, np.min(res.Q)])
        maxZ = np.max([maxZ, res.pwr])
        minZ = np.min([minZ, res.pwr])

        print maxI, minI, maxQ, minQ
    # poly = PolyCollection(verts)

    # poly.set_alpha(0.7)
    # ax.add_collection3d(poly, zs=zs, zdir='y')


    ax.set_xlabel('I')
    # ax.set_xticklabels(ax.get_xticks(), rotation=45)
    label_size = 4
    # ax.set_xticklabels(ax.get_xticks(), rotation=45)
    ax.set_xlim3d(minI, maxI)
    ax.set_ylabel('Power (dBm)')
    ax.set_ylim3d(maxZ, minZ)
    ax.set_zlabel('Q')
    ax.set_zlim3d(minQ, maxQ)
    # ax.set_facecolor('none')

    plt.show()



#     # get analytical powers from resLists
#     a_pwrs, a_ipwrs = ps.evaluatePwrsFromDetune(resLists)
#
#     # get the manual clicked powers
#     mc_mask, mc_ipwrs, mc_pwrs = pt.loadManualClickFile(ps.inferenceData,cutoff)
#
#     # reduce the a-fits to just those that have click counterparts
#     a_pwrs=np.asarray(a_pwrs)[mc_mask]
#     a_ipwrs=np.asarray(a_ipwrs)[mc_mask]
#     ps.nonlin_params=np.asarray(ps.nonlin_params)[mc_mask]
#
#     # compare the values
#     pt.getMatch(mc_ipwrs, a_ipwrs)
#     # pt.plotcumAccuracy(mc_pwrs, a_pwrs)
#     # pt.plotPwrGuessCompMap(mc_pwrs, a_pwrs)
#     # pt.plotIndGuessCompMap(mc_ipwrs, a_ipwrs)
#     # wrong_guesses = pt.getMissed(a_ipwrs, mc_ipwrs)
#     # pt.plotMissed(ps, wrong_guesses, a_ipwrs, mc_ipwrs)
#
#     # plot the comparisons
#     pt.plotAll(ps,a_ipwrs, mc_ipwrs)
#
# # fit_pwrs = get_pwrs()
# # ml_pwrs = get_pwrs()
# # man_pwrs = pd.get_pwrs()

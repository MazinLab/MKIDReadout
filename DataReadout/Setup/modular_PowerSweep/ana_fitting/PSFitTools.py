import numpy as np
import scraps as scr
from matplotlib import pylab as plt
from matplotlib import cm
import matplotlib.colors
import lmfit as lf
import cPickle as pickle

def plotPwrGuessCompMap(mc_pwrs, a_pwrs):
    min_pwr = min([min(mc_pwrs), min(a_pwrs)])
    max_pwr = max([max(mc_pwrs), max(a_pwrs)])
    mc_pwrs = mc_pwrs - min_pwr
    a_pwrs = a_pwrs - min_pwr

    span = max_pwr - min_pwr
    guesses_map = np.zeros((span + 1, span + 1))

    for ia, am in enumerate(mc_pwrs):
        ag = a_pwrs[ia]
        guesses_map[ag, am] += 1

    plt.imshow(guesses_map,
               origin='lower',
               interpolation='none',
               cmap=cm.coolwarm,
               extent=[min_pwr, max_pwr, min_pwr, max_pwr])
    plt.xlabel('manual')
    plt.ylabel('estimate')
    plt.colorbar(cmap=cm.afmhot)
    plt.show()

def conf_histogram():
    plt.figure(figsize=(6,6))
    # plt.plot(np.sum(guesses_map, axis=0), label='True')
    # plt.plot(np.sum(guesses_map, axis=1), label='Evaluated')
    plt.hist(independent, range(max_nClass+1), label='True', facecolor='blue', alpha=0.65) 
    plt.hist(dependent, range(max_nClass+1), label='Evaluated', facecolor='green', alpha=0.65)
    print np.histogram(dependent, range(max_nClass + 1))
    plt.legend(loc="upper left")
    plt.show()

def plotIndGuessCompMap(mc_ipwrs, a_ipwrs):
    min_pwr = min([min(mc_ipwrs), min(a_ipwrs)])
    max_pwr = max([max(mc_ipwrs), max(a_ipwrs)])
    # mc_pwrs = mc_pwrs - min_pwr
    # a_pwrs = a_pwrs - min_pwr
    print mc_ipwrs, a_ipwrs
    print min_pwr, max_pwr

    # span = max_pwr - min_pwr
    guesses_map = np.zeros((max_pwr + 1, max_pwr + 1))

    for ia, am in enumerate(mc_ipwrs):
        ag = a_ipwrs[ia]
        guesses_map[ag, am] += 1

    plt.imshow(guesses_map,
               origin='lower',
               interpolation='none',
               cmap=cm.coolwarm)
    # extent=[min_pwr,max_pwr,min_pwr,max_pwr])
    plt.xlabel('manual')
    plt.ylabel('estimate')
    plt.colorbar(cmap=cm.afmhot)
    plt.show()


def getMatch(original, metric, bins=[5, 3, 1, 0]):
    matches = np.zeros((len(bins), len(metric)))

    for ig, _ in enumerate(metric):
        for ib, b in enumerate(bins):
            if abs(metric[ig] - original[ig]) <= b:
                matches[ib, ig] = 1

    for ib, b in enumerate(bins):
        print 'within %s' % b, sum(matches[ib]) / len(metric)

    return matches

def loadManualClickFile(inferenceData, cutoff=-1):
    # print 'loading peak location data from %s' % MCFile
    # MCFile = np.loadtxt(MCFile, skiprows=0)

    # mc_mask = MCFile[:,0]
    # MC_freqs = MCFile[:,1]
    # mc_pwrs = MCFile[:,2]*-1

    mc_mask = inferenceData.good_res
    print inferenceData.good_res

    if cutoff != -1:
        cutoff = np.where(mc_mask <= cutoff)[0][-1]
    else:
        cutoff = len(mc_mask)
    
    mc_mask = mc_mask[:cutoff]
    mc_ipwrs = inferenceData.opt_iAttens[:cutoff]
    mc_pwrs = inferenceData.opt_attens[:cutoff]

    return mc_mask, mc_ipwrs, mc_pwrs
    # return {'mc_mask': mc_mask, 'mc_pwrs': mc_pwrs}


def getMissed(a_pwrs, mc_pwrs):
    wrong_guesses = []
    print a_pwrs, len(a_pwrs), a_pwrs[-1]
    print len(a_pwrs), len(mc_pwrs)
    for ig, _ in enumerate(a_pwrs):
        if abs(a_pwrs[ig] - mc_pwrs[ig]) > 0:
            wrong_guesses.append(ig)

    return wrong_guesses


def get_agreed(a_pwrs, mc_pwrs):
    agreed = []

    for ig, _ in enumerate(a_pwrs):
        if abs(a_pwrs[ig] - mc_pwrs[ig]) <= 1:
            agreed.append(ig)

    return agreed

def plotMissed(PSFitSc, wrong_guesses, a_ipwrs, mc_ipwrs):
    for i, wg in enumerate(wrong_guesses):
        plotRes(PSFitSc, a_ipwrs, mc_ipwrs, wg)


def plotAll(PSFitSc, a_ipwrs, mc_ipwrs):
    for res in range(len(mc_ipwrs)):
        plotRes(PSFitSc, a_ipwrs, mc_ipwrs, res)


def plotRes(PSFitSc, a_ipwrs, mc_ipwrs, res):
    fig, ax1 = plt.subplots()
    ax1.set_title(res)

    ax1.axvline(a_ipwrs[res], color='b', linestyle='--', label='model')
    ax1.axvline(mc_ipwrs[res], color='k', linestyle='-.', label='human')
    ax1.set_xlabel('Atten index')
    ax1.set_ylabel('Scores and 2nd vel')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    plt.legend()

    ax2 = ax1.twinx()
    ax2.plot(PSFitSc.nonlin_params[res], color='r', label='a')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    ax2.set_xlim((-2, len(PSFitSc.nonlin_params[res]) + 3))

    plt.show()

def checkpoint_reduced(resLists, checkpoint_name):
    with open(checkpoint_name, 'wb') as cn:
        pickle.dump(resLists, cn)
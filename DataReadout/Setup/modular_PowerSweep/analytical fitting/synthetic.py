'''
Code for generating atrificial resonators
'''

from sys import path
import numpy as np
import os
import scraps as scr
import lmfit as lf
import PSFitTools as pt
from an_params import *
path.append(ml_dir)
import PSFitSc as PSFitSc
from random import choice
import matplotlib.pyplot as plt

from pylab import *
from scipy.optimize import curve_fit
import numpy.polynomial.polynomial as poly
import pickle

def get_fit_params(resLists):
    '''Make a matrix of all the fit parameters for all resonators in resLists'''
    no_params = len(resLists[0][0].lmfit_vals)
    all_params = np.zeros((len(resLists),len(resLists[0]),no_params))

    for irl, resList in enumerate(resLists):
        params = np.zeros((len(resList), no_params))
        for ir, res in enumerate(resList):
            for p in range(no_params):
                params[ir,p] = res.lmfit_vals[p]

        # params = params/np.max(params, axis=0)
        all_params[irl] = params

    return all_params

def get_resList_scale(resList, out_nAttens=15):
    '''Calculates the scale of increase in a and I for a resonator with power

    Resonators are scaled based on a quadratic I increase. a quadratic increase in vIQ should do the trick also'''

    Is_min = np.zeros((len(resList)))
    vI = np.zeros((len(resList) - 1))
    a_s = np.zeros((len(resList)))

    # for i in range(1, len(resList)-1):
    #     vI[i-1] = resList[i].I[0] - resList[i - 1].I[0]

    for i, res in enumerate(resList):
        a_s[i] = res.lmfit_vals[8]
        Is_min[i] = np.min(res.I)
        # vI[i-1] = resList[i].I[0] - resList[i - 1].I[0]

    Is_min = Is_min / np.max(Is_min)
    # vI = vI / np.max(vI)

    x = np.arange(len(a_s))
    # vI_coefs = poly.polyfit(x[:-1], vI, 2)
    # vI_scale = poly.polyval(np.linspace(0,len(a_s), nAttens),vI_coefs)

    I_coefs = poly.polyfit(x, Is_min, 2)

    def a_func(x, a, c, d):
        return a*np.exp(-c*x)+d

    a_coeff, _ = curve_fit(a_func, x, a_s, p0=(1, 1e-6, 1))

    a_scale = a_func(x, *a_coeff)
    # plt.plot(x, a_scale)

    I_scale = poly.polyval(np.linspace(0,len(x)-1,out_nAttens),I_coefs)
    a_scale = a_func(np.linspace(0,len(x)-1,out_nAttens), *a_coeff)
    # plt.plot(np.linspace(0,len(x)-1,out_nAttens), a_scale)
    # plt.show()
    return I_scale, a_scale #vI_scale,

def get_res_params(resLists, r=2, p= 0):
    return resLists[r][p].lmfit_vals

def plot_allparams(data):
    titles = ['df', 'f0', 'qc', 'qi', 'gain0', 'gain1', 'pgain0', 'pgain1', 'a']
    for p in range(len(titles)):
        plt.title(titles[p])
        for irl in range(len(data)):
            plt.plot(data[irl, :, p])
            plt.plot(data[irl, :, p])

        plt.show()

    # for d, _ in enumerate(data):
    #     # plt.hist(data[d], bins='auto')
    #     hist, bins = np.histogram(data[d])
    #     plt.plot(bins[:-1], hist, 'o')
    #     plt.title(titles[d])
    #     plt.show()

def select_params(data, good_res=None):
    params = np.zeros((np.shape(data)[2]))

    if good_res != None:
        good_res = good_res-1
        # print np.shape(data), len(good_res)
        data = data[good_res]
        # print np.shape(data)

    res = choice(range(len(data)))#3
    # print data[res,0,2], data[res,0,2]< 4e4, data[res,0,3], data[res,0,3]< 4e4,
    # print 1./(1./data[res,0,2] + 1./data[res,0,3])
    while data[res,0,3]< 4e4:
        print data[res,0,2], data[res,0,2]< 4e4, data[res,0,3], data[res,0,3]< 4e4
        res = choice(range(len(data)))
    # atten = choice(range(np.shape(data)[1]))
    atten = choice([0,1,2])

    for p, p_val in enumerate(data[res,atten]):
        params[p] = p_val
        # params.append(param[3])

    return params

def make_res(IData, QData, freqData, pwr):

    resName = 'RES'
    temp = 100
    dataDict = {'name':resName,'temp':temp,'pwr':pwr*-1,'freq':freqData,'I':IData,'Q':QData}
    res = scr.makeResFromData(dataDict)

    return res

def load_reduced_list():
    checkpoint_name = cacheDir + 'reduced_resLists_' + inferenceFile + '.pkl'
    if not os.path.isfile(checkpoint_name):
        # get resLists
        ps.resListsFile = cacheDir + 'resLists_' + inferenceFile + '.pkl'
        resLists = ps.loadresLists()

        # sanity check
        print np.shape(resLists)

        data = get_fit_params(resLists)
        hist, bins = np.histogram(data[2])
        plt.plot(bins[:-1], hist, 'o')
        plt.show()

        # get analytical powers from resLists
        a_pwrs, a_ipwrs = ps.evaluatePwrsFromDetune(resLists)

        # get the manual clicked powers
        mc_mask, mc_ipwrs, mc_pwrs = pt.loadManualClickFile(ps.inferenceData, cutoff)

        # reduce the a-fits to just those that have click counterparts
        a_pwrs = np.asarray(a_pwrs)[mc_mask]
        a_ipwrs = np.asarray(a_ipwrs)[mc_mask]
        ps.nonlin_params = np.asarray(ps.nonlin_params)[mc_mask]

        # get what are presumably the good resonators
        agreements = pt.get_agreed(a_ipwrs, mc_ipwrs)

        # reduce the data
        print np.shape(resLists)
        resLists = np.asarray(resLists)
        resLists = resLists[agreements]
        print np.shape(resLists)

        # save the good resonator data for later
        checkpoint_name = cacheDir + 'reduced_resLists_' + inferenceFile + '.pkl'
        pt.checkpoint_reduced(resLists, checkpoint_name)
    #
    # else:

def make_resList(data, good_res):

    # get random value for each param
    paramsVec = select_params(data, good_res)
    # print paramsVec

    start = choice(range(pwr_start-10, pwr_start+10))
    pwrs = np.arange(start,start+nAttens,1)

    freqs = np.linspace(paramsVec[1]-wind_band_left,paramsVec[1]+wind_band_left, freq_samples)

    resList = []
    for r in range(nAttens):
        paramsVec[-1] = a_scale[r]
        # paramsVec[5] = paramsVec[5] * I_scale[r]
        cmplxResult = scr.cmplxIQ_fit(paramsVec, freqs)

        I, Q = np.split(cmplxResult, 2)
        # I[r] = I[r] + I[r - 1] * vI_scale[r]
        # Q[r] = Q[r] + Q[r - 1] * vI_scale[r]

        I = I * I_scale[r]
        Q = Q * I_scale[r]

        res = make_res(I, Q, freqs, pwrs[r])
        resList.append(res)

    # ps.plotResListData(resList, plot_fits=False)
    # plt.show()
    return resList, freqs

def save_resLists(flag='_synthetic'):
    '''pickle the resLists in Scraps form'''
    # remove '.pkl'
    resListsFile = ps.resListsFile[:-4]
    resListsFile = resListsFile + flag + '.pkl'
    with open(resListsFile, 'wb') as rlf:
        pickle.dump(resLists, rlf)

# def calculate_vIQ(resLists):


def save_raw_train(flag='_synthetic'):
    '''pickle the resLists in numpy arrays for NN to read'''
    # remove '.pkl'
    PSFile = saveDir+inferenceFile[:-16]
    print PSFile
    PSFile = PSFile+ flag + '.pkl'
    print PSFile

    res_nums = len(resLists)
    print res_nums
    num_attens = len(resLists[0])
    print num_attens
    num_freqs = np.shape(res_freqs)[1]#len(resLists[0][0].freqs)
    print num_freqs
    freqs = np.zeros((res_nums, num_freqs))
    vIQs = np.zeros((res_nums,num_attens,num_freqs-1))
    Is = np.zeros((res_nums,num_attens,num_freqs))
    Qs = np.zeros((res_nums,num_attens,num_freqs))
    attens = np.zeros((res_nums,num_attens))
    # resIDs = np.zeros((res_nums))

    resIDs = range(res_nums)

    for ir, resList in enumerate(resLists):
        freqs[ir] = res_freqs[ir]#resList[0].freqs
        for ia, res in enumerate(resList):
            Is[ir, ia] = res.I
            Qs[ir, ia] = res.Q
            attens[ir, ia] = res.pwr
            for f in range(1,num_freqs):
                vIQs[ir,ia,f-1] = np.sqrt((Is[ir,ia,f]-Is[ir,ia,f-1])**2 + (Qs[ir,ia,f]-Qs[ir,ia,f-1])**2)

    with open(PSFile, 'wb') as f:
        pickle.dump(resIDs,f)
        pickle.dump(freqs, f)
        pickle.dump(vIQs, f)
        pickle.dump(Is, f)
        pickle.dump(Qs, f)
        pickle.dump(attens, f)


    for ia in range(num_attens):
        plt.plot(Is[2,ia],Qs[2,ia])
        plt.show()
        plt.plot(vIQs[2,ia])
        plt.show()

def perturb_vals(var):
    return var

if __name__ == "__main__":

    # PowerSweep fit object
    ps = PSFitSc.PSFitSc()

    # kludge for now since under sampled res1 a can't describe the correct model
    first_res = cacheDir + 'just_first_resLists_' + inferenceFile + '.pkl'
    ps.resListsFile = first_res
    resLists = ps.loadresLists()
    I_scale, a_scale = get_resList_scale(resLists[0], out_nAttens=15)

    ps.resListsFile = cacheDir + 'resLists_' + inferenceFile + '.pkl'
    resLists = ps.loadresLists()

    # get list of all the params
    data = get_fit_params(resLists)

    # sanity check again
    # plot_allparams(data)

    resLists = []
    res_freqs = []
    for r in range(num_fakes):
        res_data = make_resList(data, good_res)
        resLists.append(res_data[0])
        res_freqs.append(res_data[1])

    # save_resLists()
    save_raw_train()

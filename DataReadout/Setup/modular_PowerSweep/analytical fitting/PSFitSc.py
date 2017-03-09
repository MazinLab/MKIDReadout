'''
Script with useful tools for fitting power sweep with Scraps
'''

import sys

import numpy as np
import scraps as scr
from matplotlib import pylab as plt
import lmfit as lf
from an_params import *

# sys.path.append(ml_dir)
sys.path.append('/Data/PythonProjects/MkidDigitalReadout/MkidDigitalReadout/DataReadout/Setup/modular_PowerSweep/')
for p in sys.path: print p
from PSFitMLData import *

class PSFitSc():
    # def __init__(self, saveDir = 'ExampleData/PowerSweep/', inferenceFile = 'ps_r7_100mK_a_20161016-155917'):
    def __init__(self):
        # All data cleaning is done using the original pipeline
        self.inferenceData = PSFitMLData(h5File = saveDir+inferenceFile+'.h5', useAllAttens = True)
        self.inferenceData.loadRawTrainData()

        # resList = self.fitresList(r = 110)
        # plotResListData(resList)
        # resSweep = self.fitresSweep(resList)
        # plotresSweepParamsVsPwr(resSweep)

        cacheDir = saveDir + 'cache/' # used to tf_meta_data
        self.resListsFile = cacheDir + 'resLists_' + inferenceFile + '.pkl'
        self.resSweepsFile = cacheDir + 'resSweeps_' + inferenceFile + '.pkl'

        self.max_a = 5
        self.force_a_trend = force_a_trend
        self.prev_a= self.max_a
        self.prev_ad = self.max_a

    def fitresList(self, r=0):
        # create a resList object (same res many powers) from the input data
        fileDataDicts=[]
        for i,pwr in enumerate(self.inferenceData.attens[r][::4]):
            resName = 'RES'
            temp = 100
            freqData = self.inferenceData.freqs[r]
            IData = self.inferenceData.Is[r,i]
            QData = self.inferenceData.Qs[r,i]
            dataDict = {'name':resName+'_%i'%r,'temp':temp,'pwr':pwr*-1,'freq':freqData,'I':IData,'Q':QData}
            fileDataDicts.append(dataDict)

        resList = [scr.makeResFromData(fileDataDict) for fileDataDict in fileDataDicts]

        for res in resList:
            self.do_fit(res)

        # reset prev_a after it was modified in do_fit (if it was)
        self.prev_a = self.max_a

        return resList

    def comp_double(self, r=0):
        # same as fitresList except runs both do_fit and do_fit_double
        fileDataDicts = []
        for i, pwr in enumerate(self.inferenceData.attens[r][::4]):
            resName = 'RES'
            temp = 100
            freqData = self.inferenceData.freqs[r]
            IData = self.inferenceData.Is[r, i]
            QData = self.inferenceData.Qs[r, i]

            dataDict = {'name': resName + '_%i' % r, 'temp': temp, 'pwr': pwr * -1, 'freq': freqData, 'I': IData,
                        'Q': QData}
            fileDataDicts.append(dataDict)

        resList = [scr.makeResFromData(fileDataDict) for fileDataDict in fileDataDicts]
        resList2 = [scr.makeResFromData(fileDataDict) for fileDataDict in fileDataDicts]

        chi2_singles = []
        chi2_doubles = []

        # for res in resList:
        #     chi2_singles.append(self.do_fit(res))
        # self.prev_a = self.max_a

        for res2 in resList2:
            chi2_doubles.append(self.do_fit_double(res2))
        self.prev_ad = self.max_a

        # sanity check
        # plt.plot(chi2_singles, label='single')
        # plt.plot(chi2_doubles, label='double')
        # plt.legend()
        # plt.show()

        # if sum(chi2_singles) < sum(chi2_doubles):
        #     return resList
        # else:
        #     return resList2
        return resList2

    def do_fit(self, res):

        # force_a_trend forcing the nonliniearity param a to be basically always decreasing with increasing atten
        if self.force_a_trend != None:
            max_a = self.prev_a * self.force_a_trend
        else:
            max_a = self.max_a
        kwargs = {'max_a': max_a}
        res.load_params(scr.cmplxIQ_params, **kwargs)

        # kwargs = {'maxfev': 10000} # useful for speeding things up
        # res.do_lmfit(scr.cmplxIQ_fit, **kwargs)
        res.do_lmfit(scr.cmplxIQ_fit)

        self.prev_a = res.lmfit_vals[-1]

        residual = sum(res.residualI ** 2 + res.residualQ ** 2)
        return residual

    def do_fit_double(self, res):
        # if residual > 10000:
        # max_a = self.prev_a * 3
        # max_ad = self.prev_ad * 3
        max_a = self.max_a
        max_ad = self.max_a
        kwargs = {'max_a': max_a, 'max_ad': max_ad}
        print 'Fitting for two peaks'
        res.load_params(scr.cmplxIQ_params_cols, **kwargs)
        # kwargs={'maxfev':2000}
        res.do_lmfit(scr.cmplxIQ_fit_cols)

        a_loc = res.lmfit_labels.index('a')
        self.prev_a = res.lmfit_vals[a_loc]

        ad_loc = res.lmfit_labels.index('ad')
        self.prev_ad = res.lmfit_vals[ad_loc]

        print 'prev_a', self.prev_a
        print 'prev_ad', self.prev_ad

        residual = sum(res.residualI ** 2 + res.residualQ ** 2)
        print residual
        return residual

    def fitres(self, r=0, p=0):
        fileDataDicts=[]

        pwr = self.inferenceData.attens[r][p]
        resName = 'RES'
        temp = 100
        freqData = self.inferenceData.freqs[r]
        IData = self.inferenceData.Is[r,p]
        QData = self.inferenceData.Qs[r,p]
        
        dataDict = {'name':resName+'_%i'%r,'temp':temp,'pwr':pwr*-1,'freq':freqData,'I':IData,'Q':QData}
        fileDataDicts.append(dataDict)

        resList = [scr.makeResFromData(fileDataDict) for fileDataDict in fileDataDicts]

        for res in resList:
            self.do_fit(res)

        self.prev_a = self.max_a
        return resList

    def fitresLists(self, num_res = -1):
        resLists=[]
        if num_res == -1:
            num_res = np.shape(self.inferenceData.freqs)[0]
            print 'This may take a while'
        
        resonators = range(num_res)

        for r in resonators:
            sys.stdout.write('\rfitting res: %i of %i' % (r, num_res))
            sys.stdout.flush()

            resList = self.fitresList(r=r)
            
            resLists.append(resList)

        if self.resListsFile!= None:
            with open(self.resListsFile, 'wb') as cf:
                pickle.dump(resLists, cf)   

        return resLists

    def loadresLists(self):
        print 'resListsfile exists', self.resListsFile
        with open(self.resListsFile, 'rb') as cf:
            resLists = pickle.load(cf)
        return resLists

    def fitresSweep(self, resList):
        resSweep = scr.ResonatorSweep(resList, index='block')
        return resSweep

    def fitresSweeps(self, resLists):
        resSweeps = []
        for r, resList in enumerate(resLists):
            resSweep = scr.ResonatorSweep(resList, index='block')
            resSweeps.append(resSweep)

        if self.resSweepsFile != None:
            with open(self.resSweepsFile, 'wb') as cf:
                pickle.dump(resSweeps, cf)  

        return resSweeps

    def loadresSweeps(self):
        print 'resSweepsfile exists', self.resSweepsFile
        with open(self.resSweepsFile, 'rb') as cf:
            resSweeps = pickle.load(cf)
        return resSweeps

    def fit_a(self, resSweep):
        a = np.asarray(resSweep['a'])

        def fcn2min(params, x, data):
            A = params['A'].value
    #         model = A*math.exp(l*x)
            model = (A*(x + B)) ** 2
            return model - data

        a_params = lf.Parameters()
        a_params.add('A', value=-0.1)#, min=-1e9, max=0)
        a_params.add('B', value=resSweep.pvec, min=resSweep.pvec * 0.95, max=esSweep.pvec * 1.05)

        # lin_regime = 12
        # minner = lf.Minimizer(fcn2min, a_params, fcn_args=(resSweep.pvec[:lin_regime], f0s[:lin_regime]))
        # kws = {'options': {'max_iter': 100}}
        # result = minner.minimize()
        # m, c = np.asarray(result.params)
        #
        # fit = resSweep.pvec * m + c
        #
        # # lf.report_fit(result)
        #
        # diff = fit - f0s
        return a


    # def evaluate_pwrs(self, resSweeps):
    #     qis_guess = []
    #     q0s_guess = []
    #     chis_guess =[]
    #     pwrsList =[]
    #
    #     for resSweep in resSweeps:
    #         # qi = np.asarray(resSweep['qi'].idxmax(axis=1))
    #         # q0 = np.asarray(resSweep['q0'].idxmax(axis=1))
    #         # chi = np.asarray(resSweep['redchi'].idxmax(axis=1))
    #
    #         # qis_guess.append(qi)
    #         # q0s_guess.append(q0)
    #         # chis_guess.append(chi)
    #
    #         f0s = np.asarray(resSweep['f0'])[0]
    #         diff_f0s = (f0s[0]-f0s)/f0s[0]
    #
    #         def fcn2min(params,x,data):
    #             m = params['m'].value
    #             c = params['c'].value
    #             model = m*x + c
    #             return model - data
    #
    #         f0_params = lf.Parameters()
    #         f0_params.add('m', value = -1e6, min = -1e9, max = 0)
    #
    #         f0_guess =resSweep['f0'].iloc[0,0]
    #         f0_params.add('c', value = f0_guess, min = f0_guess*0.95, max = f0_guess*1.05)
    #
    #         lin_regime = 12
    #         minner =lf.Minimizer(fcn2min,f0_params,fcn_args=(resSweep.pvec[:lin_regime], f0s[:lin_regime]))
    #         kws = {'options': {'max_iter':100}}
    #         result = minner.minimize()
    #         m,c = np.asarray(result.params)
    #
    #         fit = resSweep.pvec* m +c
    #
    #         # lf.report_fit(result)
    #
    #         diff = fit-f0s
    #         # print np.where(diff[lin_regime:]>2e4)[0][0]-resSweep[0] + lin_regime
    #         # f0_guess = resSweep.pvec[np.where(diff>2e4)[0][0]]
    #
    #         try:
    #             # f0_guess = np.where(diff[lin_regime:]>2e4)[0][0]-resSweep[0] + lin_regime
    #             f0_guess = resSweep.pvec[np.where(diff[lin_regime:]>2e4)[0][0]+ lin_regime]
    #         except:
    #             f0_guess = resSweep.pvec[0] #0
    #
    #         pwrsList.append(f0_guess)
    #
    #     return pwrsList

    def evaluatePwrsFromDetune(self, resLists):
        print 'running evaluatePwrsFromDetune'
        pwrsList = []
        ipwrsList = []
        print len(resLists),len(resLists[0])
        self.nonlin_params = np.zeros((len(resLists),len(resLists[0])))
        for irl, resList in enumerate(resLists):
            resList_pwrs=[]
            # nonlin_params = np.zeros((len(resList)))
            # self.plotResListData(resList)
            for ir, res in enumerate(resList):
                # print res.lmfit_vals[-1]
                self.nonlin_params[irl,ir] = np.abs(res.lmfit_vals[-1]) #, self.lmfit_labels
                resList_pwrs.append(res.pwr)
            # print nonlin_params
                # print self.nonlin_params[irl]
            try:

                pwr = np.where(self.nonlin_params[irl]<0.5)[0][0]
                # print pwr
            except:
                # pwr = len(resLists[0])-1
                pwr = argmin(self.nonlin_params[irl])
            # print irl, self.nonlin_params[irl], pwr

            # plt.figure(figsize=(8,3))
            # cm = plt.get_cmap('rainbow_r')
            # ax = plt.subplot(111)
            # no_points = len(resList_pwrs)
            # ax.set_color_cycle([cm(1.*i/(no_points-1)) for i in range(no_points-1)])
            # for i in range(no_points-1):
            #     ax.plot(resList_pwrs[i:i+2], self.nonlin_params[irl][i:i+2])
            # plt.axvline(resList_pwrs[pwr], color = 'k', linestyle='--')
            # plt.show()
            pwrsList.append(resList[pwr].pwr)
            ipwrsList.append(pwr)

            # plt.plot(range(len(nonlin_params)), nonlin_params)
            # plt.axvline(pwr)

            # plotResListData(resList)
        # print pwrsList
        return pwrsList, ipwrsList


    # def loadManualClickFile(self, MCFile, cutoff=-1):
    #     print 'loading peak location data from %s' % MCFile
    #     MCFile = np.loadtxt(MCFile, skiprows=0)

    #     mc_mask = MCFile[:,0]
    #     MC_freqs = MCFile[:,1]
    #     mc_pwrs = MCFile[:,2]*-1

    #     if cutoff != 1:
    #         cutoff = np.where(mc_mask<=cutoff)[0][-1]
    #     # cutoff = -1
    #     print cutoff
    #     mc_mask = map(int,mc_mask)[:cutoff]
    #     mc_pwrs = mc_pwrs[:cutoff]

    #     # q0s_guess = np.asarray(q0s_guess)[good_mask]
    #     # chis_guess = np.asarray(chis_guess)[good_mask]
    #     # qis_guess = np.asarray(qis_guess)[good_mask]
    #     # print len(q0s_guess)
    #     return {'mc_mask': mc_mask, 'mc_pwrs': mc_pwrs}

    # def getMissed(a_pwrs, mc_pwrs):
    #     wrong_guesses=[]
    #     for ig, _ in enumerate(a_pwrs):            
    #         if abs(a_pwrs[ig]-mc_pwrs[ig]) >0:
    #             wrong_guesses.append(ig)

    #     return wrong_guesses 

    def plotResListData(self, resList, plot_types=None, plot_fits=True):
        if plot_types == None:
            plot_types= ['IQ',
                        'LogMag']
        if plot_fits:
            plot_fits = [True] * np.ones((len(plot_types)))
        else:
            plot_fits = [False] * np.ones((len(plot_types)))

        scr.plotResListData(resList,
                                    plot_types=plot_types,
                                    color_by='pwrs',
                                    fig_size=4,
                                    num_cols = 2,
                                    force_square=True,
                                    plot_fits = plot_fits) #<-- change this to true to overplot the best fits

        # plt.show()
     
    def plotresSweepParamsVsPwr(self,resSweep):
        scr.plotResSweepParamsVsPwr(resSweep,
                                    num_cols=3,
                                plot_keys=['f0','df', 'qi','qc','q0', 'redchi', 'a'])#, 'a', 'f02','df2', 'qi2','qc2', 'a2'])


        plt.show() 

    def plotAnaFitEsimates(self, resSweep, f0s, diff, a_pwr):
        plt.plot(resSweep.pvec, f0s, 'k--')
        plt.plot(resSweep.pvec, fit, 'r-')

        plt.plot(resSweep.pvec, diff)

        try:
            plt.axvline(a_pwr)
        except:
            pass
        plt.show()  



if __name__ == "__main__":
    # saveDir = 'ExampleData/PowerSweep/'
    # inferenceFile = 'ps_r7_100mK_a_20161016-155917'
    # ps = PSFitSc(saveDir = saveDir, inferenceFile = inferenceFile)
    # resSweeps = ps.resSweeps
    #
    # # a_pwrs = ps.evaluate_pwrs(resSweeps)
    #
    # # MCFile = saveDir+inferenceFile[:-16] + '.txt' # manual click
    # # manClickFile = em.loadManualClickFile(MCFile)
    # # mc_mask = manClickFile['mc_mask']
    # # mc_pwrs = manClickFile['mc_pwrs']
    #
    # # a_pwrs=np.asarray(a_pwrs)[mc_mask]
    #
    # # em.plotPwrGuessCompMap(mc_pwrs, a_pwrs)
    # # em.plotcumAccuracy(mc_pwrs, a_pwrs)

    # PowerSweep fit object
    ps = PSFitSc()

    resLists = ps.fitresLists(num_res=50)
    # resLists = ps.loadresLists()
    for resList in resLists:
        ps.plotResListData(resList)

    plt.show()
    for r in range(3, 9):
        print r
        # resList = ps.comp_double(r=r)
        resList = ps.fitresList(r=r)

        for res in resList:
            print res.lmfit_vals
        ps.plotResListData(resList)
        resSweep = ps.fitresSweep(resList)
        ps.plotresSweepParamsVsPwr(resSweep)

    # ps.fit_as_double(resList)


    ps.evaluatePwrsFromDetune([resList])



    # get resLists
    # ps.resListsFile = cacheDir + 'resLists_' + inferenceFile + '.pkl'
    # resLists = ps.loadresLists()

    # get analytical powers from resLists
    # a_pwrs, a_ipwrs = ps.evaluatePwrsFromDetune(resLists)
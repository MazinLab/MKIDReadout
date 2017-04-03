'''
Script with useful tools for fitting power sweep with Scraps
'''
from __future__ import division 
import sys

import matplotlib
# matplotlib.use('QT4Agg')
import numpy as np
import scraps as scr

from matplotlib import pylab as plt

import lmfit as lf
from an_params import *
import synthetic as sn

# sys.path.append(ml_dir)
sys.path.append('/home/rupert/PythonProjects/MkidDigitalReadout/DataReadout/Setup/modular_PowerSweep')
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

        # cacheDir = saveDir + 'cache/' # used to tf_meta_data
        self.resListsFile = cacheDir + 'resLists_' + inferenceFile + '.pkl'
        self.resSweepsFile = cacheDir + 'resSweeps_' + inferenceFile + '.pkl'

        # self.min_a = 0
        # self.force_a_trend = force_a_trend
        # self.prev_a= self.min_a
        # self.prev_ad = self.max_a

    def fit_with_emcee(self, res):
        import pprint as pp
        import pygtc
        # for res in resList:
        res.do_emcee(scr.cmplxIQ_fit, nwalkers=30, steps = 1000, burn=200)
        print res.hasChain

        chains = res.emcee_result.flatchain

        print '\nHead of chains:'
        pp.pprint(chains.head())

        diffs = zip(res.mle_labels, (res.mle_vals - res.lmfit_vals) * 100 / res.lmfit_vals)

        print '\nPercent difference:'

        pp.pprint(diffs)

        cmplxResult = scr.cmplxIQ_fit(res.mle_vals, res.freq)
        I, Q = np.split(cmplxResult, 2)
        res2 = sn.make_res(I, Q, res.freq, res.pwr+1)

        cmplxResult = scr.cmplxIQ_fit(res.lmfit_vals, res.freq)
        I, Q = np.split(cmplxResult, 2)
        res3 = sn.make_res(I, Q, res.freq, res.pwr+2)

        ps.plotResListData([res,res2,res3], plot_fits=False)
        plt.show()


        try:
            figGTC = pygtc.plotGTC(chains, truths=[res.lmfit_vals])
        except RuntimeError:
            pass
        plt.show()


    def fitresList(self, r=0):
        # create a resList object (same res many powers) from the input data
        fileDataDicts=[]
        # for i,pwr in enumerate(self.inferenceData.attens[r]):
        # for i, pwr in zip(range(len(self.inferenceData.attens[r]) -1,0, -1), self.inferenceData.attens[r]):
        # for i in range(len(self.inferenceData.attens[r])-1,-1, -1):    
        for i in range(nAttens):
            resName = 'RES'
            temp = 100
            freqData = self.inferenceData.freqs[r]
            IData = self.inferenceData.Is[r,i]
            QData = self.inferenceData.Qs[r,i]
            pwr = self.inferenceData.attens[r,i]
            dataDict = {'name':resName+'_%i'%r,'temp':temp,'pwr':pwr,'freq':freqData,'I':IData,'Q':QData}
            fileDataDicts.append(dataDict)

        resList = [scr.makeResFromData(fileDataDict) for fileDataDict in fileDataDicts]

        for res in resList:
            self.do_fit(res)

        # reset prev_a after it was modified in do_fit (if it was)
        # self.prev_a = self.min_a

        return resList

    # def check_double(self, r=18):
    #     for i in [len(self.inferenceData.attens[r])-1]:

    #         resName = 'RES'
    #         temp = 100
    #         freqData = self.inferenceData.freqs[r]
    #         IData = self.inferenceData.Is[r, i]
    #         QData = self.inferenceData.Qs[r, i]
    #         pwr = self.inferenceData.attens[r, i]

    #         print i, pwr
    #     plt.plot(IData, QData, '-o')
    #     plt.show()

    #     def intersect(a, b):
    #         """ return the intersection of two lists """
    #         return list(set(a) & set(b))
    #     intersection = np.zeros((len(IData)-1, len(IData)-1))
    #     for c in range(0, len(IData)-1):#range(30,70):
    #         for s in range(c, len(IData)-1):#range(30,70):
    #             a = np.asarray([IData[c], IData[c+1]])
    #             b = np.asarray([IData[s], IData[s+1]])      
    #             Icross = intersect(a,b)
    #             print Icross
    #             print c, s, a, b,
    #             a = np.asarray([QData[c], QData[c+1]])
    #             b = np.asarray([QData[s], QData[s+1]])
    #             Qcross = intersect(a,b)
    #             print a, b, Icross, Qcross, Icross and Qcross
    #             # if np.any(a)!=np.any(b):
    #             # if c != s:
    #             # if np.all(a==b) == False:
    #             #     intersection[c,s] = Icross and Qcross

    def check_double(self,r=18):
        for i in [len(self.inferenceData.attens[r])-1]:

            resName = 'RES'
            temp = 100
            freqData = self.inferenceData.freqs[r]
            IData = self.inferenceData.Is[r, i]
            QData = self.inferenceData.Qs[r, i]
            pwr = self.inferenceData.attens[r, i]

            print i, pwr
        plt.plot(IData, QData, '-o')
        plt.show()
        intersection = np.zeros((len(IData)-1, len(IData)-1))
        for c in range(0, len(IData)-1):#range(30,70):
            for s in range(c, len(IData)-1):#range(30,70):
                # # calculate center
                # print c, s,[IData[c], IData[c+1]], [QData[c], QData[c+1]], [Icenter, Qcenter]
                a = np.asarray([IData[c], IData[c+1]])
                b = np.asarray([IData[s], IData[s+1]])
                difference=a-b
                Icross=(np.sign(difference*np.roll(difference,1))<1)[1:]
                print c, s, a, b,
                a = np.asarray([QData[c], QData[c+1]])
                b = np.asarray([QData[s], QData[s+1]])
                difference=a-b
                Qcross=(np.sign(difference*np.roll(difference,1))<1)[1:]
                print a, b, Icross, Qcross, Icross and Qcross
                # if np.any(a)!=np.any(b):
                # if c != s:
                if np.all(a==b) == False:
                    intersection[c,s] = Icross and Qcross

        print intersection
        intersections = np.where(intersection == True)
        print intersections, intersections[0], np.shape(intersections)
        print intersections[1] - intersections[0]

        real_loc = np.where(intersections[1] - intersections[0] != 1)[0][0]
        intersections = [intersections[0][real_loc], intersections[1][real_loc]]
        print intersections

    def comp_double(self, r=0):
        # same as fitresList except runs both do_fit and do_fit_double
        # 7178608.2227 30905.2820391 818044.178344 40311.652302
        fileDataDicts = []

        # for i in range(len(self.inferenceData.attens[r]))[::4]:
        # for i in [len(self.inferenceData.attens[r])-1]:
        # for i in range(len(self.inferenceData.attens[r])):
        for i in range(nAttens):    
            resName = 'RES'
            temp = 100
            freqData = self.inferenceData.freqs[r]
            IData = self.inferenceData.Is[r, i]
            QData = self.inferenceData.Qs[r, i]
            pwr = self.inferenceData.attens[r, i]
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
        # self.prev_ad = self.max_a

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

    def fit_double_split(self, r=18):
        '''fit collision by splitting into two windows'''
        fileDataDicts = []
        # for i, pwr in enumerate(self.inferenceData.attens[r]):
        r = 18
        for i in [len(self.inferenceData.attens[r])-1]:
            resName = 'RES'
            temp = 100
            width = len(self.inferenceData.freqs[r])

            freqData = self.inferenceData.freqs[r, width/2:]
            print width
            print freqData
            print freqData[:width/2]
            IData = self.inferenceData.Is[r, i, width/2:]
            QData = self.inferenceData.Qs[r, i, width/2:]


            pwr = self.inferenceData.attens[r, i]
            print i, pwr
            dataDict = {'name': resName + '_%i' % r, 'temp': temp, 'pwr': pwr * -1, 'freq': freqData, 'I': IData,
                        'Q': QData}
            fileDataDicts.append(dataDict)

        resList = [scr.makeResFromData(fileDataDict) for fileDataDict in fileDataDicts]
        # resList2 = [scr.makeResFromData(fileDataDict) for fileDataDict in fileDataDicts]

        chi2_singles = []
        # chi2_doubles = []

        # for res in resList:
        #     chi2_singles.append(self.do_fit(res))
        # self.prev_a = self.max_a

        for res in resList:
            self.do_fit(res)
        # self.prev_ad = self.max_a

        # sanity check
        # plt.plot(chi2_singles, label='single')
        # plt.plot(chi2_doubles, label='double')
        # plt.legend()
        # plt.show()

        # if sum(chi2_singles) < sum(chi2_doubles):
        #     return resList
        # else:
        #     return resList2

        fileDataDicts = []
        # for i, pwr in enumerate(self.inferenceData.attens[r]):
        for i in [len(self.inferenceData.attens[r])-1]:
            resName = 'RES'
            temp = 100
            width = len(self.inferenceData.freqs[r])
            freqData = self.inferenceData.freqs[r, :width/2]
            IData = self.inferenceData.Is[r, i, :width/2]
            QData = self.inferenceData.Qs[r, i, :width/2]
            pwr = self.inferenceData.attens[r, i]
            print i, pwr
            plt.plot(IData, QData)
            plt.show()
            dataDict = {'name': resName + '_%i' % r, 'temp': temp, 'pwr': pwr * -1, 'freq': freqData, 'I': IData,
                        'Q': QData}
            fileDataDicts.append(dataDict)

        resList = [scr.makeResFromData(fileDataDict) for fileDataDict in fileDataDicts]

        return resList

    def do_fit(self, res):

        # force_a_trend forcing the nonliniearity param a to be basically always decreasing with increasing atten
        # if self.force_a_trend != None:
        #     max_a = self.prev_a * self.force_a_trend
        # else:
        #     max_a = self.max_a
        # kwargs = {'max_a': max_a}

        # if self.force_a_trend != None:
        #     print 'prev_a', self.prev_a, self.prev_a / self.force_a_trend
        #     min_a = self.prev_a / self.force_a_trend
        # else:
        #     min_a = self.min_a
        # kwargs = {'min_a': min_a}
        # # kwargs = {'f0_guess': f0_guess}
        # res.load_params(scr.cmplxIQ_params, **kwargs)
        res.load_params(scr.cmplxIQ_params)

        # kwargs = {'maxfev': 10000} # useful for speeding things up
        # res.do_lmfit(scr.cmplxIQ_fit, **kwargs)
        res.do_lmfit(scr.cmplxIQ_fit)

        # self.prev_a = res.lmfit_vals[-1]
        # self.prev_f0 = res.lmfit_vals[]

        residual = sum(res.residualI ** 2 + res.residualQ ** 2)
        return residual

    def do_fit_double(self, res):
        # if residual > 10000:
        # max_a = self.prev_a * 3
        # max_ad = self.prev_ad * 3
        # max_a = self.max_a
        # max_ad = self.max_a
        # kwargs = {'max_a': max_a, 'max_ad': max_ad}
        print 'Fitting for two peaks'
        # res.load_params(scr.cmplxIQ_params_cols, **kwargs)
        res.load_params(scr.cmplxIQ_params_cols)
        # kwargs={'maxfev':2000}
        res.do_lmfit(scr.cmplxIQ_fit_cols)

        # a_loc = res.lmfit_labels.index('a')
        # self.prev_a = res.lmfit_vals[a_loc]

        # ad_loc = res.lmfit_labels.index('ad')
        # self.prev_ad = res.lmfit_vals[ad_loc]

        # print 'prev_a', self.prev_a
        # print 'prev_ad', self.prev_ad

        residual = sum(res.residualI ** 2 + res.residualQ ** 2)
        print residual
        return residual

    def inspect(self,r):
        # freqData = self.inferenceData.freqs[r]
        print np.shape(self.inferenceData.Is)
        IData = self.inferenceData.Is[r,10]
        QData = self.inferenceData.Qs[r,10]       

        print r
        plt.plot(IData**2 + QData**2)
        plt.show()
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

    def fitres_wfit(self, r=0, p=0):
        pwr = self.inferenceData.attens[r][p]
        resName = 'RES'
        temp = 100
        freqData = self.inferenceData.freqs[r]
        IData = self.inferenceData.Is[r, p]
        QData = self.inferenceData.Qs[r, p]

        dataDict = {'name': resName + '_%i' % r, 'temp': temp, 'pwr': pwr * -1, 'freq': freqData, 'I': IData,
                    'Q': QData}

        fileDataDict =dataDict

        res = scr.makeResFromData(fileDataDict, paramsFn = scr.cmplxIQ_params, fitFn = scr.cmplxIQ_fit)
        print 'Do fit results exist for the first object? ', res.hasFit

        return res

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

    def save_resLists(self, resLists):
        

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
                pwr = np.argmin(self.nonlin_params[irl])
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
     
    def plotresSweepParamsVsPwr(self,resSweep, plot_keys=None):
        if plot_keys == None:
            plot_keys= ['f0','df', 'qi','qc','q0', 'redchi', 'a']#, 'f0d','dfd', 'qid','qcd', 'ad']
        # if plot_fits:
        #     plot_fits = [True] * np.ones((len(plot_types)))
        # else:
        #     plot_fits = [False] * np.ones((len(plot_types)))
        scr.plotResSweepParamsVsPwr(resSweep,
                                    num_cols=3,
                                plot_keys=plot_keys)


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
    #
    # res = ps.fitres_wfit(r=0,p=15)
    # ps.fit_with_emcee(res)

    # resLists = ps.fitresLists(num_res=50)

    # resList = ps.fit_double_split(18)

    resLists = ps.loadresLists()
    # for resList in resLists:
    #     ps.plotResListData(resList)
    #     resSweep = ps.fitresSweep(resList)
    #     ps.plotresSweepParamsVsPwr(resSweep)
    #     plt.show()


    # # resLists = []
    # for r in range(329, 669): #318 329
    #     print r
    #     if r in doubles:
    #         print 'fitting for two peaks'
    #         resList = ps.comp_double(r)
    #     else:
    #         resList = ps.fitresList(r)
    #     resLists.append(resList)
    # # # resList = ps.fitresList(1)
    # # # ps.evaluatePwrsFromDetune([resList])

    # #     resSweep = ps.fitresSweep(resList)
    # # # # resLists = ps.loadresLists()
    # # # # for resList in resLists:
    # # # #     ps.plotResListData(resList)
    # #     ps.plotResListData(resList)
    # #     ps.plotresSweepParamsVsPwr(resSweep)
    # # plt.show()
    # ps.save_resLists(resLists)

    resses = [0,4, 5,14,15,33]

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
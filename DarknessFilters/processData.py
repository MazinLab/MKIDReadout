import numpy as np
import makeTemplate as mT
import makeFilters as mF
import os

def processData(directory,filterType='matched'):
    '''
    Loop through .dat files in the directory and create filter coefficients and thresholds for each
    INPUTS:
    directory - path for the folder containing the data
    filterType - type of filter to export 
    '''
    #get .dat files into list
    fileList=[]
    for item in os.listdir(directory):
        if os.path.isfile(os.path.join(directory,item)) and item.endswith('.dat'):
            fileList.extend(item)
    
    
    for fileName in fileList
        #load data
        rawData=np.loadtxt(os.path.join(directory,fileName))

        #make template
        template,time , noiseSpectrumDict, _, _ = mT.makeTemplate \       
        (rawData,nSigmaTrig=5.,numOffsCorrIters=3,isVerbose=isVerbose,isPlot=isPlot, sigPass=.5)
    
        #make filter
        if filterType=='matched':
            filterCoef=mF.makeMatchedFilter(template, noiseSpectrumDict['noiseSpectrum'], nTaps=50, tempOffs=75)
        elif filterType=='supermatched':
            fitfunc=lambda x, a, t0 : a*exp(-x/t0)
            t=time[len(time)*1/5:len(time)*4/5]
            expData=template[len(time)*1/5:len(time)*4/5]
            coef, _ = opt.curve_fit(fitfunc,t,expData, [-1 , 30e-6])
            filterCoef=mF.makeSuperMatchedFilter(template, noiseSpectrumDict['noiseSpectrum'], coef[1], nTaps=50, tempOffs=75)
        else:
            raise ValueError('proccessData: filterType not defined')
        #save filter coefficients to an appropriately named file    
        

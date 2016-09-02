import numpy as np
import makeTemplate as mT
import makeFilters as mF
import os
import sys

def processData(directory,filterType='matched',isVerbose=False):
    '''
    Loop through .dat files in the directory and create filter coefficient files for each
    INPUTS:
    directory - path for the folder containing the data
    filterType - type of filter to export 
    '''
    #get .dat files into list
    fileList=[]
    for item in os.listdir(directory):
        if os.path.isfile(os.path.join(directory,item)) and item.endswith('.dat'):
            fileList.extend(item)
    
    #loop through all the phase stream data files in the directories
    for index, fileName in enumerate(fileList):

        #load data
        rawData=np.loadtxt(os.path.join(directory,fileName))

        #make template
        template, time , noiseSpectrumDict, _, _ = mT.makeTemplate \       
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
        saveFileName, _= os.path.splitext(fileName)
        saveFileName+='_filter.dat'
        np.savetxt(saveFileName, filterCoef)
        
        #print progress to terminals
        if isVerbose:
            if index!=len(fileList)-1
                perc=round(float(index)/(len(fileList)-1))
                sys.stdout.write("Percent of filters created: %.1f%%  \r" % (perc) )
                sys.stdout.flush()
            else:
                print "processData: Percent of filters created: 100%"  
        

import numpy as np
import makeTemplate as mT
import makeFilters as mF
import os
import sys
import matplotlib.pyplot as plt

def processData(directory,filterType='matched',isVerbose=False):
    '''
    Loop through .npz files in the directory and create filter coefficient files for each
    INPUTS:
    directory - path for the folder containing the data
    filterType - type of filter to export 
    '''
    #set flag for log file output
    logFileFlag=0
    
    #delete old log and filter coefficients if exists
    if os.path.isfile(os.path.join(directory,"log_file.txt")):
        os.remove(os.path.join(directory,"log_file.txt"))
    if os.path.isfile(os.path.join(directory,'filter_coefficients.txt')):
        os.remove(os.path.join(directory,'filter_coefficients.txt'))    
    
    #print progress to terminal
    if isVerbose:
        sys.stdout.write("Percent of filters created: 0.0%  \r")
        sys.stdout.flush()
        
    #get .dat files into list
    fileList=[]
    for item in os.listdir(directory):
        if os.path.isfile(os.path.join(directory,item)) and item.endswith('.npz'):
            fileList.append(item)
            
    #get channel numbers for each file
    channelNumber=[]#np.zeros((1,len(fileList)))
    timeStamp=[]
    badNameInd=[] 
    for index, fileName in enumerate(fileList):
        #extract channel number from filename (assuming snap_X_chX_DATE-time ... format)
        if fileName.split('_')[2][0:2]=='ch':
            channelNumber.append(int(fileName.split('_')[2][2:]))
            timeString=fileName.split('_')[3]
            timeString=timeString.split('-')[0]+timeString.split('-')[1].split('.')[0]
            timeStamp.append(int(timeString))
        else:
            logFileFlag=1
            badNameInd.append(index)
            with open(os.path.join(directory,"log_file.txt"),'a') as logfile:
                logfile.write("Removed '{0}' from file list due to incorrect name format \r".format(fileName))
    
    #remove filenames with incorrect formats
    fileList=[element for i, element in enumerate(fileList) if i not in badNameInd]         
    
    #sort file list by channel number 
    sortedIndices=np.argsort(channelNumber)
    fileList=[fileList[i] for i in sortedIndices]
    timeStamp=[timeStamp[i] for i in sortedIndices]
    channelNumber=[channelNumber[i] for i in sortedIndices]
     
    
    #find duplicate channel numbers and remove all but the most recent data file
    goodIndicies=[]
    uniqueChannelNumbers=list(set(channelNumber))
    uniqueChannelNumbers.sort()
    for uniqueChannel in uniqueChannelNumbers:
        indexList=[]
        timeStampList=[]
        for channelInd, channel in enumerate(channelNumber):
            #create list of duplicate channels and their time stamps
            if channel==uniqueChannel:
                indexList.append(channelInd)
                timeStampList.append(timeStamp[channelInd])
        #sort timestamps
        sortedIndices=np.argsort(timeStampList)
        
        #print warning about duplicates to logfile
        if len(sortedIndices)>1:
            with open(os.path.join(directory,"log_file.txt"),'a') as logfile:
                for ind in sortedIndices[:-1]:
                    logFileFlag=1
                    logfile.write("Removed '{0}' from file list due to duplicate channel number \r".format(fileList[indexList[ind]]))
        #append channel number index with the largest timestamp to the good index list
        goodIndicies.append(indexList[sortedIndices[-1]])
    
    #pick filenames removing duplicate channel numbers
    fileList=[fileList[i] for i in goodIndicies]            
     
    #initialize filter coefficient array
    filterArray=np.zeros((50,len(fileList)))
            
    #loop through all the phase stream data files in the directories
    for index, fileName in enumerate(fileList):
        try:
            #load data
            rawData=np.load(os.path.join(directory,fileName))
            key=rawData.keys()
            rawData=rawData[key[0]]
        
            #make template
            template, time , noiseSpectrumDict, _, _ = mT.makeTemplate \
            (rawData,nSigmaTrig=5.,numOffsCorrIters=3, sigPass=.05)
        
            #make filter
            if filterType=='matched':
                filterCoef=mF.makeMatchedFilter(template, noiseSpectrumDict['noiseSpectrum'], nTaps=50, tempOffs=95)
            else:
                raise ValueError('proccessData: filterType not defined')
        
            #add filter coefficients to array
            filterArray[:,index]=filterCoef
            
        except:
            logFileFlag=1
            with open(os.path.join(directory,"log_file.txt"),'a') as logfile:
                logfile.write("File '{0}' filter calculation failed. Replaced with delta function filter \r".format(fileName))
            filterArray[0,index]=1 #delta function filter = no filter
            
        #print progress to terminal
        if isVerbose:
            if index!=len(fileList)-1:
                perc=round(float(index+1)/(len(fileList))*100)
                sys.stdout.write("Percent of filters created: %.1f%%  \r" % (perc) )
                sys.stdout.flush()
            else:
                print "Percent of filters created: 100%    "    
        
    np.savetxt(os.path.join(directory,'filter_coefficients.txt'),filterArray) 
    
    with open(os.path.join(directory,"log_file.txt"),'a') as logfile:
        logfile.write("\rFile list that was itterated over: \r")
        for fileName in fileList:
            logfile.write("{0} \r".format(fileName))
             
    if logFileFlag==1:
        print 'check log file for warnings'
                      
if __name__ == '__main__':
    processData('/mnt/data0/Darkness/20160723_nz',isVerbose=True)                
        

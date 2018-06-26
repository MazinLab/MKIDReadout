import numpy as np
import makeTemplate as mT
import makeFilters as mF
import makeNoiseSpectrum as mNS
import os
import sys
import matplotlib.pyplot as plt
import pickle

def processData(directory,defaultFilter,filterType='matched',isVerbose=False):
    '''
    Loop through .npz files in the directory and create filter coefficient files for each
    INPUTS:
    directory - path for the folder containing the data
    filterType - type of filter to export 
    '''
    
    #make default Template (renormalize and flip)
    defaultTemplate=-defaultFilter/np.max(np.abs(defaultFilter))

    #set flag for log file output
    logFileFlag=0
    
    #set counter for number of times filter failed
    filterFail=0
    
    #check before deleting files
    continuingFlag=0
    if os.path.isfile(os.path.join(directory,"log_file_20170407.txt")):
        answer=query_yes_no("Are you continuing a stopped calculation?")
        if answer is True:
            continuingFlag=1
        else:
            answer=query_yes_no("Are you sure that you want to delete previous filter calculations?")
            if answer is False:
                return

    #delete old log and filter coefficients if exists if told to do so
    if not continuingFlag:
        if os.path.isfile(os.path.join(directory,"log_file_20170407.txt")):
            os.remove(os.path.join(directory,"log_file_20170407.txt"))
        if os.path.isfile(os.path.join(directory,'filter_coefficients_20170407.txt')):
            os.remove(os.path.join(directory,'filter_coefficients_20170407.txt'))    
        if os.path.isfile(os.path.join(directory,'template_coefficients_20170407.txt')):
            os.remove(os.path.join(directory,'template_coefficients_20170407.txt'))
        if os.path.isfile(os.path.join(directory,'filter_type_20170407.txt')):
            os.remove(os.path.join(directory,'filter_type_20170407.txt')) 
        if os.path.isfile(os.path.join(directory,'noise_data_20170407.txt')):
            os.remove(os.path.join(directory,'noise_data_20170407.txt'))   
        if os.path.isfile(os.path.join(directory,'rough_templates_20170407.txt')):
            os.remove(os.path.join(directory,'rough_templates_20170407.txt'))
        if os.path.isfile(os.path.join(directory,'file_list_20170407.txt')):
            os.remove(os.path.join(directory,'file_list_20170407.txt'))
        if os.path.isfile(os.path.join(directory,'filters_fourier_20170407.txt')):
            os.remove(os.path.join(directory,'filters_fourier_20170407.txt'))
        
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
            if not continuingFlag:
                with open(os.path.join(directory,"log_file_20170407.txt"),'a') as logfile:
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
        if len(sortedIndices)>1 and not continuingFlag:
            with open(os.path.join(directory,"log_file_20170407.txt"),'a') as logfile:
                for ind in sortedIndices[:-1]:
                    logFileFlag=1
                    logfile.write("Removed '{0}' from file list due to duplicate channel number \r".format(fileList[indexList[ind]]))
        #append channel number index with the largest timestamp to the good index list
        goodIndicies.append(indexList[sortedIndices[-1]])
    
    #warn about unexpected files
    if logFileFlag and not continuingFlag:
        print "Unexpected files in the current directory. Check the log file to make sure the program removed the right ones!"
        with open(os.path.join(directory,"log_file_20170407.txt"),'a') as logfile:
            logfile.write("\r")
            
    
    #print progress to terminal
    if isVerbose:
        sys.stdout.write("Percent of filters created: 0.0%  \r")
        sys.stdout.flush()
            
    #pick filenames removing duplicate channel numbers
    fileList=[fileList[i] for i in goodIndicies] 
    
    #fileList=fileList[0:2]
    #fileList=[fileList[2]] #uncomment this line for debugging particular files

    #start at a particular file if continuing calculation
    if continuingFlag: 
        filters=np.loadtxt(os.path.join(directory,"filter_coefficients_20170407.txt"))
        num=filters.shape[0]
        try:
            fileListOriginal=fileList
            fileList=fileList[num:]
            with open(os.path.join(directory,'file_list_20170407.txt'), 'wb') as fp:
                pickle.dump(fileListOriginal, fp)
        except: 
            print 'No more files to itterate over'
            return
    else:
        #save file list if not continuing from previous calculation
        with open(os.path.join(directory,'file_list_20170407.txt'), 'wb') as fp:
            pickle.dump(fileList, fp)
    
    #initialize arrays
    filterArray=np.zeros((len(fileList),50))
    templateArray=np.zeros((len(fileList),50))
    roughTemplateArray=np.zeros((len(fileList),50))
    typeArray=np.zeros(len(fileList)) 
    noiseArray=np.zeros((len(fileList),60)) 
    filterNoiseArray=np.zeros((len(fileList),60))
            
    #loop through all the phase stream data files in the directories
    for index, fileName in enumerate(fileList):
        #reinitialize noise flag
        noiseFlag=0

        #load data
        try:
            errorFlag=0
            #load data
            rawData=np.load(os.path.join(directory,fileName))
            key=rawData.keys()
            rawData=rawData[key[0]]
        except:
            with open(os.path.join(directory,"log_file_20170407.txt"),'a') as logfile:
                logfile.write("{1}: File '{0}' data failed to load. Using default template as filter \r".format(fileName,index))
            #use default filter and templates
            templateArray[index,:]=defaultTemplate
            filterArray[index,:]=defaultFilter
            
            #save data 
            with open(os.path.join(directory,'filter_coefficients_20170407.txt'),'a') as filters:
                np.savetxt(filters,np.atleast_2d(filterArray[index,:]))
            with open(os.path.join(directory,'template_coefficients_20170407.txt'),'a') as templates:
                np.savetxt(templates,np.atleast_2d(templateArray[index,:]))
            with open(os.path.join(directory,'rough_templates_20170407.txt'),'a') as rough:
                np.savetxt(rough,np.atleast_2d(roughTemplateArray[index,:]))
            with open(os.path.join(directory,'filter_type_20170407.txt'),'a') as types:
                np.savetxt(types,np.atleast_1d(typeArray[index]))
            with open(os.path.join(directory,'noise_data_20170407.txt'),'a') as noise:
                np.savetxt(noise,np.atleast_2d(noiseArray[index,:]))
            with open(os.path.join(directory,'filters_fourier_20170407.txt'),'a') as noise:
                np.savetxt(noise,np.atleast_2d(filterNoiseArray[index,:]))

            continue    

        #make template    
        try:
            template, time , noiseSpectrumDict, templateList, _ = mT.makeTemplate \
            (rawData,nSigmaTrig=5.,numOffsCorrIters=3, sigPass=.05,defaultFilter=defaultFilter)  
            roughTemplateArray[index,:]=templateList[-1][0:50]   
            noiseFlag=1          
            #check for bad template (fall times greater than 7 and less than 50 .. assuming 50 points given)
            # if np.trapz(template[0:50])>-3 or np.trapz(template[0:50])<-31.6 or templateList[-1][0]!=-1:
            #     errorFlag=1
            #     raise ValueError('proccessData: template not correct')
            #add template coefficients to array    
            templateArray[index,:]=template[0:50]  
            templateFlag=1;              
        except:
            #add template coefficients to array
            templateArray[index,:]=defaultTemplate
            templateFlag=0;      

        #make filter    
        try:
            filterCoef = mF.makeWienerFilter(template)
            # if templateFlag:
            #     filterCoef=mF.makeWienerFilter(template, noiseSpectrumDict['noiseSpectrum'])
            # elif noiseFlag:
            #     filterCoef=mF.makeWienerFilter(np.concatenate((defaultTemplate,np.zeros(10))), noiseSpectrumDict['noiseSpectrum'])
            # else:
            #     data = mT.hpFilter(rawData)
            #     noiseSpectrumDict = mNS.makeWienerNoiseSpectrum(data,numBefore=60,numAfter=0,template=defaultFilter)
            #     filterCoef=mF.makeWienerFilter(np.concatenate((defaultTemplate,np.zeros(10))),noiseSpectrumDict['noiseSpectrum'])

            #add filter coefficients to array 
            filterArray[index,:]=filterCoef[0:50]
            filterFlag=1;
        except:
            #add filter coefficients to array
            if templateFlag:
                filterArray[index,:]=-template/np.dot(template,template)
            else:
                filterArray[index,:]=defaultFilter
            noiseArray[index,:]=np.zeros(60)
            filterFlag=0;     
        
        #log results and categorize them in the type array
        with open(os.path.join(directory,"log_file_20170407.txt"),'a') as logfile:
            if templateFlag==0 and filterFlag==0:
                logfile.write("{1}: File '{0}' template and filter calculation failed. Using default template as filter \r".format(fileName,index))
                typeArray[index]=0            
            elif templateFlag==1 and filterFlag==0:
                logfile.write("{1}: File '{0}' filter calculation failed. Using calculated template as filter \r".format(fileName,index))
                typeArray[index]=1
            elif templateFlag==0 and filterFlag==1:
                logfile.write("{1}: File '{0}' template calculation failed. Using default template with noise as filter \r".format(fileName,index))
                typeArray[index]=2
            else:
                logfile.write("{1}: File '{0}' calculation successful \r".format(fileName,index))
                typeArray[index]=3                                
        
        if filterFlag:
            filterNoiseArray[index,:]=np.abs(np.fft.fft(filterCoef))**2  
               
        #write new data to file 
        with open(os.path.join(directory,'filter_coefficients_20170407.txt'),'a') as filters:
            np.savetxt(filters,np.atleast_2d(filterArray[index,:]))
        with open(os.path.join(directory,'template_coefficients_20170407.txt'),'a') as templates:
            np.savetxt(templates,np.atleast_2d(templateArray[index,:]))
        with open(os.path.join(directory,'rough_templates_20170407.txt'),'a') as rough:
            np.savetxt(rough,np.atleast_2d(roughTemplateArray[index,:]))
        with open(os.path.join(directory,'filter_type_20170407.txt'),'a') as types:
            np.savetxt(types,np.atleast_1d(typeArray[index]))
        with open(os.path.join(directory,'noise_data_20170407.txt'),'a') as noise:
            np.savetxt(noise,np.atleast_2d(noiseArray[index,:]))
        with open(os.path.join(directory,'filters_fourier_20170407.txt'),'a') as noise:
            np.savetxt(noise,np.atleast_2d(filterNoiseArray[index,:]))  
          
        #print progress to terminal
        if isVerbose:
            if index!=len(fileList)-1:
                if continuingFlag:
                    index0=index+len(fileListOriginal)-len(fileList)
                    fileList0=fileListOriginal
                else:
                    index0=index
                    fileList0=fileList
                perc=round(float(index0+1)/(len(fileList0))*100)
                sys.stdout.write("Percent of filters created: %.1f%%  \r" % (perc) )
                sys.stdout.flush()
            else:
                print "Percent of filters created: 100%    "    
    
    #count number of each type of filter    
    unique, counts = np.unique(typeArray,return_counts=True)
    countdict=dict(zip(unique, counts))
    if 0 not in countdict.keys():
        countdict[0]=0    
    if 1 not in countdict.keys():
        countdict[1]=0
    if 2 not in countdict.keys():
        countdict[2]=0
    if 3 not in countdict.keys():
        countdict[3]=0   

    #print final results 
    print "{0}% of pixels using optimal filters".format(round(countdict[3]/float(len(fileList))*100,2))
    print "{0}% of pixels using default template with noise as filter".format(round(countdict[2]/float(len(fileList))*100,2))   
    print "{0}% of pixels using calculated template as filter".format(round(countdict[1]/float(len(fileList))*100,2))
    print "{0}% of pixels using default template as filter".format(round(countdict[0]/float(len(fileList))*100,2))
    
    #log final results
    with open(os.path.join(directory,"log_file_20170407.txt"),'a') as logfile:
        logfile.write("\rFile list that was itterated over: \r")
        for fileName in fileListOriginal:
            logfile.write("{0} \r".format(fileName))
        logfile.write("\r{0}% of pixels using optimal filters \r".format(round(countdict[3]/float(len(fileList))*100,2)))
        logfile.write("{0}% of pixels using default template with noise as filter \r".format(round(countdict[2]/float(len(fileList))*100,2)))
        logfile.write("{0}% of pixels using calculated template as filter \r".format(round(countdict[1]/float(len(fileList))*100,2)))
        logfile.write("{0}% of pixels using default template as filter \r".format(round(countdict[0]/float(len(fileList))*100,2)))    

def query_yes_no(question, default="no"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

                 
if __name__ == '__main__':
    defaultFilter=np.loadtxt('/mnt/data0/SDR/Projects/Filters/matched50_15.0us.txt')
    processData('/mnt/data0/Darkness/20170403/optimal_filters/122_data/',defaultFilter,isVerbose=True)                
        
               
        

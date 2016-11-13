import numpy as np
import os
from datetime import datetime
from Roach2Controls import Roach2Controls

def takePhaseData(arrayName='ukko', duration=30, freqChannels=range(0,1024), ipIDs=range(112,123), paramFile='../DataReadout/ChannelizerControls/Darkness_V2.param'):
    dt = datetime.today()
    dirPath = os.path.join(os.getcwd(),testData, str(dt.year)+str(dt.month).zfill(2)+
        str(dt.day).zfill(2)+str(dt.hour).zfill(2)+str(dt.minute).zfill(2)+str(dt.second).zfill(2))
    os.mkdir(dirPath)
    for ipID in ipIDs
        roach = Roach2Controls('10.0.0.'+str(ipID), paramFile)
        roach.connect()
        for ch in freqChannels:
            phaseStream = roach.takePhaseStreamDataOfFreqChannel(ch, duration)
            np.savetxt(os.path.join(dirPath,str(ipID)+'_'+str(ch).zfill(4)+'_'+str(duration)+'.dat'),phaseStream)

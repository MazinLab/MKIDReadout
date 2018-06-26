from Roach2Controls import Roach2Controls
from getAdcAttens import *
import os, sys

if __name__=='__main__':
    roachList = []
    specDictList = []
    plotSnaps = True
    startAtten = 40

    for arg in sys.argv[1:]:
        ip = '10.0.0.'+arg
        roach = Roach2Controls(ip, 'DarknessFpga_V2.param', True)
        roach.connect()
        roach.initializeV7UART()
        roachList.append(roach)
        snapDict = roach.snapZdok()
        specDict = streamSpectrum(snapDict['iVals'], snapDict['qVals'])
        specDictList.append(specDict)

    figList = []
    axList = []
    for i,specDict in enumerate(specDictList):
        fig,ax = plt.subplots(1, 1)
        ax.plot(specDict['times'], specDict['signal'].real, color='b', label='I')
        ax.plot(specDict['times'], specDict['signal'].imag, color='g', label='Q')
        ax.set_title('Roach ' + roachList[i].ip[-3:] + ' Timestream')
        ax.set_xlabel('Time (us)')
        ax.set_xlim([0,0.5])
        ax.legend()

    plt.show()

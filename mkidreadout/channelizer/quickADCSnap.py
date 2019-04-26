
#TODO merge this functionality into __main__ of Roach2Controls
import argparse

from Roach2Controls import Roach2Controls
from adcTools import *

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Take quick ADC snapshots')
    parser.add_argument('roaches', nargs='+', type=str, help='List of ROACH numbers (last 3 digits of IP)')
    parser.add_argument('-s', '--plot-spectrum', action='store_true', help='Plot spectrum of snapshot')
    clOptions = parser.parse_args()
    roachList = []
    specDictList = []
    plotSnaps = True
    startAtten = 40

    for roachNum in clOptions.roaches:
        ip = '10.0.0.'+roachNum
        roach = Roach2Controls(ip, 'darknessfpga.param', True)
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

        if clOptions.plot_spectrum:
            fig,ax = plt.subplots(1,1)
            ax.plot(specDict['freqsMHz'], specDict['spectrumDb'])
            ax.set_title('Roach ' + roachList[i].ip[-3:] + ' Spectrum')
            ax.set_xlabel('Frequency (MHz)')
            ax.set_ylabel('Amplitude (dB)')


    plt.show()

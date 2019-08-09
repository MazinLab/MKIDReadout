"""
Author:    Matt Strader

"""
from __future__ import print_function

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

from mkidcore.corelog import getLogger
from mkidreadout.channelizer.Roach2Controls import Roach2Controls


def checkRamp(rampVals, nBits=12, bPlot=False):
    maxRampVal = 2 ** nBits
    offset = 2 ** (nBits - 1)
    rampStep = 1
    rampModel = offset + rampVals[0] + np.arange(0, len(rampVals), rampStep)
    rampModel = (np.array(rampModel, dtype=np.int) % maxRampVal) - offset
    rampErrors = rampVals - rampModel
    errorsFound = np.any(rampErrors != 0)

    if bPlot:
        fig, (ax, ax2) = plt.subplots(2, 1)
        x = np.arange(len(rampVals))
        ax.step(x, rampVals)
        ax.step(x, rampModel)
        ax2.step(x, rampVals - rampModel)
        plt.show()

    return errorsFound

def findCal(roach, bPlot=False):
    snapDict = roach.snapZdok()
    errorInI = checkRamp(snapDict['iVals'], bPlot=bPlot)
    errorInQ = checkRamp(snapDict['qVals'], bPlot=bPlot)
    initialError = errorInI | errorInQ
    if not initialError:
        getLogger(__name__).info('keep initial state')
        # started at valid solution, so return index 0
        return {'solutionFound': True, 'solution': 0}
    else:
        nSteps = 60
        stepSize = 4
        failPattern = findCalPattern(roach, nSteps=nSteps, stepSize=stepSize)['failPattern']
        getLogger(__name__).info('fail pat %s', failPattern)
        passPattern = (failPattern == 0.)
        getLogger(__name__).info('pass pat %s', passPattern)
        try:
            firstRegionSlice = scipy.ndimage.find_objects(scipy.ndimage.label(passPattern)[0])[0][0]
            getLogger(__name__).info('First Region Slice %s', firstRegionSlice)
            regionIndices = np.arange(len(passPattern))[firstRegionSlice]
            getLogger(__name__).info('Region indices %s', regionIndices)
            solutionIndex = regionIndices[0] + (len(regionIndices) - 1) // 2
            getLogger(__name__).debug('Solution index %s', solutionIndex)
            roach.incrementMmcmPhase(solutionIndex * stepSize)
            getLogger(__name__).info('solution found %s', solutionIndex * stepSize)
            return {'solutionFound': True, 'solution': solutionIndex * stepSize}
        except IndexError:
            return {'solutionFound': False}


def findCalPattern(roach, bPlot=False, nSteps=60, stepSize=4):
    failPattern = np.zeros(nSteps)
    stepIndices = np.arange(0, nSteps * stepSize, stepSize)

    totalChange = 0
    for iStep in range(nSteps):
        roach.incrementMmcmPhase(stepSize=stepSize)
        totalChange += stepSize

        snapDict = roach.snapZdok()
        errorInI = checkRamp(snapDict['iVals'], bPlot=bPlot)
        errorInQ = checkRamp(snapDict['qVals'], bPlot=bPlot)
        failPattern[iStep] = errorInI | errorInQ

    roach.fpga.write_int('adc_in_pos_phs', 1)
    roach.incrementMmcmPhase(stepSize=totalChange)
    roach.fpga.write_int('adc_in_pos_phs', 0)

    return {'failPattern': failPattern, 'stepIndices': stepIndices}


if __name__ == '__main__':
    if len(sys.argv) > 1:
        ip = '10.0.0.' + sys.argv[1]
    else:
        print('Usage:', sys.argv[0], 'roachNum')
    print(ip)

    roach = Roach2Controls(ip, '/mnt/data0/MkidDigitalReadout/DataReadout/ChannelizerControls/DarknessFpga_V2.param',
                           True, False)
    roach.connect()
    roach.initializeV7UART()
    roach.sendUARTCommand(0x4)
    time.sleep(15)

    time.sleep(.01)
    if not roach.fpga.is_running():
        print('Firmware is not running. Start firmware, and calibrate qdr first!')
        exit(0)
    roach.fpga.get_system_information()
    print('Fpga Clock Rate:', roach.fpga.estimate_fpga_clock())
    roach.fpga.write_int('run', 1)

    roach.loadFullDelayCal()

    calDict = findCal(roach, True)
    print(calDict)

    roach.sendUARTCommand(0x5)

    print('DONE!')

"""
Takes in a raw beammap and finds an optimal physical shift in pixel coordinates to move the raw beammap such that it
matches
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
from astropy.stats import mad_std
import sys, os
# from mkidreadout.utils.readDict import readDict


def readInFrequencies(powerSweepFiles):
    """Takes power sweep files for an array and makes a list with all of them accessible. Each row contains
    a [resID, Frequency (Hz), power], for this we only care about matching resID with frequency."""
    freqarr = np.loadtxt(powerSweepFiles[0])
    for i in range(len(powerSweepFiles) - 1):
        sweep = np.loadtxt(powerSweepFiles[i + 1])
        freqarr = np.concatenate((freqarr, sweep))
    return freqarr


def matchFreqToResID (map):
    frequencies = readInFrequencies(FreqSweepFiles)
    mapWithFreqs = np.full((len(map), len(map[0])+1), float("NaN"))
    for i in range(len(map)):
        mapWithFreqs[i][0], mapWithFreqs[i][1], mapWithFreqs[i][2], mapWithFreqs[i][3] = map[i]
    for j in range(len(frequencies)):
        idx = np.where(frequencies[j][0] == map[:, 0])[0][0]
        mapWithFreqs[idx][4] = (frequencies[j][1] / (10 ** 6))

    return mapWithFreqs


def grabFeedline(rawmap, feedlinenumber):
    tempmap = matchFreqToResID(rawmap)
    feedline = tempmap[np.where(feedlinenumber == np.floor(tempmap[:, 0] / 10000))[0]]
    return feedline


def isincorrectfeedlineX(resonator, feedlinenumber, flipX=True):
    if flipX :
        if feedlinenumber == 1:
            if 126 <= int(np.floor(resonator[2])) <= 139:
                return True
            else:
                return False
        if feedlinenumber == 2:
            if 112 <= int(np.floor(resonator[2])) <= 125:
                return True
            else:
                return False
        if feedlinenumber == 3:
            if 98 <= int(np.floor(resonator[2])) <= 111:
                return True
            else:
                return False
        if feedlinenumber == 4:
            if 84 <= int(np.floor(resonator[2])) <= 97:
                return True
            else:
                return False
        if feedlinenumber == 5:
            if 70 <= int(np.floor(resonator[2])) <= 83:
                return True
            else:
                return False
        if feedlinenumber == 6:
            if 56 <= int(np.floor(resonator[2])) <= 69:
                return True
            else:
                return False
        if feedlinenumber == 7:
            if 42 <= int(np.floor(resonator[2])) <= 55:
                return True
            else:
                return False
        if feedlinenumber == 8:
            if 28 <= int(np.floor(resonator[2])) <= 41:
                return True
            else:
                return False
        if feedlinenumber == 9:
            if 14 <= int(np.floor(resonator[2])) <= 27:
                return True
            else:
                return False
        if feedlinenumber == 10:
            if 0 <= int(np.floor(resonator[2])) <= 13:
                return True
            else:
                return False
    else :
        if feedlinenumber == 10:
            if 126 <= int(np.floor(resonator[2])) <= 139:
                return True
            else:
                return False
        if feedlinenumber == 9:
            if 112 <= int(np.floor(resonator[2])) <= 125:
                return True
            else:
                return False
        if feedlinenumber == 8:
            if 98 <= int(np.floor(resonator[2])) <= 111:
                return True
            else:
                return False
        if feedlinenumber == 7:
            if 84 <= int(np.floor(resonator[2])) <= 97:
                return True
            else:
                return False
        if feedlinenumber == 6:
            if 70 <= int(np.floor(resonator[2])) <= 83:
                return True
            else:
                return False
        if feedlinenumber == 5:
            if 56 <= int(np.floor(resonator[2])) <= 69:
                return True
            else:
                return False
        if feedlinenumber == 4:
            if 42 <= int(np.floor(resonator[2])) <= 55:
                return True
            else:
                return False
        if feedlinenumber == 3:
            if 28 <= int(np.floor(resonator[2])) <= 41:
                return True
            else:
                return False
        if feedlinenumber == 2:
            if 14 <= int(np.floor(resonator[2])) <= 27:
                return True
            else:
                return False
        if feedlinenumber == 1:
            if 0 <= int(np.floor(resonator[2])) <= 13:
                return True
            else:
                return False


def isonarrayY(resonator):
    if 0 <= int(np.floor(resonator[3])) <= 145:
        return True
    return False


def matchFeedlineToDesign (feedline, design, method):
    flnumber = int(np.floor(feedline[0][0] / 10000))
    minfreq = np.min(feedline[:, 4][~np.isnan(feedline[:, 4])])
    maxfreq = np.max(feedline[:, 4][~np.isnan(feedline[:, 4])])
    medfreq = np.median(feedline[:, 4][~np.isnan(feedline[:, 4])])
    meanfreq = np.mean(feedline[:, 4][~np.isnan(feedline[:, 4])])
    matchlist = []
    feedlinecopy = np.copy(feedline)
    for i in range(len(feedline)):
        if not np.isnan(feedline[i][2]) and isincorrectfeedlineX(feedlinecopy[i], flnumber):
            feedlinecopy[i][2] = np.floor(feedlinecopy[i][2]) % 14
        if not np.isnan(feedline[i][3]):
            feedlinecopy[i][3] = np.floor(feedlinecopy[i][3])
    for j in range(len(feedlinecopy)):
        if not np.isnan(feedlinecopy[j][2]) and not np.isnan(feedlinecopy[j][3]) and \
                0 <= int(feedlinecopy[j][2]) <= 13 and isonarrayY(feedlinecopy[j]) and not np.isnan(feedlinecopy[j][4]):
            if method == "minimum":
                feedlinecopy[j][4] = feedlinecopy[j][4] - minfreq
            elif method == "maximum":
                designmax = np.max(design.flatten())
                maxdiff = maxfreq - designmax
                feedlinecopy[j][4] = feedlinecopy[j][4] - maxdiff
            elif method == "median":
                designmed = np.median(design.flatten())
                meddiff = medfreq - designmed
                feedlinecopy[j][4] = feedlinecopy[j][4] - meddiff
            elif method == "mean":
                designmean = np.mean(design.flatten())
                meandiff = meanfreq - designmean
                feedlinecopy[j][4] = feedlinecopy[j][4] - meandiff
            else :
                feedlinecopy[j][4] = feedlinecopy[j][4]
            designfreqatlocation = design[int(feedlinecopy[j][3])][int(feedlinecopy[j][2])]
            residual = feedlinecopy[j][4] - designfreqatlocation
            temparray = [feedlinecopy[j][0], feedlinecopy[j][4], designfreqatlocation, residual]
            matchlist.append(temparray)
    matchlist = np.array(matchlist)
    return matchlist


def shiftFeedline (feedline, xshift, yshift):
    newfeedline = np.copy(feedline)
    newfeedline[:, 2] = newfeedline[:, 2] + xshift
    newfeedline[:, 3] = newfeedline[:, 3] + yshift
    return newfeedline


def getshiftedmaps (feedline, design, maxshiftx, maxshifty, method):
    xmax = np.abs(maxshiftx)
    if xmax > 10 :
        xmax = 10
    ymax = np.abs(maxshifty)
    if ymax > 50 :
        ymax = 50
    xvals = np.linspace(-1 * xmax, xmax, 2 * xmax + 1)
    yvals = np.linspace(-1 * ymax, ymax, 2 * ymax + 1)
    shiftedmaps = []
    for i in range(len(yvals)):
        for j in range(len(xvals)):
            tempmatch = matchFeedlineToDesign(shiftFeedline(feedline, xvals[j], yvals[i]), design, method)
            temparray = [tempmatch, xvals[j], yvals[i], len(tempmatch), mad_std(tempmatch[:, 3]), np.std(tempmatch[:, 3])]
            shiftedmaps.append(temparray)
    shiftedmaps = np.array(shiftedmaps)
    return shiftedmaps


def plotmodelvdata (shiftedmaps, maxshiftx, maxshifty):
    flnum = int(shiftedmaps[int(len(shiftedmaps)/2)][0][0][0]/10000)
    fig, axes = plt.subplots((2*abs(maxshifty)+1), (2*abs(maxshiftx)+1), sharex='all', sharey='all')
    for i in range(len(shiftedmaps)):
        meas, model = shiftedmaps[i][0][:, 1], shiftedmaps[i][0][:, 2]
        axes[int(shiftedmaps[i][2]+maxshifty)][int(shiftedmaps[i][1]+maxshiftx)].plot(model, meas, ',')
    fig.text(0.5, 0.04, 'Design f (MHz)', ha='center')
    fig.text(0.04, 0.5, 'Measured f (MHz)', va='center', rotation='vertical')
    plt.suptitle('Design vs. Measured Frequency by Physical Shift : Feedline ' + str(flnum))
    plt.show()


def plotresidualhistograms (shiftedmaps, maxshiftx, maxshifty):
    flnum = int(shiftedmaps[int(len(shiftedmaps) / 2)][0][0][0] / 10000)
    fig2, axes2 = plt.subplots((2*abs(maxshifty)+1), (2*abs(maxshiftx)+1), sharex='all', sharey='all')
    plt.suptitle('Residuals by Physical Shift : Feedline ' + str(flnum))
    for i in range(len(shiftedmaps)):
        residuals = shiftedmaps[i][0][:, 3]
        axes2[int(shiftedmaps[i][2]+maxshifty)][int(shiftedmaps[i][1]+maxshiftx)].hist(residuals, bins=np.linspace(-1500, 1500, 600))
    fig2.text(0.5, 0.04, 'Residual Distance (MHz)', ha='center')
    plt.show()


def plotMADs (shiftedmaps):
    mads = np.float_(np.reshape(shiftedmaps[:, 4], (7, 7)))
    plt.figure(3)
    plt.imshow(mads, extent=[-mads.shape[1] / 2., mads.shape[1] / 2., mads.shape[0] / 2., -mads.shape[0] / 2.])
    plt.xlabel('x shift')
    plt.ylabel('y shift')
    plt.title('MAD Standard Deviation by physical shift')
    plt.colorbar(orientation='vertical')
    plt.show()


def plotStdDevs (shiftedmaps):
    stds = np.float_(np.reshape(shiftedmaps[:, 5], (7, 7)))
    plt.figure(4)
    plt.imshow(stds, extent=[-stds.shape[1] / 2., stds.shape[1] / 2., stds.shape[0] / 2., -stds.shape[0] / 2.])
    plt.colorbar(orientation='vertical')
    plt.xlabel('x shift')
    plt.ylabel('y shift')
    plt.show('Standard deviation value by physical shift')
    plt.show()


def findbestshift (shiftedmaps):
    mad_idx = np.where(shiftedmaps[:, 4] == np.amin(shiftedmaps[:, 4]))[0][0]
    std_idx = np.where(shiftedmaps[:, 5] == np.amin(shiftedmaps[:, 5]))[0][0]
    if mad_idx == std_idx:
        bestx = shiftedmaps[mad_idx][1]
        besty = shiftedmaps[mad_idx][2]
    else :
        bestx = float('NaN')
        besty = float('NaN')
    return np.array([bestx , besty])


def testfeedline (feedlinenumber, diagnosticplots=True):
    testedfeedline = grabFeedline(rawbeammap, feedlinenumber)
    feedlineshifts = getshiftedmaps(testedfeedline, newdes, 3, 3, 'median')
    bestshift = findbestshift(feedlineshifts)
    if diagnosticplots:
        # plotmodelvdata(feedlineshifts, 3, 3)
        # plotresidualhistograms(feedlineshifts, 3, 3)
        plotMADs(feedlineshifts)
        plotStdDevs(feedlineshifts)
    return bestshift


def testarray (feedlinesmeasured):
    feedlines = np.array(feedlinesmeasured)
    bestshifts = np.full((len(feedlines), 3), np.nan)
    for i in range(len(feedlines)):
        bestshifts[i][0], bestshifts[i][1] = testfeedline(feedlines[i])
        bestshifts[i][2] = feedlines[i]
    return bestshifts


def shiftFullMap (rawmap, bestshiftarray):
    xshifts = bestshiftarray[:, 0]
    yshifts = bestshiftarray[:, 1]
    newmap = matchFreqToResID(rawmap)
    freqs = newmap[:, 4]
    freqs = freqs[~np.isnan(freqs)]
    medianF = np.median(freqs)
    designMed = np.median(designmap.flatten())
    medianDiff = medianF - designMed
    newmap[:, 4] = newmap[:, 4] - medianDiff
    if all(x == xshifts[0] for x in xshifts) and all(y == yshifts[0] for y in yshifts):
        shiftX = xshifts[0]
        shiftY = yshifts[0]
        newmap[:, 2] = np.floor(newmap[:, 2] + shiftX)
        newmap[:, 3] = np.floor(newmap[:, 3] + shiftY)
    else:
        shiftX = np.floor(np.mean(xshifts))
        shiftY = np.floor(np.mean(yshifts))
        newmap[:, 2] = np.floor(newmap[:, 2] + shiftX)
        newmap[:, 3] = np.floor(newmap[:, 3] + shiftY)
    return newmap, [shiftX, shiftY]


def singlePixelResidual (pixel):
    """Input a row from one of the maps, this MUST contain a frequency in the final (5th) column, returns the
    difference between measured and design frequency at that point"""
    measuredF = pixel[4]
    if not np.isnan(measuredF):
        designF = designarray[np.int(np.floor(pixel[3]))][np.int(np.floor(pixel[2]))]
        residual = measuredF - designF
        return residual
    return float("NaN")


def getMapSpread (beammap):
    residuals = np.zeros(len(beammap))
    counter = 0
    counterUnder30 = 0
    if not len(beammap[0]) == 5:
        mapToCheck = matchFreqToResID(beammap)
    else:
        mapToCheck = np.copy(beammap)
    for i in range(len(beammap)):
        if not np.isnan(mapToCheck[i][2]) and not np.isnan(mapToCheck[i][3]) and not np.isnan(mapToCheck[i][4])\
                and isonarrayY(mapToCheck[i]) and isincorrectfeedlineX(mapToCheck[i], np.floor(mapToCheck[i][0]/10000)):
            residuals[i] = singlePixelResidual(mapToCheck[i])
            counter = counter + 1
            if residuals[i] <= 30 and not np.isnan(residuals[i]):
                counterUnder30 = counterUnder30 + 1
        else:
            residuals[i] = float("NaN")
    residuals = residuals[~np.isnan(residuals)]
    return np.std(residuals), mad_std(residuals), counter, counterUnder30


def writeFiles (analyzedMap):
    np.savetxt("cookedmap.txt", analyzedMap[:, 0:4])



if __name__ == "__main__":
    # cfgFn = sys.argv[1]
    # if not os.path.isfile(cfgFn):
    #     mdd = os.environ['MKID_DATA_DIR']
    #     cfgFn = os.path.join(mdd, cfgFn)
    # paramDict = readDict()
    # paramDict.read_from_file(cfgFn)
    # print(paramDict['designMap'],paramDict['rawBeamMap'])
    #
    # designmap = np.genfromtxt(paramDict['designMap'])
    # rawbeammap = np.genfromtxt(paramDict['rawBeamMap'])
    # freqsweeps = paramDict['freqSweepFiles']
    # listofmeasuredFLs = paramDict['FLs']

    designMap = r'mapcheckertesting\mec_feedline.txt'
    rawBeamMap = r'\mapcheckertesting\beammapTestData\newtest\RawMapV1.txt'
    freqSweepFiles = r'\mapcheckertesting\beammapTestData\newtest\ps_*.txt'
    listofmeasuredFLs = [1, 5, 6, 7, 8, 9, 10]
    designmap = np.genfromtxt(designMap)
    rawbeammap = np.genfromtxt(rawBeamMap)
    freqsweeps = freqSweepFiles

    # SHOULD NOT HAVE TO BE CHANGED, THIS IS JUST USED FOR FORMATTING
    designarray = np.roll(np.tile(designmap, 10), 1, axis=1)
    FreqSweepFiles = glob.glob(freqsweeps)
    newdes = np.roll(designmap, 1, axis=1)

    hypatiatest = testarray(listofmeasuredFLs)
    cookedmap, shiftvector = shiftFullMap(rawbeammap, hypatiatest)
    s1, m1, n1, n30_1 = getMapSpread(cookedmap)
    s2, m2, n2, n30_2 = getMapSpread(rawbeammap)

    print("The ideal shift was "+str(shiftvector[0])+" pixels in x and "+str(shiftvector[1])+"pixels in y.")
    print("We placed "+str(n1)+" pixels, including "+str(n30_1)+" within 30 MHz of the design frequency, compared to "+
          str(n2)+" pixels before shifting, which had "+str(n30_2)+" within 30 MHz of the design frequency.")
    print("THe spread in errors is {0:1.3f}".format(m1)+" MHz, compared to {0:1.3f}".format(m2)+" before shifting, a "
          "{0:1.3}".format(100-(m1/m2*100))+"% reduction.")
    writeFiles(cookedmap)
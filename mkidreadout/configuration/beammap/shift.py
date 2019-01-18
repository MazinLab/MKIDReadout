import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import mad_std
import scipy.optimize as opt
from mkidreadout.configuration.beammap.beammap import Beammap, DesignArray
from mkidreadout.configuration.beammap.utils import isResonatorOnCorrectFeedline, placeResonatorOnFeedline

class BeammapShifter(object):
    """
    Needs: Beammap object (that must have frequencies loaded in)
    If a good shift is found, returns: Best shift vector, Beammap object with coordinates shifted,
    and design array with frequencies fitted
    If no shift is found, returns: nothing, will raise an exception that no reliable shift was found

    Designed for use in clean.py
    """
    def __init__(self, designFL, beammap, instrument):
        self.instrument = instrument
        self.design = np.flipud(np.fliplr(np.roll(np.genfromtxt(designFL), 1, 1)))
        self.beammap = beammap
        self.feedlines = []
        self.appliedShift = np.array((np.nan, np.nan))
        self.shiftedBeammap = Beammap()
        self.feedlineShifts = None
        self.designArray = DesignArray()
        if self.instrument.lower() == 'mec':
            self.feedlines = [Feedline(i, self.beammap, self.design, instrument='mec') for i in range(1, 11)]
        elif self.instrument.lower() == 'darkness':
            self.feedlines = [Feedline(i, self.beammap, self.design, instrument='darkness') for i in range(1, 6)]
        else:
            raise Exception('Provided instrument not implemented!')

    def run(self):
        """
        From the feedline objects, determines if an appropriate shift was found, then applies the shift if it is
        :returns an Applied Shift vector (will be (nan, nan) if no shift applied), a Beammap object with the coordinates
        shifted, and the array of design frequencies(140x146 for MEC, 80x125 for DARKNESS), which the frequencies
        appropriately fit to the data:
        """
        self.feedlineShifts = np.array([f.bestshiftvector for f in self.feedlines])
        self.chooseAppliedShift()
        if np.isfinite(self.appliedShift[0]) and np.isfinite(self.appliedShift[1]):
            shiftedData = np.concatenate([f.feedlineData for f in self.feedlines])
            self.shiftedBeammap.setData(shiftedData)
            self.shiftedBeammap.xCoords = self.beammap.xCoords + self.appliedShift[0]
            self.shiftedBeammap.yCoords = self.beammap.yCoords + self.appliedShift[1]
            tempDesignArray = np.concatenate([f.fitDesign for f in self.feedlines], axis=1)
            self.designArray.load(tempDesignArray)
            self.designArray.reshape()
        else:
            self.shiftedBeammap = None
            self.designArray = None
            raise Exception("No shift applied: There was no best shift found")

    def chooseAppliedShift(self):
        """
        :return The applied shift found, currently deemed appropriate if the same shift is found for half or more of the
        feedlines analyzed:
        """
        xshifts = self.feedlineShifts[:, 0]
        xshifts = xshifts[np.isfinite(xshifts)]
        yshifts = self.feedlineShifts[:, 1]
        yshifts = yshifts[np.isfinite(yshifts)]
        for xshift in xshifts:
            if len((xshift == xshifts)) >= len(xshifts) / 2:
                self.appliedShift[0] = xshift
        for yshift in yshifts:
            if len((yshift == yshifts)) >= len(yshifts) / 2:
                self.appliedShift[1] = yshift
        if self.appliedShift[0] == np.nan or self.appliedShift[1] == np.nan:
            raise Exception('The beammap shifting code did not find a good best shift vector :(')


class Feedline(object):
    """
    Needs: Feedline Number, A Beammap Object with coordinates in pixel space (not times), the design feedline text file,
    instrument: DARKNESS or MEC

    Optional Arguments: maxX/Yshifts is how far in the x or y direction that we will search for a best shift; flip is if
    the feedlines were in numerical order (1, 2, 3...) or reverse (10, 9, 8...); Order determines what the order of
    polynomial we try to fit the frequency data to.

    Returns: Feedline data: resIDs, flags, x and y coordinates, and frequency for that resonator (if read out)
    Design map with frequencies fitted
    Best shift vector (x,y)

    If the feedline was not read out on the array, returns unmodified feedline data and design map, best shift vector
    will be (nan, nan)
    """
    def __init__(self, feedlinenumber, beammap, designFL, maxXshift=3, maxYshift=3, flip=False, order=5, instrument=''):
        self.beammap = beammap
        self.design = designFL
        self.flNum = feedlinenumber
        self.maxX = maxXshift
        self.maxY = maxYshift
        self.flip = flip
        self.order = order
        self.instrument = instrument
        self.resIDs = beammap.get('resids', self.flNum)
        self.flags = beammap.get('flags', self.flNum)
        self.xcoords = np.floor(beammap.get('xcoords', self.flNum))
        self.ycoords = np.floor(beammap.get('ycoords', self.flNum))
        self.frequencies = beammap.get('frequencies', self.flNum)
        self.xshifts = np.linspace(-1 * self.maxX, self.maxX, (2 * self.maxX) + 1).astype(int)
        self.yshifts = np.linspace(-1 * self.maxY, self.maxY, (2 * self.maxY) + 1).astype(int)
        self.shiftedXcoords = np.full((len(self.yshifts), len(self.xshifts), len(self.xcoords)), np.nan)
        self.shiftedYcoords = np.full((len(self.yshifts), len(self.xshifts), len(self.ycoords)), np.nan)
        self.minFreq = None
        self.residuals = None
        self.matchedfreqs = None
        self.MAD_std = np.zeros(((2 * self.maxY + 1), (2 * self.maxX + 1)))
        self.std = np.zeros(((2 * self.maxY + 1), (2 * self.maxX + 1)))
        self.bestshiftXcoords = None
        self.bestshiftYcoords = None
        self.bestshiftResiduals = None
        self.bestMatchedFreqs = None
        self.leastSquaresSolution = None
        self.placedPixels = 0

        self.bestshiftvector = np.array((np.nan, np.nan))
        self.feedlineData = None
        self.fitDesign = None
        self.processFeedline()

    def processFeedline(self):
        """
        Goes through the physical shifting and frequency fitting process if a feedline was read out
        If a feedline was not read out on the array, this function will not try to shift/fit data
        """
        if not np.all(np.isnan(self.xcoords)) and not np.all(np.isnan(self.ycoords)):
            self.minFreq = np.min(self.frequencies[np.isfinite(self.frequencies)])
            self.frequencies = self.frequencies - self.minFreq
            self.makeShiftCoords()
            self.findResidualsForAllShifts()
            self.getBestShift()
            self.feedlineData = np.transpose([self.resIDs, self.flags, self.xcoords, self.ycoords, self.frequencies])
            self.fitFrequencies()
            self.countPixelsPlaced()
        else:
            self.feedlineData = np.transpose([self.resIDs, self.flags, self.xcoords, self.ycoords, self.frequencies])
            self.fitDesign = self.design

    def makeShiftCoords (self):
        """
        Takes the original (x,y) coordinates and creates all of the shifted x and y coordinates based on the
        maximum shift in x (self.maxX) and maximum shift in y (self.maxY). THe shifts will range from -maxX to +maxX
        and -maxY to +maxY.
        in the X-by-Y-by-N shiftedX/Ycoords arrays, the [row, column, :] index will refer to a specific shift, with the
        third dimension being the shifted X or Y coordinates
        """
        for i in self.yshifts:
            for j in self.xshifts:
                self.shiftedXcoords[i][j] = self.xcoords + self.xshifts[j]
                self.shiftedYcoords[i][j] = self.ycoords + self.yshifts[i]

    def matchMeastoDes(self, xcoords, ycoords):
        """
        Needs: a list of x coordinates and a list of y coordinates
        Returns: The residual frequencies if the feedline was shifted to these x,y coordinates based on the design map
        """
        residuals = []  # Design - Measured
        matchedf = []  # This is the measured frequency at the point that we are matching to the design frequency
        desf = []  # Design frequency that matches (in physical space) the measured frequency
        for i in range((len(self.resIDs))):
            if isResonatorOnCorrectFeedline(self.resIDs[i], xcoords[i], ycoords[i], self.instrument, self.flip):
                if np.isfinite(xcoords[i]) and np.isfinite(ycoords[i]) and np.isfinite(self.frequencies[i]):
                    x, y = placeResonatorOnFeedline(xcoords[i], ycoords[i], self.instrument)
                    residual = self.design[y][x] - self.frequencies[i]
                    residuals.append(residual)
                    matchedf.append(self.frequencies[i])
                    desf.append(self.design[y][x])
                else:
                    residuals.append(np.nan)
                    matchedf.append(np.nan)
                    desf.append(np.nan)
                    raise Exception("Pixel was not assigned a frequency so we could not find a residual for it. This"
                                    "beammap may already have been cleaned")
            else:
                residuals.append(np.nan)
                matchedf.append(np.nan)
                desf.append(np.nan)

        return np.array(residuals), np.array(matchedf), np.array(desf)

    def findResidualsForAllShifts(self):
        """
        For each (x,y) shift, return the frequency residuals when compared to the design feedline
        """
        self.residuals = np.full((len(self.yshifts), len(self.xshifts), len(self.xcoords)), np.nan)
        self.matchedfreqs = np.full((len(self.yshifts), len(self.xshifts), 2, len(self.xcoords)), np.nan)
        for i in range(len(self.yshifts)):
            for j in range(len(self.xshifts)):
                self.residuals[i][j] = self.matchMeastoDes(self.shiftedXcoords[i][j], self.shiftedYcoords[i][j])[0]
                self.matchedfreqs[i][j][0] = self.matchMeastoDes(self.shiftedXcoords[i][j], self.shiftedYcoords[i][j])[1]
                self.matchedfreqs[i][j][1] = self.matchMeastoDes(self.shiftedXcoords[i][j], self.shiftedYcoords[i][j])[2]

    def removeNaNsfromArray(self, array):
        array = array[np.isfinite(array)]
        return array

    def getBestShift(self):
        """
        Calculates the standard deviation and Median Absolute Deviation standard deviation of the frequency residuals
        for each physical shift. Then, determines what physical shift gives the minimum standard deviation (by both
        measures) and, if the two are the same, gives the best shift vector, otherwise returns a
        'no-shift' vector (nan,nan).
        This also returns the x and y coordinates which correspond to that vector, and the frequency residuals. The
        self.bestMatchedFreqs is the array of frequencies that were matched to the design array (if a resonator was
        shifted off of the feedline it will not be counted)
        """
        for i in range(len(self.yshifts)):
            for j in range(len(self.xshifts)):
                self.MAD_std[i][j] = mad_std(self.removeNaNsfromArray(self.residuals[i][j]))
                self.std[i][j] = np.std(self.removeNaNsfromArray(self.residuals[i][j]))
        MAD_idx = np.unravel_index(self.MAD_std.argmin(), self.MAD_std.shape)
        std_idx = np.unravel_index(self.std.argmin(), self.std.shape)
        if np.array_equal(MAD_idx, std_idx):
            self.bestshiftvector = np.array((self.xshifts[MAD_idx[1]], self.yshifts[MAD_idx[0]]))
            self.bestshiftXcoords = self.shiftedXcoords[MAD_idx[0]][MAD_idx[1]]
            self.bestshiftYcoords = self.shiftedYcoords[MAD_idx[0]][MAD_idx[1]]
            self.bestshiftResiduals = self.residuals[MAD_idx[0]][MAD_idx[1]]
            self.bestMatchedFreqs = self.matchedfreqs[MAD_idx[0]][MAD_idx[1]]
        else :
            self.bestshiftvector = np.array((0, 0))
            self.bestshiftXcoords = self.shiftedXcoords[(2 * self.maxY + 1) // 2][(2 * self.maxX + 1) // 2]
            self.bestshiftYcoords = self.shiftedYcoords[(2 * self.maxY + 1) // 2][(2 * self.maxX + 1) // 2]
            self.bestshiftResiduals = self.residuals[(2 * self.maxY + 1) // 2][(2 * self.maxX + 1) // 2]
            self.bestMatchedFreqs = self.matchedfreqs[(2 * self.maxY + 1) // 2][(2 * self.maxX + 1) // 2]

    def guessInitialParams(self):
        """
        From the frequency data create an initial guess of the parameters to fit the model (design frequencies) to the
        measured frequency data. This will return the coefficients to a Nth-order polynomial as specified by self.order
        that will be the initial guess for the non-linear least squares regression
        """
        data = self.removeNaNsfromArray(self.bestMatchedFreqs[0])
        model = self.removeNaNsfromArray(self.bestMatchedFreqs[1])
        params = np.polyfit(model, data, self.order, full=True)[0]
        return params

    def makeLstSqResiduals(self, parameters):
        """
        Takes in the parameters for the Nth-order polynomial to fit the frequency data to, then finds the data-model
        frequency residuals that the scipy.optimize.least_squares method needs
        """
        p = np.poly1d(parameters)
        data = self.removeNaNsfromArray(self.bestMatchedFreqs[0])
        model = self.removeNaNsfromArray(self.bestMatchedFreqs[1])
        error = data - p(model)
        return error

    def fitFrequencies(self):
        """
        Finds the non-linear least squares fit to our measured data. Modifies the model (feedline design frequency
        array) based on the least squares solution.
        """
        primaryGuess = self.guessInitialParams()
        self.leastSquaresSolution = opt.least_squares(self.makeLstSqResiduals, primaryGuess)
        coefficients = np.poly1d(self.leastSquaresSolution.x)
        self.fitDesign = coefficients(self.design)

    def countPixelsPlaced(self):
        """
        Determines how many pixels we placed on the feedline after shifting
        """
        for resonator in self.feedlineData:
            if resonator[1] == 0 and np.isfinite(resonator[2]) and np.isfinite(resonator[3]):
                self.placedPixels = self.placedPixels + 1

    # The following functions are optional diagnostic functions to corroborate if shifting the feedline resulted in
    # the majority of pixels being placed "well". "Well" being defined as having its frequency be closer to the
    # frequency the design map specifies than any of its adjacent neighbors' design frequencies

    def compareNearestNeighbors(self):
        """
        For the shifted feedline, find the number of pixels where their frequencies are closest to the design feedline (fitted)
        as well as the number of pixels that were placed at a coordinate where its frequency is closer to that of an adjacent
        pixel's design frequency
        """
        self.nearestNeighborFreqLocation = np.full((len(self.feedlineData), 2), np.nan)
        for i in range(len(self.feedlineData)):
            if np.isfinite(self.feedlineData[i][2]) and np.isfinite(self.feedlineData[i][3]) and np.isfinite(self.feedlineData[i][4]):
                nearestNeighborFrequencyLocation = self.findNearestNeighborFrequency(self.feedlineData[i])
                self.nearestNeighborFreqLocation[i][0] = nearestNeighborFrequencyLocation[1] - 1
                self.nearestNeighborFreqLocation[i][1] = nearestNeighborFrequencyLocation[0] - 1

        xvals = self.nearestNeighborFreqLocation[:, 0]
        yvals = self.nearestNeighborFreqLocation[:, 1]

        tl = len(np.where((xvals == -1) & (yvals == -1))[0])
        t = len(np.where((xvals == 0) & (yvals == -1))[0])
        tr = len(np.where((xvals == 1) & (yvals == -1))[0])
        l = len(np.where((xvals == -1) & (yvals == 0))[0])
        c = len(np.where((xvals == 0) & (yvals == 0))[0])
        r = len(np.where((xvals == 1) & (yvals == 0))[0])
        bl = len(np.where((xvals == -1) & (yvals == 1))[0])
        b = len(np.where((xvals == 0) & (yvals == 1))[0])
        br = len(np.where((xvals == 1) & (yvals == 1))[0])
        self.wellPlacedPixels = c
        self.totalPlacedPixels = tl + t + tr + l + c + r + bl + b + br

    def findNearestNeighborFrequency (self, resonator):
        """
        For a given resonator, find the design frequency at each adjacent pixel and determine if the measured frequency
        is closest to where it was placed or if it is closer to the design frequency at an adjacent pixel
        """
        resX, resY = placeResonatorOnFeedline(resonator[2], resonator[3], self.instrument)
        resXp1 = int(resX+1)
        resXm1 = int(resX-1)
        resYp1 = int(resY+1)
        resYm1 = int(resY-1)
        nearestneighborfreqs = [[self.fitDesign[resYm1][resXm1], self.fitDesign[resYm1][resX], self.fitDesign[resYm1][resXp1]],
                                [self.fitDesign[resY][resXm1], self.fitDesign[resY][resX], self.fitDesign[resY][resXp1]],
                                [self.fitDesign[resYp1][resXm1], self.fitDesign[resYp1][resX], self.fitDesign[resYp1][resXp1]]]
        neighborresids = np.abs(nearestneighborfreqs - resonator[4])
        min_place = np.unravel_index(neighborresids.argmin(), neighborresids.shape)
        return min_place

    def plotNearestNeighborInfo(self):
        """
        Plots the data from the "nearest neighbor frequency" analysis. Each spot on the plot corresponds to where the
        nearest neighbor frequency was in relation to a pixel. So, ideal would be that the center point is the highest
        value, meaning we placed most pixels at a point where they most closely matched the feedline design frequency
        array at the same point
        """
        u = self.nearestNeighborFreqLocation[:, 0]
        v = self.nearestNeighborFreqLocation[:, 1]

        tl = len(np.where((u == -1) & (v == -1))[0])
        t = len(np.where((u == 0) & (v == -1))[0])
        tr = len(np.where((u == 1) & (v == -1))[0])
        l = len(np.where((u == -1) & (v == 0))[0])
        c = len(np.where((u == 0) & (v == 0))[0])
        r = len(np.where((u == 1) & (v == 0))[0])
        bl = len(np.where((u == -1) & (v == 1))[0])
        b = len(np.where((u == 0) & (v == 1))[0])
        br = len(np.where((u == 1) & (v == 1))[0])

        heights = np.array([[tl, t, tr], [l, c, r], [bl, b, br]])
        plt.imshow(heights, extent=[-heights.shape[1]/2., heights.shape[1]/2., -heights.shape[0]/2., heights.shape[0]/2. ])
        plt.colorbar()
        plt.show()

    def nearestNeighborQuiver(self):
        """
        Creates a quiver plot where each vector starts at where the pixel was placed and points to which of its nearest
        neighbors' design frequencies its own frequency is closest to. If there is a (0,0) vector, that means that we
        placed the resonator at a location where it is most closely matched to the design frequency at that location.
        """
        x = self.removeNaNsfromArray(self.feedlineData[:, 2])
        y = self.removeNaNsfromArray(self.feedlineData[:, 3])
        u = self.removeNaNsfromArray(self.nearestNeighborFreqLocation[:, 1])
        v = self.removeNaNsfromArray(self.nearestNeighborFreqLocation[:, 0])
        plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1)
        plt.show()


# if __name__ == '__main__':
# USED FOR TESTING ON A LOCAL MACHINE
#
#     designFlPath = '/mnt/data0/nswimmer/Repositories/mapcheckertesting/mec_feedline.txt'
#     rawBM = Beammap()
#     rawBM.load('/mnt/data0/nswimmer/Repositories/mapcheckertesting/beammapTestData/newtest/RawMapV1.txt')
#     rawBM.loadFrequencies(r'/mnt/data0/nswimmer/Repositories/mapcheckertesting/beammapTestData/newtest/ps*.txt')
#     shifter = BeammapShifter(designFlPath, rawBM, "mec")

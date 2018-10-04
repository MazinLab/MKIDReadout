import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import mad_std
import scipy.optimize as opt
from mkidreadout.configuration.beammap.beammap import Beammap
from mkidreadout.configuration.beammap.utils import isResonatorOnCorrectFeedline, placeResonatorOnFeedline

class BeammapShifter(object):
    """
    Needs: Beammap object (that must have frequencies loaded in)
    If a good shift is found, returns: Best shift vector, Beammap object with coordinates shifted,
    and design array with frequencies fitted
    If no shift is found, returns: nothing, will raise an exception that no reliable shift was found
    """
    def __init__(self, designFL, beammap, instrument):
        self.instrument = instrument
        self.design = np.flipud(np.fliplr(np.roll(np.genfromtxt(designFL), 1, 1)))
        self.beammap = beammap
        self.feedlines = []
        self.appliedShift = np.array((np.nan, np.nan))
        self.shiftedBeammap = Beammap()
        self.feedlineShifts = None
        self.designArray = None
        self.createFeedlines()
        self.process()

    def createFeedlines(self):
        if self.instrument.lower() == 'mec':
            self.feedlines = [Feedline(i, self.beammap, self.design, 3, 3, instrument='mec') for i in range(1, 11)]
        elif self.instrument.lower() == 'darkness':
            self.feedlines = [Feedline(i, self.beammap, self.design, 3, 3, instrument='darkness') for i in range(1, 6)]
        else:
            raise Exception('Provided instrument not implemented!')

    def process(self):
        self.feedlineShifts = np.array([f.bestshiftvector for f in self.feedlines])
        self.chooseAppliedShift()
        if np.isfinite(self.appliedShift[0]) and np.isfinite(self.appliedShift[1]):
            shiftedData = np.concatenate([f.newFeedline for f in self.feedlines])
            self.shiftedBeammap.setData(shiftedData)
            self.designArray = np.concatenate([f.fitDesign for f in self.feedlines], axis=1)
        else:
            self.shiftedBeammap = None
            self.designArray = None
            raise Exception("No shift applied: There was no best shift found")

    def chooseAppliedShift(self):
        xshifts = self.feedlineShifts[:, 0]
        xshifts = xshifts[np.isfinite(xshifts)]
        yshifts = self.feedlineShifts[:, 1]
        yshifts = yshifts[np.isfinite(yshifts)]
        for xshift in xshifts:
            if len(np.where(xshift == xshifts)[0]) == len(xshifts):
                self.appliedShift[0] = xshift
        for yshift in yshifts:
            if len(np.where(yshift == yshifts)[0]) == len(yshifts):
                self.appliedShift[1] = yshift
        if self.appliedShift[0] == np.nan or self.appliedShift[1] == np.nan:
            raise Exception('The beammap shifting code did not find a good best shift vector :(')


class Feedline(object):
    def __init__(self, feedlinenumber, beammap, designFL, maxXshift, maxYshift, flip=False, order=5, instrument=''):
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
        if not np.all(np.isnan(self.xcoords)) and not np.all(np.isnan(self.ycoords)):
            self.getFreqExtrema()
            self.frequencies = self.frequencies - self.minF
            self.makeShiftCoords()
            self.findResidualsForAllShifts()
            self.getBestShift()
            self.newFeedline = np.transpose([self.resIDs, self.flags, self.bestshiftXcoords, self.bestshiftYcoords, self.frequencies])
            self.fitFrequencies()
            self.countPixelsPlaced()
        else:
            self.newFeedline = np.transpose([self.resIDs, self.flags, self.xcoords, self.ycoords, self.frequencies])
            self.fitDesign = self.design
            self.bestshiftvector = np.array((np.nan, np.nan))

    def getFreqExtrema (self):
        self.minF = np.min(self.frequencies[np.isfinite(self.frequencies)])
        # self.maxF = np.max(self.frequencies[np.isfinite()(self.frequencies)])
        # self.medianF = np.median(self.frequencies[np.isfinite()(self.frequencies)])
        # self.meanF = np.mean(self.frequencies[np.isfinite()(self.frequencies)])

    def makeShiftCoords (self):
        self.xshifts = np.linspace(-1 * self.maxX, self.maxX, (2 * self.maxX) + 1).astype(int)
        self.yshifts = np.linspace(-1 * self.maxY, self.maxY, (2 * self.maxY) + 1).astype(int)
        self.shiftedXcoords = np.full((len(self.yshifts), len(self.xshifts), len(self.xcoords)), float('NaN'))
        self.shiftedYcoords = np.full((len(self.yshifts), len(self.xshifts), len(self.ycoords)), float('NaN'))
        for i in self.yshifts:
            for j in self.xshifts:
                self.shiftedXcoords[i][j] = self.xcoords + self.xshifts[j]
                self.shiftedYcoords[i][j] = self.ycoords + self.yshifts[i]



    def matchMeastoDes(self, xcoords, ycoords):
        temparray = []
        matchedf = []
        desf = []
        for i in range((len(xcoords))):
            if isResonatorOnCorrectFeedline(self.resIDs[i], xcoords[i], ycoords[i], self.instrument, self.flip):
                if np.isfinite(xcoords[i]) and np.isfinite(ycoords[i]) and np.isfinite(self.frequencies[i]):
                    x,y = placeResonatorOnFeedline(xcoords[i],ycoords[i],self.instrument)
                    residual = self.design[y][x] - self.frequencies[i]
                    temparray.append(residual)
                    matchedf.append(self.frequencies[i])
                    desf.append(self.design[y][x])
            else:
                temparray.append(np.nan)
                matchedf.append(np.nan)
                desf.append(np.nan)

        return np.array(temparray), np.array(matchedf), np.array(desf)

    def findResidualsForAllShifts(self):
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
        self.MAD_std = np.zeros(((2 * self.maxY + 1), (2 * self.maxX + 1)))
        self.std = np.zeros(((2 * self.maxY + 1), (2 * self.maxX + 1)))
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
        data = self.removeNaNsfromArray(self.bestMatchedFreqs[0])
        model = self.removeNaNsfromArray(self.bestMatchedFreqs[1])
        params = np.polyfit(model, data, self.order, full=True)[0]
        return params

    def makeLstSqResiduals(self, parameters):
        p = np.poly1d(parameters)
        data = self.removeNaNsfromArray(self.bestMatchedFreqs[0])
        model = self.removeNaNsfromArray(self.bestMatchedFreqs[1])
        error = data - p(model)
        return error

    def fitFrequencies(self):
        primaryGuess = self.guessInitialParams()
        self.leastSquaresSol = opt.least_squares(self.makeLstSqResiduals, primaryGuess)
        coeffs = np.poly1d(self.leastSquaresSol.x)
        self.fitDesign = coeffs(self.design)

    def countPixelsPlaced(self):
        counter = 0
        for i in self.newFeedline:
            if i[1] == 0 and np.isfinite(i[2]) and np.isfinite(i[3]):
                counter = counter + 1
        self.placedPixels = counter

    # def compareNearestNeighbors(self):
    #     self.nearestNeighborFreqLocation = np.full((len(self.newFeedline), 2), np.nan)
    #     for i in range(len(self.newFeedline)):
    #         if np.isfinite(self.newFeedline[i][2]) and np.isfinite(self.newFeedline[i][3]) and np.isfinite(self.newFeedline[i][4]):
    #             self.nearestNeighborFreqLocation[i][0] = self.findNearestNeighborFrequency(self.newFeedline[i])[1] - 1
    #             self.nearestNeighborFreqLocation[i][1] = -1 * (self.findNearestNeighborFrequency(self.newFeedline[i])[0] - 1)
    #
    #     xvals = self.nearestNeighborFreqLocation[:, 0]
    #     yvals = self.nearestNeighborFreqLocation[:, 1]
    #
    #     tl = len(np.where((xvals == -1) & (yvals == -1))[0])
    #     t = len(np.where((xvals == 0) & (yvals == -1))[0])
    #     tr = len(np.where((xvals == 1) & (yvals == -1))[0])
    #     l = len(np.where((xvals == -1) & (yvals == 0))[0])
    #     c = len(np.where((xvals == 0) & (yvals == 0))[0])
    #     r = len(np.where((xvals == 1) & (yvals == 0))[0])
    #     bl = len(np.where((xvals == -1) & (yvals == 1))[0])
    #     b = len(np.where((xvals == 0) & (yvals == 1))[0])
    #     br = len(np.where((xvals == 1) & (yvals == 1))[0])
    #     self.wellPlacedPixels = c
    #     self.totalPlacedPixels = tl + t + tr + l + c + r + bl + b + br
    #
    # def findNearestNeighborFrequency (self, resonator):
    #     resX = int(resonator[2]) % 14
    #     resXp1 = int(resX+1) % 14
    #     resXm1 = int(resX-1) % 14
    #     resY = int(resonator[3]) % 146
    #     resYp1 = int(resY+1) % 146
    #     resYm1 = int(resY-1) % 146
    #     nearestneighborfreqs = [[self.fitDesign[resYm1][resXm1], self.fitDesign[resYm1][resX], self.fitDesign[resYm1][resXp1]],
    #                             [self.fitDesign[resY][resXm1], self.fitDesign[resY][resX], self.fitDesign[resY][resXp1]],
    #                             [self.fitDesign[resYp1][resXm1], self.fitDesign[resYp1][resX], self.fitDesign[resYp1][resXp1]]]
    #     neighborresids = np.abs(nearestneighborfreqs - resonator[4])
    #     min_place = np.unravel_index(neighborresids.argmin(), neighborresids.shape)
    #     return min_place
    #
    # def plotNearestNeighborInfo(self):
    #     u = self.nearestNeighborFreqLocation[:, 0]
    #     v = self.nearestNeighborFreqLocation[:, 1]
    #
    #     tl = len(np.where((u == -1) & (v == -1))[0])
    #     t = len(np.where((u == 0) & (v == -1))[0])
    #     tr = len(np.where((u == 1) & (v == -1))[0])
    #     l = len(np.where((u == -1) & (v == 0))[0])
    #     c = len(np.where((u == 0) & (v == 0))[0])
    #     r = len(np.where((u == 1) & (v == 0))[0])
    #     bl = len(np.where((u == -1) & (v == 1))[0])
    #     b = len(np.where((u == 0) & (v == 1))[0])
    #     br = len(np.where((u == 1) & (v == 1))[0])
    #
    #     heights = np.array([[tl, t, tr], [l, c, r], [bl, b, br]])
    #     plt.imshow(heights, extent=[-heights.shape[1]/2., heights.shape[1]/2., -heights.shape[0]/2., heights.shape[0]/2. ])
    #     plt.colorbar()
    #     plt.show()
    #
    # def nearestNeighborQuiver(self):
    #     x = self.removeNaNsfromArray(self.newFeedline[:, 2])
    #     y = self.removeNaNsfromArray(self.newFeedline[:, 3])
    #     u = self.removeNaNsfromArray(self.nearestNeighborFreqLocation[:, 1])
    #     v = self.removeNaNsfromArray(self.nearestNeighborFreqLocation[:, 0])
    #     plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1)
    #     plt.show()


if __name__ == '__main__':

    designFlPath = '/mnt/data0/nswimmer/Repositories/mapcheckertesting/mec_feedline.txt'
    rawBM = Beammap()
    rawBM.load('/mnt/data0/nswimmer/Repositories/mapcheckertesting/beammapTestData/newtest/RawMapV1.txt')
    rawBM.loadFrequencies(r'/mnt/data0/nswimmer/Repositories/mapcheckertesting/beammapTestData/newtest/ps*.txt')
    shifter = BeammapShifter(designFlPath, rawBM, "mec")

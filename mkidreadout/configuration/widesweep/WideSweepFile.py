import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal
from scipy.interpolate import UnivariateSpline


def peaks(y, nsig, m=2, returnDict=False):
    """
    Find the peaks in a vector (spectrum) which lie nsig above the
    standard deviation of all peaks in the spectrum.

    y -- vector in which to locate peaks
    nsig -- number of sigma above the standard deviation of all peaks to search

    return -- vector holding indices of peak locations in y

    Intended to duplicate logic of peaks.pro

    nsig is NOT the number of sigma above the noise in the spectrum.
    It's instead a measure of the significance of a peak. First, all
    peaks are located. Then the standard deviation of the peaks is
    calculated using ROBUST_SIGMA (see Goddard routines online). Then
    peaks which are NSIG above the sigma of all peaks are selected.

    """

    print "begin peaks with nsig, m=", nsig, m
    d0 = y - np.roll(y, -1)
    d1 = y - np.roll(y, 1)
    pk = np.arange(y.size)[np.logical_and(d0 > 0, d1 > 0)]
    npk = pk.size
    yp = y[pk]
    # reject outliers more than m=2 sigma from median
    delta = np.abs(yp - np.median(yp))
    mdev = np.median(delta)
    s = delta / mdev if mdev else 0
    ypGood = y[np.where(s < m)]  # using a subset of y
    mn = ypGood.mean()
    sig = ypGood.std()
    big = pk[yp > mn + nsig * sig]

    # to remove multiple identifications of the same peak (not collisions)
    minPeakDist = 60
    cluster = []
    clusters = []
    for pks in range(len(big) - 1):
        dist = abs(big[pks] - big[pks + 1])
        cluster.append(pks)
        if dist > minPeakDist:
            clusters.append(cluster)
            cluster = []

    indrem = []
    for c in range(len(clusters)):
        try:
            trueind = np.argmax(y[big[clusters[c]]])
            falseind = np.where(big[clusters[c]] != big[clusters[c]][trueind])[0]
            indrem = np.concatenate((indrem, np.array(clusters[c])[falseind]))
        except ValueError:
            pass
    big = np.delete(big, indrem)

    if returnDict:
        return {"big": big, "pk": pk, "yp": yp, "m": m}
    else:
        return big


class WideSweepFile(object):
    """
    Handle data written by the program SegmentedSweep.vi
    
    The first seven lines are header information.
    Each remaining line is frequency, I, sigma_I, Q, sigma_Q

    """
    def __init__(self,fileName):
        file = open(fileName,'r')
        #(self.fr1,self.fspan1,self.fsteps1,self.atten1) = \
        #    file.readline().split()
        #(self.fr2,self.fspan2,self.fsteps2,self.atten2) = \
        #    file.readline().split()
        #(self.ts,self.te) = file.readline().split()
        #(self.Iz1,self.Izsd1) = [float(x) for x in file.readline().split()]
        #(self.Qz1,self.Qzsd1) = [float(x) for x in file.readline().split()]
        #(self.Iz2,self.Izsd2) = [float(x) for x in file.readline().split()]
        #(self.Qz2,self.Qzsd2) = [float(x) for x in file.readline().split()]
        (self.ts,self.te) = 0.100, 0.100
        (self.Iz1,self.Izsd1) = 0.000, 0.000
        (self.Qz1,self.Qzsd1) = 0.000, 0.000
        (self.Iz2,self.Izsd2) = 0.000, 0.000
        (self.Qz2,self.Qzsd2) = 0.000, 0.000
        file.close()
        try:
            self.data1 = np.loadtxt(fileName)
        except: self.data1 = np.loadtxt(fileName, skiprows=3)
        self.loadedFileName=fileName
        self.x = self.data1[:,0]
        self.n = len(self.x)
        ind = np.arange(self.n)
        Iz = np.where(ind<self.n/2, self.Iz1, self.Iz2)
        self.I = self.data1[:,1]
        self.I = self.I - Iz
        self.Ierr = 0.001000000 #self.data1[:,2]
        Qz = np.where(ind<self.n/2, self.Qz1, self.Qz2)
        self.Q = self.data1[:,2] - Qz
        self.Qerr = 0.001000000#self.data1[:,4]
        self.mag = np.sqrt(np.power(self.I,2) + np.power(self.Q,2))

    def fitSpline(self, splineS=1, splineK=3):
        x = self.data1[:,0]
        y = self.mag        
        self.splineS = splineS
        self.splineK = splineK
        spline = UnivariateSpline(x,y,s=self.splineS, k=self.splineK)
        self.baseline = spline(x)

    def findPeaks(self, m=2, useDifference=True):
        if useDifference:
            diff = self.baseline - self.mag
        else:
            diff = -self.mag            
        self.peaksDict = peaks(diff,m,returnDict=True)
        self.peaks = self.peaksDict['big']
        self.pk = self.peaksDict['pk']

    def findPeaksThreshold(self,threshSigma):
        self.fptThreshSigma = threshSigma
        values = self.mag-self.baseline
        self.fptHg = np.histogram(values,bins=100)
        self.fptCenters = 0.5*(self.fptHg[1][:-1] + self.fptHg[1][1:])
        self.fptAverage = np.average(self.fptCenters,weights=self.fptHg[0])
        self.fptStd = np.sqrt(np.average((self.fptCenters-self.fptAverage)**2, 
                                      weights=self.fptHg[0]))
        thresh = self.fptAverage - threshSigma*self.fptStd
        ind = np.arange(len(values))[values < thresh]
        self.threshIntervals = interval()
        for i in ind-1:
            self.threshIntervals = threshIntervals | interval[i-0.6,i+0.6]
        self.peaks = np.zeros(len(threshIntervals))

        iPeak = 0
        for threshInterval in self.threshIntervals:
            i0 = int(math.ceil(self.threshInterval[0]))
            i1 = int(math.ceil(self.threshInterval[1]))
            peak = np.average(self.x[i0:i1],weights=np.abs(values[i0:i1]))
            self.peaks[iPeak] = peak
            x0 = self.x[i0]
            x1 = self.x[i1]
            iPeak += 1

    def filter(self, order=4, rs=40, wn=0.1):
        b,a = signal.cheby2(order, rs, wn, btype="high", analog=False)
        self.filtered = signal.filtfilt(b,a,self.mag)

    def fitFilter(self, order=4, rs=40, wn=0.5):
        self.filter(order=order, rs=rs, wn=wn)
        self.baseline = self.mag-self.filtered

    def createPdf(self, pdfFile, deltaF=0.15, plotsPerPage=5):
        nx = int(deltaF*len(self.x)/(self.x.max()-self.x.min()))
        pdf_pages = PdfPages(pdfFile)
        db = 20*np.log10(self.mag/self.mag.max())
        startNewPage = True
        for i0 in range(0,len(self.x),nx):
            if startNewPage:
                fig = plt.figure(figsize=(8.5,11), dpi=100)
                iPlot = 0
                startNewPage = False
            iPlot += 1
            ax = fig.add_subplot(plotsPerPage, 1, iPlot)
            ax.plot(self.x[i0:i0+nx],db[i0:i0+nx])
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("S21(db)")
            if iPlot == plotsPerPage:
                startNewPage = True
                pdf_pages.savefig(fig)
        if not startNewPage:
            pdf_pages.savefig(fig)
        pdf_pages.close()

    def resFit(self,ind0,ind1):
        """
        Logic copied from MasterResonatorAnalysis/resfit.pro
        """
        if ind0 < len(self.x)/2:
            iZero = self.Iz1
            qZero = self.Qz1
        else:
            iZero = self.Iz2
            qZero = self.Qz2

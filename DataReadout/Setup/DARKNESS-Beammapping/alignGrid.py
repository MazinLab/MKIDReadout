import numpy as np
import scipy.ndimage as sciim
import matplotlib.pyplot as plt

class BMAligner:

    def __init__(self, beamListFn, usFactor=10):
        self.beamListFn = beamListFn
        self.usFactor = usFactor
        self.resIDs, self.flags, self.rawXs, self.rawYs = np.loadtxt(beamListFn, unpack=True)
        self.makeRawImage()

    def makeRawImage(self):
        self.rawImage = np.zeros((np.max(self.rawXs)*self.usFactor+1, np.max(self.rawYs)*self.usFactor+1))
        for i, resID in enumerate(self.resIDs):
            if self.flags[i] == 0:
                self.rawImage[round(self.rawXs[i]*self.usFactor), round(self.rawYs[i]*self.usFactor)] = 1

    def fftRawImage(self):
        self.rawImageFFT = np.abs(np.fft.fft2(self.rawImage))
        self.rawImageFreqs = [np.fft.fftfreq(self.rawImageFFT.shape[0]), np.fft.fftfreq(self.rawImageFFT.shape[1])]
    
    def findAngleAndScale(self):
        maxFiltFFT = sciim.maximum_filter(self.rawImageFFT, size=10)
        locMaxMask = (self.rawImageFFT==maxFiltFFT)
        
        maxInds = np.argsort(self.rawImageFFT, axis=None)
        maxInds = maxInds[::-1]

        # find four nonzero k-vectors
        nMaxFound = 0
        i = 0
        kvecList = np.zeros((2,2))
        while nMaxFound < 2:
            maxCoords = np.unravel_index(maxInds[i], self.rawImageFFT.shape)

            if locMaxMask[maxCoords]==1 and maxCoords!=(0,0):
                kvec = np.array([self.rawImageFreqs[0][maxCoords[0]], self.rawImageFreqs[1][maxCoords[1]]])
                if nMaxFound == 0:
                    kvecList[0] = kvec
                    nMaxFound += 1
                else:
                    if np.abs(np.dot(kvecList[0], kvec)) < 0.05*np.linalg.norm(kvec)**2:
                        kvecList[1] = kvec
                        nMaxFound += 1
                    

            i += 1
                
        xKvecInd = np.argmax(np.abs(kvecList[:,0]))
        yKvecInd = np.argmax(np.abs(kvecList[:,1]))

        assert xKvecInd!=yKvecInd, 'x and y kvecs are the same!'

        xKvec = kvecList[xKvecInd]
        yKvec = kvecList[yKvecInd]

        if xKvec[0]<0:
            xKvec *= -1

        if yKvec[1]<0:
            yKvec *= -1


        anglex = np.arctan2(xKvec[1], xKvec[0])
        angley = np.arctan2(yKvec[1], yKvec[0]) - np.pi/2

        assert (anglex - angley)/anglex < 0.01, 'x and y kvecs are not perpendicular!'

        self.angle = (anglex + angley)/2
        self.xScale = 1/(self.usFactor*np.linalg.norm(xKvec))
        self.yScale = 1/(self.usFactor*np.linalg.norm(yKvec))

        print 'angle:', self.angle
        print 'x scale:', self.xScale
        print 'y scale:', self.yScale
        
    def rotateAndScaleCoords(self):
        c = np.cos(-self.angle)
        s = np.sin(-self.angle)
        rotMat = np.array([[c, -s], [s, c]])
        rawCoords = np.stack((self.rawXs, self.rawYs))
        coords = np.dot(rotMat,rawCoords)
        coords = np.transpose(coords)
        coords[:,0] /= self.xScale
        coords[:,1] /= self.yScale
        self.coords = coords
        
    def plotCoords(self):
        plt.plot(self.coords[:,0], self.coords[:,1])
        plt.show()
            


                
            

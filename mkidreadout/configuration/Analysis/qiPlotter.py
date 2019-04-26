import os

import matplotlib.pyplot as plt
import numpy as np

mdd = os.environ['MKID_DATA_DIR']
beammapFileName = 'Beammap/finalMap_20170403.txt'
autofitFileName = 'Faceless_FL3-freqs-good-fits.txt'
xSize = 80
ySize = 25
upperQiLimit = 100000
feedline = 3
makeYAvgPlot = True

beammapFile = os.path.join(mdd,beammapFileName)
autofitFile = os.path.join(mdd,autofitFileName)
beammap = np.loadtxt(beammapFile)
autofit = np.loadtxt(autofitFile)

beammap = beammap[(feedline-1)*2000:feedline*2000]
beammap[:,3] = beammap[:,3] - (feedline-1)*ySize

qImage = np.zeros((xSize, ySize))
beammap[:,2:3] = np.floor(beammap[:,2:3])

beammappedResInd = np.where(beammap[:,1]==0)[0]
beammappedRes = beammap[beammappedResInd,:]
#print beammappedResInd

#find indices of beammapped resonators in autofit file
findRes = lambda resID: np.where(resID==autofit[:,1])[0][0]
#print np.array(autofit[:,1], dtype=np.int32)
beammappedResIndAutofit = np.asarray(map(findRes, beammappedRes[:,0]))
#print beammappedResIndAutofit
beammappedQis = autofit[beammappedResIndAutofit, 5]
print beammappedQis

for i in range(len(beammappedQis)):
    if 0<=beammappedRes[i,2]<xSize and 0<=beammappedRes[i,3]<ySize and 0<beammappedQis[i]:
        qImage[beammappedRes[i,2], beammappedRes[i,3]] = beammappedQis[i]

qImageZeros = np.where(qImage==0)
qImage = np.transpose(qImage)
im = plt.imshow(qImage, interpolation='nearest', clim=(0,upperQiLimit))
plt.colorbar(im, orientation = 'horizontal')
plt.scatter(qImageZeros[0], qImageZeros[1], marker=',', color='black', s=40) #mark unmapped resonators
plt.show()

if makeYAvgPlot:
    qImage = np.transpose(qImage)
    medQs = np.median(qImage, axis=1)
    plt.plot(medQs)
    plt.xlabel('X')
    plt.ylabel('Median Q')
    plt.show()

    qImage[qImageZeros] = np.nan
    medQs = np.nanmedian(qImage, axis=1)
    plt.plot(medQs)
    plt.title('Qi vs x, ignoring unbeammapped pixels')
    plt.xlabel('X')
    plt.ylabel('Median Q')
    plt.show()

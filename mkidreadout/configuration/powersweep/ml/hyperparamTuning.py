import numpy as np
from wpsnn import WPSNeuralNet
from mkidcore.readdict import ReadDict

initialConfig = '/home/neelay/data/20200121/wpsnn.cfg'
paramDict = {'trainSatThresh': [1, 2], 
             'trainUPThresh': [2, 3, 4], 
             'useIQV': [True, False], 
             'freqWinSize':[20, 30, 40], 
             'attenWinSize':[2, 3, 4], 
             #'num_filt1': [20, 30, 40], 
             #'num_filt2': [50, 70, 90],
             #'num_filt3': [70, 90, 120], 
             'conv_win1': [4, 5, 7], 
             'conv_win2': [[2, 5], [2, 7], [3, 5]], 
             'conv_win3': [[1, 3], [1, 4]]
             'n_pool1': [[1, 2], [2, 2], [2, 3]], 
             'n_pool2': [[1, 2], [2, 2]],
             'n_pool3': [[1, 2], [1, 3]]}

nOptIters = 20
paramSpaceShape = ()


for k, v in paramDict:
    paramSpaceShape += (len(v),)

paramSpaceMask = np.ones(paramSpaceShape)
mlDict = ReadDict(initialConfig)

for i in range(nOptIters)
    validCoords = np.array(np.where(paramSpaceMask))
    paramInd = np.random.randint(0, validCoords.shape[1])
    paramCoords = tuple(validCoords[:, paramInd])
    paramSpaceMask[paramCoords] = 0 #don't sample this point later
    print '--------------------------------'
    print 'Current iter {}/{}'.format(i, nOptIters)
    for j, (k, v) in enumerate(paramDict):
        mlDict.update({k: v[paramCoords[i]}) #update the current parameter with coordinate value chosen earlier
        print '{}: {}'.format(k, v)

    mlClass = WPSNeuralNet(mlDict)
    mlClass.initializeAndTrainModel()

def validateModel(modelDir):
    pass

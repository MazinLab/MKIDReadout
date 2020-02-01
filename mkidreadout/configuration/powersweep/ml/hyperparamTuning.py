import numpy as np
import mkidreadout.configuration.sweepdata as sd
from wpsnn import WPSNeuralNet
from mkidcore.readdict import ReadDict
import findResonatorsWPS as finder
from compareResLists import retrieveManResList, retrieveMLResList
from checkWPSPerformance import matchResonators, matchAttens

initialConfig = '/home/neelay/data/20200121/wpsnn.cfg'
rootDir = '/home/neelay/data/20200121/val0'
initParamFile = '/home/neelay/data/20200121/val0_params.txt'
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

rawTrainFiles0 = ['psData_222.npz', 'psData_223.npz', 'psData_228.npz', 'psData_229.npz', 'psData_232.npz', 'psData_233.npz', 'psData_238.npz', 'psData_239.npz']
rawTrainLabels0 = ['psData_222_metadata_out.txt', 'psData_223_metadata_out.txt', 'psData_228_metadata_out.txt', 'psData_229_metadata_out.txt', 'psData_232_metadata_out.txt', 'psData_233_metadata_out.txt', 'psData_238_metadata_out.txt', 'psData_239_metadata_out.txt']
rawTrainFiles1 = ['psData_224.npz', 'psData_225.npz', 'psData_228.npz', 'psData_229.npz', 'psData_232.npz', 'psData_233.npz', 'psData_238.npz', 'psData_239.npz']
rawTrainLabels1 = ['psData_224_metadata_out.txt', 'psData_225_metadata_out.txt', 'psData_228_metadata_out.txt', 'psData_229_metadata_out.txt', 'psData_232_metadata_out.txt', 'psData_233_metadata_out.txt', 'psData_238_metadata_out.txt', 'psData_239_metadata_out.txt']

valFiles0 = ['psData_224.npz', 'psData_225.npz']
valLabels0 = ['psData_224_metadata_out.txt', 'psData_225_metadata_out.txt']
valFiles1 = ['psData_222.npz', 'psData_223.npz']
valLabels1 = ['psData_222_metadata_out.txt', 'psData_223_metadata_out.txt']
valFileDir = '/home/neelay/data/20191113'

nOptIters = 5
paramSpaceShape = ()

for k, v in paramDict.items():
    paramSpaceShape += (len(v),)

with open(initParamFile) as f:
    for k, v in paramDict.items():
        f.write('{}: {}'.format(k, v)
    f.write('paramSpaceShape:', paramSpaceShape)

paramSpaceMask = np.ones(paramSpaceShape)
mlDict = ReadDict(initialConfig)

for i in range(nOptIters)
    validCoords = np.array(np.where(paramSpaceMask))
    paramInd = np.random.randint(0, validCoords.shape[1])
    paramCoords = tuple(validCoords[:, paramInd])
    paramSpaceMask[paramCoords] = 0 #don't sample this point later
    modelID = ''
    print '--------------------------------'
    print 'Current iter {}/{}'.format(i, nOptIters)
    for j, (k, v) in enumerate(paramDict):
        if k == 'attenWinSize':
            mlDict.update({'attenWinAbove': v[paramCoords[i]]})
            mlDict.update({'attenWinBelow': v[paramCoords[i]]})
        mlDict.update({k: v[paramCoords[i]]}) #update the current parameter with coordinate value chosen earlier
        print '   {}: {}'.format(k, v)
        modelID += str(paramCoords[i]) #concat param indices in paramdict

    #Crossval 0 train
    trainFileID = modelID[:5] + '_0' #rest of parameters are neural net specific
    modelName = modelID + '_0'
    modelDir = os.path.join(rootDir, modelName)
    trainNPZ = trainFileID + '.npz'
    mlDict.update({'modelDir': modelDir})
    mlDict.update({'modelName': modelName})
    mlDict.update({'trainNPZ': trainNPZ})
    mlDict.update({'rawTrainFiles': rawTrainFiles0})
    mlDict.update({'rawTrainLabels': rawTrainLabels0})

    mlClass = WPSNeuralNet(mlDict)
    mlClass.initializeAndTrainModel()

    #Crossval 1 train
    trainFileID = modelID[:5] + '_1' #rest of parameters are neural net specific
    modelName = modelID + '_1'
    modelDir = os.path.join(rootDir, modelName)
    trainNPZ = trainFileID + '.npz'
    mlDict.update({'modelDir': modelDir})
    mlDict.update({'modelName': modelName})
    mlDict.update({'trainNPZ': trainNPZ})
    mlDict.update({'rawTrainFiles': rawTrainFiles1})
    mlDict.update({'rawTrainLabels': rawTrainLabels1})

    mlClass = WPSNeuralNet(mlDict)
    mlClass.initializeAndTrainModel()
        
        


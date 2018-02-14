from PSFitNNModel import *
from readDict import readDict

if len(sys.argv)<2:
    print 'Must supply config file!'
    exit(1)

mlDictFile = sys.argv[1]
mlDict = readDict()
mlDict.readFromFile(mlDictFile)

mlClass = mlClassification(mlDict)
mlClass.initializeAndTrainModel()

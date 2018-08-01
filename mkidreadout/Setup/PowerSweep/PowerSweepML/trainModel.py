'''
Script to train ML model. Model architecture
is defined in PSFitNNModel, and parameters
and training data are defined in specified 
config file.
Usage: python trainNNModel.py <mlConfigFile>
Trained model is saved in .meta file in 
modelDir specified in config file.
'''

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

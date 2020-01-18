'''
Script to train ML model. Model architecture
is defined in PSFitNNModel, and parameters
and training data are defined in specified 
config file.
Usage: python trainNNModel.py <mlConfigFile>
Trained model is saved in .meta file in 
modelDir specified in config file.
'''

from wpsnn import *

from mkidcore.readdict import ReadDict

if len(sys.argv)<2:
    print 'Must supply config file!'
    exit(1)

mlDictFile = sys.argv[1]
mlDict = ReadDict()
mlDict.readFromFile(mlDictFile)

mlClass = WPSNeuralNet(mlDict)
mlClass.initializeAndTrainModel()

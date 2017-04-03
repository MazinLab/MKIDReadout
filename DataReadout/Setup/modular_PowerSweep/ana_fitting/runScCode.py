import sys, os
# from PSFitMLClass import *
# from Hal_wholeres import *
# from PSFit_mlImgCube3Layer2d import *
from PSFitMLFullRes import *
from PSFitSc import *
# import mScraps as ms


sys.path.append('/home/rupert/Documents/MkidDigitalReadout/MkidDigitalReadout/DataReadout/Setup/PowerSweep/')
from PSFitMLData_origPSFile import *

saveDir = 'ExampleData/PowerSweep/'
inferenceFile = 'ps_r7_100mK_a_20161016-155917'
inferenceData = PSFitMLData(h5File = saveDir+inferenceFile+'.h5', useAllAttens = True) 


initialFile = None
inferenceFile = None
if len(sys.argv) > 2:
    initialFileName = sys.argv[1]
    inferenceFileName = sys.argv[2]
    mdd = os.environ['MKID_DATA_DIR']
    initialFile = os.path.join(mdd,initialFileName)
    inferenceFile = os.path.join(mdd,inferenceFileName)
else:
    print "need to specify an initial and inference filename located in MKID_DATA_DIR"
    exit()


mlClass = mlClassification(initialFile=initialFile, inferenceFile=inferenceFile)

# mlClass.makeTrainData()

mlClass.mlClass()

mlClass.evaluateModel()

mlClass.findAtten(inferenceFile=inferenceFile, searchAllRes=True, showFrames=False, res_nums=50)
    # mlClass.inferenceData.savePSTxtFile(flag='x')  


import sys, os
from PSFitMLClass import *

if __name__ == "__main__":
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

    mlClass.findAtten(inferenceFile=inferenceFile, searchAllRes=False, showFrames=False, res_nums=50)
    mlClass.inferenceData.savePSTxtFile(flag='x')  


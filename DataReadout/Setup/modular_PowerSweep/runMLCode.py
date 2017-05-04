import sys, os
# from PSFitMLClass import *
# from Hal_wholeres import *
# from PSFit_mlImgCube3Layer2d import *
# from PSFitMLFullRes import *

# from .modular_PowerSweep import Hal_fullres as mlc
import matplotlib.pyplot as plt
import Hal_fullres as mlc
import hal_binary as mlb
import PSFitMLData as mld
import PSFitMLTools as mlt
from PCA import PCA
from ml_params import *
sys.path.insert(0, '/home/rupert/PythonProjects/MkidDigitalReadout/DataReadout/Setup/modular_PowerSweep/ana_fitting')
import PSFitSc as PSFitSc
import PSFitTools as pt
from an_params import *
import glob
from difflib import SequenceMatcher
from os import walk



# def get_mlData(h5File, trainFile):
#     if not os.path.isfile(mldir + trainFile):
#         print 'Could not find train file. Making new training images from initialFile'

#     # print rawTrainFiles[0]
#     for h5File in [rawTrainFiles[0]]:
#         h5File = os.path.join(mdd,h5File)
#         mlData = mld.PSFitMLData(h5File = h5File)
#         mlData.makeBinTrainData()

#     return mlData

def train_bin_NN():
    if not os.path.isfile(mldir + trainBinFile):
        print 'Could not find binary train file. Making new training images from initial h5File'
        
        # for h5File in [rawTrainFiles[0]]:
        h5File = rawTrainFiles[0]
        h5File = os.path.join(mdd,h5File)
        mlData = mld.PSFitMLData(h5File = h5File)
        # mlData.makeBinTrainData()

    mlClass = mlb.mlClassification()
    mlClass.train()

    return mlClass


def get_power_NN(h5File = None, PSFile=None):
    # print os.path.join(trainDir, modelDir)
    # ***************** move this to MLData ********************
    if not os.path.exists(os.path.join(trainDir, modelDir)):
        os.makedirs(os.path.join(trainDir, modelDir))

    print os.path.join(trainDir, modelDir, trainFile)
    if not os.path.isfile(os.path.join(trainDir, modelDir, trainFile)):
        print 'Could not find train file. Making new training images from initial h5File'

        if h5File == None:
            if len(rawTrainFiles) > 1:
                auto_append = True
            else:
                auto_append = False

            for h5File in rawTrainFiles: #[rawTrainFiles[0]]:
                h5File = os.path.join(mdd,h5File)
                mlData = mld.PSFitMLData(h5File, PSFile, auto_append = auto_append)
                mlData.makeTrainData(res_per_class)
        
        logFileName = trainFile[:-4] + '.log'
        try:
            logFile = open(os.path.join(trainDir, logFileName), 'w')
        except IOError:
            os.makedirs(trainDir)
            logFile = open(os.path.join(trainDir, logFileName), 'w')
        logFile.write('Training data made using:')
        for h5File in rawTrainFiles:
            logFile.write(h5File)
        
        logFile.write('res_per_class ' + str(res_per_class))
        logFile.write('level_train ' + str(level_train))
        logFile.close()
    # ***************** move this to MLData ********************

    kwargs = {'nClass': nClass,
    'xWidth' : xWidth,
    'modelDir': modelDir,
    'fully_connected': fully_connected}

    mlClass = mlc.mlClassification(**kwargs)

    kwargs = {'trainFile': trainFile,
    'batches': batches, 
    'trainReps': trainReps, 
    'plot_missed': plot_missed,
    'plot_confusion': plot_confusion,
    'max_learning_rate': max_learning_rate, 
    'min_learning_rate': min_learning_rate, 
    'decay_speed': decay_speed,
    'fully_connected': fully_connected,
    'plot_activations': plot_activations,
    'view_train': view_train,
    'do_PCA': do_PCA,
    'view_train_hist': view_train_hist}


    mlClass.getModel(**kwargs)

    return mlClass

def eval_model(mlClass):
    mlClass.evalModel()
 
    h5File = evalFile#rawTrainFiles[0]
    h5File = os.path.join(mdd,h5File)
    mlt.evaluateModel(mlClass, h5File)

    return mlClass

def eval_powers(mlClass):

    h5File = evalFile
    h5File = os.path.join(mdd,h5File)
    mlClass.findPowers(inferenceFile=h5File)
    
    # try:
    #     mlData
    # except UnboundLocalError:
    #     mlData = mld.PSFitMLData(h5File = h5File)
    #     # mlData.makeTrainData(res_per_class)
    #     mlData.loadRawTrainData()
    # mlData = mlt.get_opt_atten_from_ind(mlData, mlClass.atten_guess)
    # mlData.savePSTxtFile(flag='x-diff_model')  


    return mlClass.atten_guess
    
def power_PCA(mlData):
    PCA(mlData)

def compare_train_data():
    h5File = rawTrainFiles[1]
    h5File = os.path.join(mdd,h5File)
    PSFile1 = h5File[:-19] + '.txt' #'x-reduced.txt'
    PSFile2 = h5File[:-19] + 'x-reduced.txt'
    agreed_res = mlt.compare_train_data(h5File, PSFile1, PSFile2)
    new_data = mlt.reduce_PSFile(PSFile2, agreed_res)
    mlData = mld.PSFitMLData(h5File=h5File, PSFile=PSFile1)
    mlData.good_res = new_data[:,0]
    mlData.opt_freqs = new_data[:,1]
    mlData.opt_attens = new_data[:,2]
    mlData.savePSTxtFile(flag='man_agreed')  

def compare_man_ins():
    '''compare two manual inspection sets'''
    h5File = evalFile
    h5File = os.path.join(mdd,h5File)
    PSFile1 = h5File[:-19] + '.txt' #'x-reduced.txt'
    PSFile2 = h5File[:-19] + 'x-reduced.txt'
    
    _, opt_attens1, opt_attens2 = mlt.compare_train_data(h5File, PSFile1, PSFile2, get_power_NN, True)
    
    mlt.plot_confusion(opt_attens1, opt_attens2, 'Manual Inspection A Class', 'Manual Inspection B Class')

def compare_vIQ_man_ins():
    h5File = evalFile
    h5File = os.path.join(mdd,h5File)
    mlData = mld.PSFitMLData(h5File = h5File)
    mlData.loadRawTrainData() 

    vIQ_attens = mlt.eval_vIQ_attens(mlData)
    
    mlt.plot_confusion(mlData.opt_iAttens, vIQ_attens, 'Manual Inspection Class', 'IQ Velocity Class')

    return vIQ_attens

def compare_ana_NN():
    ps = PSFitSc.PSFitSc()

    # get resLists
    ps.resListsFile = cacheDir + 'resLists_' + inferenceFile + '.pkl'
    
    # resLists = ps.fitresLists(num_res =20)
    resLists = ps.loadresLists()

    # get analytical powers from resLists
    a_pwrs, a_ipwrs = ps.evaluatePwrsFromDetune(resLists)

    # get machine learning powers
    mlClass = get_power_NN()
    # ml_ipwrs = mlClass.atten_guess
    ml_ipwrs = eval_powers(mlClass)

    h5file = os.path.join(mdd,evalFile)
    rawTrainData = mld.PSFitMLData(h5File = h5file)
    rawTrainData.loadRawTrainData() # run again to get 

    # res remaining after manual flagging and cut to include up to res_nums
    ml_ipwrs=ml_ipwrs[rawTrainData.good_res]

    # get human clicked powers
    mc_mask, mc_ipwrs, mc_pwrs = pt.loadManualClickFile(ps.inferenceData,cutoff)

    # print np.shape(a_ipwrs), np.shape(ml_ipwrs), np.shape(mc_ipwrs)
    
    # mlt.plot_confusion(mc_ipwrs, ml_ipwrs, 'Manual Inspection Class', 'Neural Network Class')
    # mlt.plot_confusion(mc_ipwrs, a_ipwrs, 'Manual Inspection Class', 'Analytical Class')
    # mlt.plot_confusion(ml_ipwrs, a_ipwrs, 'Neural Network Class', 'Analytical Class' )
    

    # mlt.plot_powers_hist(a_ipwrs, ml_ipwrs, mc_ipwrs)

    return a_ipwrs

def compare_ana_man_ins():
    a_ipwrs = compare_ana_NN()
    vIQ_pwrs = compare_vIQ_man_ins()

    mlt.plot_confusion(vIQ_pwrs, a_ipwrs, 'IQ velocity Class', 'Analytical Class')

if __name__ == "__main__":
    # if do_bin_class:
    #     mlClass = train_bin_NN()
    #     eval_powers(mlClass)

    # if do_power_class:
    #     # h5File = rawTrainFiles[0]
    #     # h5File = os.path.join(mdd,h5File)
    #     # PSFile = h5File[:-19] + 'man_agreed.txt' #'x-reduced.txt'
    #     # mlClass = get_power_NN(PSFile)

    # mlClass = get_power_NN()
    # mlClass = eval_model(mlClass)
    # eval_powers(mlClass)   
     
    # compare_train_data()

    # compare_ana_NN()
    # hoarder()

    # compare_man_ins()
    # compare_vIQ_man_ins()

    compare_ana_man_ins()
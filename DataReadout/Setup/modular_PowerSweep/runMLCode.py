import sys, os
# from PSFitMLClass import *
# from Hal_wholeres import *
# from PSFit_mlImgCube3Layer2d import *
# from PSFitMLFullRes import *

# from .modular_PowerSweep import Hal_fullres as mlc
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
# mldir, batches, trainReps, plot_missed,  trainFile, trainBinFile, rawTrainFiles, mdd, do_bin_class, do_power_class, res_per_class, max_learning_rate, min_learning_rate, decay_speed 

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


def train_power_NN(PSFile=None):

    if not os.path.isfile(mldir + trainFile):
        print 'Could not find train file. Making new training images from initial h5File'

        for h5File in [rawTrainFiles[0]]:
            h5File = os.path.join(mdd,h5File)
            mlData = mld.PSFitMLData(h5File = h5File, PSFile=PSFile)
            mlData.makeTrainData(res_per_class)

    mlClass = mlc.mlClassification()

    kwargs = {'batches': batches, 
    'trainReps': trainReps, 
    'plot_missed': plot_missed,
    'plot_confusion': plot_confusion,
    'max_learning_rate': max_learning_rate, 
    'min_learning_rate': min_learning_rate, 
    'decay_speed': decay_speed,
    'fully_connected': fully_connected,
    'plot_activations': plot_activations}

    mlClass.train(**kwargs)
 
    # h5File = rawTrainFiles[0]
    # h5File = os.path.join(mdd,h5File)
    # mlt.evaluateModel(mlClass, h5File)
    return mlClass
    
def eval_powers(mlClass):
    h5File = rawTrainFiles[0]
    h5File = os.path.join(mdd,h5File)
    mlClass.findPowers(inferenceFile=h5File)
    # try:
    #     mlData
    # except UnboundLocalError:
    #     mlData = mld.PSFitMLData(h5File = h5File)
    #     # mlData.makeTrainData(res_per_class)
    #     mlData.loadRawTrainData()
    # mlData = mlt.get_opt_atten_from_ind(mlData, mlClass.atten_guess)
    # mlData.savePSTxtFile(flag='x')  

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

def compare_ana_NN():
    ps = PSFitSc.PSFitSc()

    # get resLists
    ps.resListsFile = cacheDir + 'resLists_' + inferenceFile + '.pkl'
    
    # resLists = ps.fitresLists(num_res =20)
    resLists = ps.loadresLists()

    # get analytical powers from resLists
    a_pwrs, a_ipwrs = ps.evaluatePwrsFromDetune(resLists)

    # get machine learning powers
    mlClass = train_power_NN()
    ml_ipwrs = eval_powers(mlClass)

    # get human clicked powers
    mc_mask, mc_ipwrs, mc_pwrs = pt.loadManualClickFile(ps.inferenceData,cutoff)

    print np.shape(a_ipwrs), np.shape(ml_ipwrs), np.shape(mc_ipwrs)
    mlt.plot_powers_hist(a_ipwrs, ml_ipwrs, mc_ipwrs)

if __name__ == "__main__":
    if do_bin_class:
        mlClass = train_bin_NN()
        eval_powers(mlClass)

    if do_power_class:
        # h5File = rawTrainFiles[1]
        # h5File = os.path.join(mdd,h5File)
        # PSFile = h5File[:-19] + 'man_agreed.txt' #'x-reduced.txt'
        # mlClass = train_power_NN(PSFile)

        mlClass = train_power_NN()
        eval_powers(mlClass)    
    # compare_train_data()

    #compare_ana_NN()

import os
import datetime

# data files
mdd = os.environ['MKID_DATA_DIR']
<<<<<<< HEAD
mldir = './Hal_fullres/model0/' 
trainDir = '../PowerSweep/mlTrainingData/'
trainFile = 'faceless_lf_train_fullres.pkl'
trainBinFile = 'faceless_lf_train_bin.pkl'
rawTrainFiles = ['20161016/ps_r7_100mK_a_20161016-155917.h5',
=======
# mldir = './cache/'
trainDir = './mlTrainingData'
trainFile = 'ps_train.pkl'
now = datetime.datetime.now()
modelDir = '0719-eval_20class'#'20160719-t&e'#'71%' #= now.strftime("%Y-%m-%d")

trainBinFile = 'ps_bin_train.pkl'

# rawTrainFiles = ['20170331/ps_r114_FL3_a_20170402-001946.h5']
# rawTrainFiles = ['20161016/ps_r7_100mK_a_20161016-155917.h5']
# rawTrainFiles = ['20160719/ps_r119_FL2_a_pos_20160719-214142.h5']

rawTrainFiles = ['20160106adr/ps_r4_20160106-154607.h5',
>>>>>>> a8581aac94b1876ee2faf16c1407c37d20740a9d
'20160712/ps_r115_FL1_1_20160712-225809.h5',
# '20160719/ps_r119_FL2_a_pos_20160719-214142.h5',
'20151214adr/ps_r4_20151216-144054.h5',
'20161016/ps_r7_100mK_a_20161016-155917.h5',
'20170406/ps_r115_FL2_a_20170406-151938.h5',
'20170407/ps_r115_FL2_a_20170407-011230.h5',
'20170402/ps_r114_FL3_b_20170401-233131.h5',
'20170402/ps_r114_FL3_a_20170402-001946.h5',
'20170331/ps_r114_FL3_a_20170402-001946.h5', 
'20170331/ps_r114_FL3_b_20170401-202049.h5',
'20170401/ps_r114_FL3_b_20170401-233131.h5',
'20161016/ps_r7_100mK_a_20161016-155917.h5']

# '20161016_fake/ps_r7_100mK_a_20161016-155917.h5',

<<<<<<< HEAD
# neural network hyperparameters
trainReps = 200
batches = 50
testFrac = 0.1
max_nClass = 31
=======
#'20160719/ps_r119_FL2_b_pos_20160720-034349.h5',
#20170331/ps_r118 # probably not

# evalFile = '20161016/ps_r7_100mK_a_20161016-155917.h5' 
# evalFile = '20170406/ps_r115_FL2_a_20170406-151938.h5'
evalFile = '20160719/ps_r119_FL2_a_pos_20160719-214142.h5'

# training parameters
>>>>>>> a8581aac94b1876ee2faf16c1407c37d20740a9d
res_per_class = 50
xWidth = 50
level_train = True
trainReps = 700
batches = 50
testFrac = 0.05
max_learning_rate = 0.05
min_learning_rate = 0.0005#0.0001
decay_speed = 200

# neural network hyperparameters
max_nClass = 20
nClass = max_nClass
fully_connected = False
recursive = False

# script actions
do_bin_class = False
do_power_class = True
fully_connected = False
recursive = False

# debug
view_train = False
plot_missed = False
do_PCA = False
view_train_hist = False
res_per_win = 4
plot_confusion = False # plot confusion on the test data
plot_activations = ''
plot_weights = ''

# plot_accuracy =True




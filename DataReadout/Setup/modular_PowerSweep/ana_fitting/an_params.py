import numpy as np

saveDir = '/home/rupert/Documents/Widesweep_data/20161016/'
inferenceFile = 'ps_r7_100mK_a_20161016-155917'

# saveDir = '/home/rupert/Documents/Hal_train_data/20160719/'
# inferenceFile = 'ps_r119_FL2_a_pos_20160719-214142'

# saveDir = '/home/rupert/Documents/Widesweep_data/20160407/'
# inferenceFile = ''

cutoff = -1
ml_dir = '/home/rupert/PythonProjects/MkidDigitalReadout/MkidDigitalReadout/DataReadout/Setup/modular_PowerSweep/'
cacheDir = saveDir + 'ana_cache/'
force_a_trend = 100

bifur_thresh = 0.5

# synthetic params
nAttens = 20 #15
pwr_start = 50
num_fakes = 300
good_res = np.asarray(
    [50, 49, 48, 47, 46, 45, 41, 40, 36, 35, 34, 28, 27, 26, 25, 24, 23, 22, 21, 18, 17, 11, 9, 7, 4, 3, 2, 1])
doubles = np.asarray([3, 4, 7, 9, 11, 13, 17, 24, 29, 38, 39, 52, 53, 69, 87, 88, 91, 114, 119, 140, 143, 150, 171, 189, 190, 233, 239, 250, 251, 261, 280, 284, 293, 295, 303, 312, 327, 328, 335, 336, 340, 358, 359, 370, 382, 393, 394, 400, 402, 411, 424, 441, 451, 468, 474, 482, 483, 522, 527, 552, 559, 569, 598, 608, 610, 616, 625, 626, 633, 636, 645, 649, 652, 653, 667])
wind_band_left = 4e5
wind_band_right= 1e6
min_qi = 4e4
freq_samples = 75
a_thresh = 0.5
r_thresh = 1.5

#
# # span = [0,5100]
# fspan = [4, 4.05] #[0,-1]#[4,4.1]
# wind_width = 20
# end_score = -0.2#-1.5
# datadir = '../../Data/WideSweep/'
# rawsweepfile = 'WS_FL1.txt'
# manpeakfile = 'WS_FL1-freqs-good.txt'
# fitpeakfile = 'WS_FL1-fit-span%3.3f-%3.3f.txt' % (fspan[0],fspan[1])
# print fitpeakfile
# splineS_factor = 0.01
#
# train_raw_sweep_file = 'WS_FL2.txt'
# train_man_peak_file = 'WS_FL2-freqs-good.txt'
# delta_left, delta_right
import numpy as np

saveDir = '/home/dodkins/PythonProjects/ReadTune/Data/PowerSweep/'
inferenceFile = 'ps_r7_100mK_a_20161016-155917'
cacheDir = saveDir + 'cache/'
cutoff = 50
ml_dir = '/home/dodkins/PythonProjects/MkidDigitalReadout/MkidDigitalReadout/DataReadout/Setup/PowerSweep/'
force_a_trend = 3

# synthetic params
nAttens = 15
pwr_start = 50
num_fakes = 300
good_res = np.asarray(
    [50, 49, 48, 47, 46, 45, 41, 40, 36, 35, 34, 28, 27, 26, 25, 24, 23, 22, 21, 18, 17, 11, 9, 7, 4, 3, 2, 1])
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
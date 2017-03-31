# span = [0,5100]
fspan = [0,-1]#[4,4.1]# [4, 4.05] #
wind_width = 20
end_score = -0.2#-1.5
# datadir = '../../Data/WideSweep/'
datadir = '/home/rupert/Documents/Widesweep_data/some_date/'
rawsweepfile = 'WS_FL1.txt'
manpeakfile = 'WS_FL1-freqs-good.txt'
fitpeakfile = 'WS_FL1-fit-span%3.3f-%3.3f.txt' % (fspan[0],fspan[1])
baseFile = ('.').join(rawsweepfile.split('.')[:-1])
mlFile = datadir + baseFile + '-ml.txt'
print mlFile
splineS_factor = 0.01

train_raw_sweep_file = 'WS_FL2.txt'
train_man_peak_file = 'WS_FL2-freqs-good.txt'
delta_left, delta_right = 21, 21 # 19, 9  # # 18, 25  # 20, 42  # 15, 29  # 9, 17
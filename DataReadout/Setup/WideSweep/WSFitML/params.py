# span = [0,5100]
fspan = [0,-1]#[4,4.1]# [4, 4.05] #
wind_width = 20
end_score = -0.2#-1.5
# datadir = '../../Data/WideSweep/'
datadir = '/home/rupert/Documents/Widesweep_data/20170331/'
rawsweepfile = 'Faceless_FL4.txt'
# manpeakfile = 'WS_FL1-freqs-good.txt'
# fitpeakfile = 'WS_FL1-fit-span%3.3f-%3.3f.txt' % (fspan[0],fspan[1])
baseFile = ('.').join(rawsweepfile.split('.')[:-1])
mlFile = datadir + baseFile + '-ml.txt'
print mlFile
splineS_factor = 0.01

train_raw_sweep_files = ['Faceless_FL2.txt']#,'Faceless_FL3.txt','Faceless_FL4.txt']#'Faceless_FL1.txt',
train_man_peak_files = ['Faceless_FL2-freqs-all.txt']#, 'Faceless_FL3-freqs-all.txt', 'Faceless_FL4-freqs-all.txt']# 'Faceless_FL1-freqs-all.txt'
delta_left, delta_right = 21, 21# # 18, 25  # 20, 42  # 15, 29  # 9, 17 # 21, 21 # 19, 9  
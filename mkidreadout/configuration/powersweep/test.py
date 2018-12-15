import sys

sys.path.insert(0,'/Users/dodkins/PythonProjects/mkidreadout')
sys.path.insert(0,'/Users/dodkins/PythonProjects/mkidcore')

import mkidreadout.configuration.sweepdata as sweepdata
from mkidreadout.configuration.powersweep import psmldata

psfile = '/Users/dodkins/Scratch/MEC/20181212/psData2_222.npz'
psmetafile = '/Users/dodkins/Scratch/MEC/20181212/psData2_222_metadata.txt'

metadata_out = sweepdata.SweepMetadata(file=psmetafile)

fsweepdata = psmldata.MLData(fsweep=psfile, mdata=metadata_out)
widesweep = fsweepdata.freqSweep.oldwsformat(65)

print(fsweepdata.iq_vels.shape)

# res.load_params(cmplxIQ_params)
# res.do_lmfit(cmplxIQ_fit)


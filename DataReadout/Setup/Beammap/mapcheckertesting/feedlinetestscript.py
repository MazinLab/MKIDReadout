from MKIDDigitalReadout.DataReadout.Setup.Beammap.mapcheckertesting import mapchecker
from MKIDDigitalReadout.DataReadout.Setup.Beammap.mapcheckertesting import feedline
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.optimize as opt

# This is for development on a local machine, in full use this will just point to wherever beammap test data is stored
# REMINDER NOTE: Find where the feedline design frequency file is stored on Dark to correctly use that path
noah_design_feedline_path = r"C:\Users\njswi\PycharmProjects\BeammapPredictor\predictor\mec_feedline.txt"
design_feedline=np.loadtxt(noah_design_feedline_path)
noah_beammap_path = r"C:\Users\njswi\PycharmProjects\BeammapPredictor\predictor\beammapTestData\test\finalMap_20180605.txt"
noah_freqsweeps_path = r"C:\Users\njswi\PycharmProjects\BeammapPredictor\predictor\beammapTestData\test\ps_*"


# Initialize all of the feedlines
# NOTE: The first time running this code block on an array will take 5-10 minutes (assigning the resIDs takes a while
# to read in) but after that and the .npy file with all of the array data in it is created, it should take ~0.01 seconds
starttime = time.time()
feedline1 = feedline.Feedline(noah_design_feedline_path, noah_beammap_path, noah_freqsweeps_path, 1)
# feedline2 = feedline.Feedline(noah_design_feedline_path, noah_beammap_path, noah_freqsweeps_path, 2)
# feedline3 = feedline.Feedline(noah_design_feedline_path, noah_beammap_path, noah_freqsweeps_path, 3)
# feedline4 = feedline.Feedline(noah_design_feedline_path, noah_beammap_path, noah_freqsweeps_path, 4)
feedline5 = feedline.Feedline(noah_design_feedline_path, noah_beammap_path, noah_freqsweeps_path, 5)
feedline6 = feedline.Feedline(noah_design_feedline_path, noah_beammap_path, noah_freqsweeps_path, 6)
feedline7 = feedline.Feedline(noah_design_feedline_path, noah_beammap_path, noah_freqsweeps_path, 7)
feedline8 = feedline.Feedline(noah_design_feedline_path, noah_beammap_path, noah_freqsweeps_path, 8)
feedline9 = feedline.Feedline(noah_design_feedline_path, noah_beammap_path, noah_freqsweeps_path, 9)
feedline10 = feedline.Feedline(noah_design_feedline_path, noah_beammap_path, noah_freqsweeps_path, 10)
endtime = time.time()


print('It took {0:1.4f}'.format(endtime-starttime),'seconds to make your feedlines')

# Create a feedline array, like the note above, this should be an automatic process, but for now it's done here
feedlinearray=np.array([feedline1, feedline5, feedline6, feedline7, feedline8, feedline9, feedline10])


rd, fd, md, resids = mapchecker.leastsquaremethod(feedlinearray[-1], design_feedline, 3)

plt.subplots(2,1)
plt.suptitle('Feedline 10')
plt.subplot(211)
plt.scatter(md, rd, label='measured data', marker='.', c='k')
plt.scatter(md, fd, label='fitted model data', marker='.', c='b')
plt.scatter(md, md, label='unfitted model data', marker='.', c='r')
plt.legend()
plt.ylabel('Measured Frequency (MHz)')
plt.subplot(212)
plt.scatter(md, resids, label='residuals from fitted data', marker='.', c='b')
plt.scatter(md, rd-md, label='residuals from unfit data', marker='.', c='r')
plt.legend()
plt.xlabel('Design Frequency (MHz)')
plt.ylabel('Residual Distance (MHz)')
plt.show()

plt.figure(2)
plt.hist((rd-md), bins=25, label='residuals from unfit data', alpha=0.7, color='red')
plt.hist(resids, bins=25, label='residuals from fitted data', alpha=0.7, color='blue')
plt.legend()
plt.xlabel('Residual Distance (MHz)')
plt.ylabel('Counts')
plt.title('Feedline 10')
plt.show()
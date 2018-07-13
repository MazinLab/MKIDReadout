
# coding: utf-8
'''
 FLAG DEFINITIONS
 0: Good
 1: Pixel not read out
 2: Beammap failed to place pixel
 3: Succeeded in x, failed in y
 4: Succeeded in y, failed in x
 5: Multiple locations found for pixel
 6: Beammap placed the pixel in the wrong feedline
'''

import numpy as np
from MKIDDigitalReadout.DataReadout.Setup.Beammap.mapcheckertesting import feedline
import matplotlib.pyplot as plt
import time

# THESE PATH VARIABLES ARE FOR LOCAL DEVELOPMENT ONLY, DO NOT USE
noah_design_feedline_path = r"C:\Users\njswi\PycharmProjects\BeammapPredictor\predictor\mec_feedline.txt"
# noah_feedline_file_name = "finalMap_20180605"
noah_beammap_path = r"C:\Users\njswi\PycharmProjects\BeammapPredictor\predictor\beammapTestData\test\finalMap_20180605.txt"
noah_freqsweeps_path = r"C:\Users\njswi\PycharmProjects\BeammapPredictor\predictor\beammapTestData\test\ps_*"

design_feedline=np.loadtxt(noah_design_feedline_path)

print(time.time())
fl9 = feedline.Feedline(noah_design_feedline_path, noah_beammap_path, noah_freqsweeps_path, 9)
print(time.time())
fl8 = feedline.Feedline(noah_design_feedline_path, noah_beammap_path, noah_freqsweeps_path, 8)
print(time.time())


# sarah_design_feedline_path = np.loadtxt("")
# sarah_feedline_file_name = ""
# sarah_beammap_path = np.loadtxt(r"")
# saraah_freqsweeps_path = glob.glob("")


# feedline1 = feedline.Feedline(noah_design_feedline_path, noah_beammap_path, noah_freqsweeps_path, 1)
# feedline5 = feedline.Feedline(noah_design_feedline_path, noah_beammap_path, noah_freqsweeps_path, 5)
# feedline6 = feedline.Feedline(noah_design_feedline_path, noah_beammap_path, noah_freqsweeps_path, 6)
# feedline7 = feedline.Feedline(noah_design_feedline_path, noah_beammap_path, noah_freqsweeps_path, 7)
# feedline8 = feedline.Feedline(noah_design_feedline_path, noah_beammap_path, noah_freqsweeps_path, 8)
# feedline9 = feedline.Feedline(noah_design_feedline_path, noah_beammap_path, noah_freqsweeps_path, 9)
# feedline10 = feedline.Feedline(noah_design_feedline_path, noah_beammap_path, noah_freqsweeps_path, 10)


'''
Come back to this function later if necessary, used during development to analyze a full array (as opposed to only feedlines)

Create an array map where each element is the design freqency at a given pixel coordinate
design_array=np.ndarray((146,140))
for i in range(len(design_feedline)):
    for j in range(len(design_feedline[i])):
        design_array[i][j]=design_feedline[i][j]
        design_array[i][j+14]=design_feedline[i][j]
        design_array[i][j+2*14]=design_feedline[i][j]
        design_array[i][j+3*14]=design_feedline[i][j]
        design_array[i][j+4*14]=design_feedline[i][j]
        design_array[i][j+5*14]=design_feedline[i][j]
        design_array[i][j+6*14]=design_feedline[i][j]
        design_array[i][j+7*14]=design_feedline[i][j]
        design_array[i][j+8*14]=design_feedline[i][j]
        design_array[i][j+9*14]=design_feedline[i][j]
'''

'''
Written before the feedline class was created, left until feedline class definitively works
Create an array where each element has the measured values from that

final_map_array=np.zeros((len(design_array),len(design_array[0]),5))
for n in range(len(beammap)):
    final_map_array[int(beammap[n][3])][int(beammap[n][2])][0]=beammap[n][0]
    final_map_array[int(beammap[n][3])][int(beammap[n][2])][2]=beammap[n][1]
    final_map_array[int(beammap[n][3])][int(beammap[n][2])][3]=beammap[n][2]
    final_map_array[int(beammap[n][3])][int(beammap[n][2])][4]=beammap[n][3]
'''

'''
As above, this was moved to the feedline class
def AssignFrequency(FinalMap, FreqSweeps):
    counter=0
    array = np.copy(FinalMap)
    freqarr = np.loadtxt(FreqSweeps[0])
    for i in range(len(FreqSweeps)- 1):
        sweep = np.loadtxt(FreqSweeps[i+1])
        freqarr = np.concatenate((freqarr, sweep))
    frequencies = freqarr[:,1]
    for i in range(len(FinalMap)):
        for j in range(len(FinalMap[0])):
            ResID = array[i][j][0]
            index = quicklook(ResID, freqarr)
            if index != -1 and int(FinalMap[i][j][2])==0:
                array[i][j][1] = freqarr[index][1]/(10.**6)
                counter=counter+1
            else:
                array[i][j][1] = 0
    return array,counter
'''

'''
Created for the feedline class, will be REMOVED in a further update

good_array,counter=AssignFrequency(final_map_array,FreqSweepFiles)


def feedlineassigner (model, measured_array, FL_number):
    measured=np.copy(measured_array)
    fl_num=FL_number
    array=np.zeros((len(model),len(model[0]),6))
    counter=0
    for i in range(len(model)):
        for j in range(len(model[0])):
            array[i][j][:5]=measured[i][(len(measured[0])-14*fl_num)+j]
            if int(measured[i][(len(measured[0])-14*fl_num)+j][2])==0 and measured[i][(len(measured[0])-14*fl_num)+j][1] != 0 :
                array[i][j][5]=measured[i][(len(measured[0])-14*fl_num)+j][1]
            else :
                array[i][j][5]=float('NaN')
    return array


def get_feedline(model, measured_array,FL_number,flag):
    measured=np.copy(measured_array)
    fl_num=FL_number
    residuals = np.zeros((len(model), len(model[0])))
    measuredfreqs = np.zeros((len(model), len(model[0])))
    flag_map = np.zeros((len(model), len(model[0])))
    counter=0
    if flag == 0:
        minfreq=5000
        for i in range(len(model)):
            for j in range(len(model[0])):
                if int(measured[i][(len(measured[0])-14*fl_num)+j][1]) != 0 :
                    if measured[i][(len(measured[0])-14*fl_num)+j][1] <= minfreq :
                        minfreq = np.copy(measured[i][(len(measured[0])-14*fl_num)+j][1])
        for i in range(len(model)):
            for j in range(len(model[0])):
                if int(measured[i][(len(measured[0])-14*fl_num)+j][1]) != 0 :
                    residuals[i][j] = (measured[i][(len(measured[0])-14*fl_num)+j][1] - minfreq) - model[i][j]
                    measuredfreqs[i][j] = (measured[i][(len(measured[0])-14*fl_num)+j][1] - minfreq)
                    counter=counter+1
                else :
                    residuals[i][j]=float('NaN')
                    measuredfreqs[i][j]=float('NaN')
                flag_map[i][j]=measured[i][(len(measured[0])-14*fl_num)+j][2]
        return residuals, measuredfreqs, flag_map, counter
    else :
        for i in range(len(model)):
            for j in range(len(model[0])):
                if int(measured[i][(len(measured[0])-14*fl_num)+j][1]) != 0 :
                    measuredfreqs[i][j] = measured[i][(len(measured[0])-14*fl_num)+j][1]
                    counter=counter+1
                else :
                    residuals[i][j]=float('NaN')
                    measuredfreqs[i][j]=float('NaN')
                flag_map[i][j]=measured[i][(len(measured[0])-14*fl_num)+j][2]
        return measuredfreqs, flag_map, counter
'''


# This needs to be restructured to account for the data output of the feedline class
# Additionally should be utilized so that we can perform MCMC on the 'shifting' and 'stretching'
# parameters in frequency space
# Essentially, revamp this to turn it into something more like a chi squared function
def residual_std_dev_finder(residual_map, flag_map, measured_map):
    resids=np.copy(residual_map)
    flags=np.copy(flag_map)
    meas_vals=np.copy(measured_map)
    resid_vals_w_outliers=[]
    for i in range(len(resids)):
        for j in range(len(resids[i])):
            if flags[i][j]==0 and meas_vals[i][j]!=0 and np.isnan(meas_vals[i][j])==False:
                resid_vals_w_outliers.append(resids[i][j])
    resid_vals=np.array(resid_vals_w_outliers)
    std_dev_w_outliers=np.std(resid_vals_w_outliers)

    cutoff=std_dev_w_outliers*3
    indices=[]
    
    for i in range(len(resid_vals)):
        if resid_vals_w_outliers[i] >= cutoff or resid_vals_w_outliers[i] <= -1*cutoff:
            indices.append(i)
    resid_vals=np.delete(resid_vals_w_outliers,np.array(indices))
    
    std_dev=np.std(resid_vals)
    
    average_with_outliers=np.average(resid_vals_w_outliers)
    average=np.average(resid_vals)

    return std_dev, resid_vals, average, std_dev_w_outliers, resid_vals_w_outliers, average_with_outliers

# Find the range of the frequencies we've measured and then stretch them to be 4 GHz
# This should be rewritten to take in a stretching parameter so we can vary the stretch
def frequency_stretcher (frequency_array, minimum_frequency, maximum_frequency):
    min_f=minimum_frequency
    max_f=maximum_frequency
    norm_f=np.copy(frequency_array)-min_f
    stretched_f=norm_f*(4000./(max_f-min_f))
    return stretched_f

# Unworked, little tested, must be further developed
def frequency_shifter(streched_array,design_array):
    shift_array=np.zeros(((len(streched_array),len(streched_array[0]),2)))
    d_array = np.copy(design_array)
    for i in range(len(streched_array)):
        for j in range(len(streched_array[i])):
            if np.isnan(streched_array[i][j])==False:
                temp_array=np.abs(d_array-streched_array[i][j])
                index=np.unravel_index(temp_array.argmin(),temp_array.shape)
                shift_array[i][j][0]=index[0]-i
                shift_array[i][j][1]=index[1]-j

            else :
                shift_array[i][j][0],shift_array[i][j][1]=float('NaN'),float('NaN')

    return shift_array
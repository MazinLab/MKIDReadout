import numpy as np
from numpy import *
import math
import os
import PSFitMLData as mld
import Hal_fullres as mlc
from matplotlib import pylab as plt
from matplotlib import cm
import matplotlib.colors
# from params import mldir, trainFile, max_nClass, res_per_win, rawTrainFiles
from ml_params import *

def makeBinResImage(mlData, res_num, angle=0, phase_normalise=False, showFrames=False, dataObj=None):
    '''Creates a table with 3 rows: I, Q, and vel_iq for makeTrainData()

    inputs 
    res_num: index of resonator in question
    iAtten: index of attenuation in question
    angle: angle of rotation about the origin (radians)
    showFrames: pops up a window of the frame plotted using matplotlib.plot
    '''     
    # if dataObj is None:
    #     if mlClass.inferenceData is None:
    #         raise ValueError('Initialize dataObj first!')
    #     dataObj = mlClass.inferenceData
    
    xWidth= mlData.xWidth 

    loops=[]
    for iAtten in range(mlData.nClass):
        xCenter = argmax(mlData.iq_vels_orig[res_num,iAtten,:])
        start = int(xCenter - xWidth/2)
        end = int(xCenter + xWidth/2)

        if start < 0:
            start_diff = abs(start)
            start = 0
            iq_vels = mlData.iq_vels_orig[res_num, iAtten, start:end]
            iq_vels = np.lib.pad(iq_vels, (start_diff,0), 'constant', constant_values=(0))
            Is = mlData.Is_orig[res_num,iAtten,start:end]
            Is = np.lib.pad(Is, (start_diff,0), 'constant', constant_values=(Is[0]))
            Qs = mlData.Qs_orig[res_num,iAtten,start:end]
            Qs = np.lib.pad(Qs, (start_diff,0), 'constant', constant_values=(Qs[0]))
        elif end >= np.shape(mlData.freqs)[1]:
            iq_vels = mlData.iq_vels_orig[res_num, iAtten, start:end]
            iq_vels = np.lib.pad(iq_vels, (0,end-np.shape(mlData.freqs)[1]+1), 'constant', constant_values=(0))
            Is = mlData.Is_orig[res_num,iAtten,start:end]
            Is = np.lib.pad(Is, (0,end-np.shape(mlData.freqs)[1]), 'constant', constant_values=(Is[-1]))
            Qs = mlData.Qs_orig[res_num,iAtten,start:end]
            Qs = np.lib.pad(Qs, (0,end-np.shape(mlData.freqs)[1]), 'constant', constant_values=(Qs[-1]))
        else:
            iq_vels = mlData.iq_vels_orig[res_num, iAtten, start:end]
            Is = mlData.Is_orig[res_num,iAtten,start:end]
            Qs = mlData.Qs_orig[res_num,iAtten,start:end]
        #iq_vels = np.round(iq_vels * xWidth / max(mlData.iq_vels[res_num, iAtten, :]) )
        
        iq_vels = iq_vels / np.amax(mlData.iq_vels[res_num, :, :])
        res_mag = math.sqrt(np.amax(mlData.Is[res_num, :, :]**2 + mlData.Qs[res_num, :, :]**2))
        Is = Is / res_mag
        Qs = Qs / res_mag

        # Is = Is /np.amax(mlData.iq_vels[res_num, :, :])
        # Qs = Qs /np.amax(mlData.iq_vels[res_num, :, :])

        # Is = Is /np.amax(mlData.Is[res_num, :, :])
        # Qs = Qs /np.amax(mlData.Qs[res_num, :, :])

        if angle != 0:
            rotMatrix = np.array([[np.cos(angle), -np.sin(angle)], 
                                     [np.sin(angle),  np.cos(angle)]])

            Is,Qs = np.dot(rotMatrix,[Is,Qs])

        image = np.zeros((len(Is),3))
        image[:,0] = Is
        image[:,1] = Qs
        image[:,2] = iq_vels

        loops.append(image)

    if max_nClass != mlData.nClass:
        padding = np.zeros((max_nClass-len(loops),mlData.xWidth,3))
        loops = np.concatenate([loops,padding],axis=0)


    loops = loops * np.ones((len(loops),mlData.xWidth,3))
    if showFrames:
        plot_res(loops)

    return loops

def makeResImage(mlData, res_num, angle=0, pert=0, scale = 1, phase_normalise=False, showFrames=False, dataObj=None):
    '''Creates a table with 3 rows: I, Q, and vel_iq for makeTrainData()

    inputs 
    res_num: index of resonator in question
    iAtten: index of attenuation in question
    angle: angle of rotation about the origin (radians)
    showFrames: pops up a window of the frame plotted using matplotlib.plot
    '''     
    # if dataObj is None:
    #     if mlClass.inferenceData is None:
    #         raise ValueError('Initialize dataObj first!')
    #     dataObj = mlClass.inferenceData
    
    xWidth= mlData.xWidth 

    loops=[]
    # for iAtten in range(mlData.nClass):
    for iAtten in range(max_nClass):
        xCenter = get_peak_idx(mlData,res_num,iAtten)
        if pert != 0:
            xCenter = xCenter + pert
        start = int(xCenter - xWidth/2)
        end = int(xCenter + xWidth/2)

        if start < 0:
            start_diff = abs(start)
            start = 0
            iq_vels = mlData.iq_vels[res_num, iAtten, start:end]
            iq_vels = np.lib.pad(iq_vels, (start_diff,0), 'constant', constant_values=(0))
            Is = mlData.Is[res_num,iAtten,start:end]
            Is = np.lib.pad(Is, (start_diff,0), 'constant', constant_values=(Is[0]))
            Qs = mlData.Qs[res_num,iAtten,start:end]
            Qs = np.lib.pad(Qs, (start_diff,0), 'constant', constant_values=(Qs[0]))
        elif end >= np.shape(mlData.freqs)[1]:
            iq_vels = mlData.iq_vels[res_num, iAtten, start:end]
            iq_vels = np.lib.pad(iq_vels, (0,end-np.shape(mlData.freqs)[1]+1), 'constant', constant_values=(0))
            Is = mlData.Is[res_num,iAtten,start:end]
            Is = np.lib.pad(Is, (0,end-np.shape(mlData.freqs)[1]), 'constant', constant_values=(Is[-1]))
            Qs = mlData.Qs[res_num,iAtten,start:end]
            Qs = np.lib.pad(Qs, (0,end-np.shape(mlData.freqs)[1]), 'constant', constant_values=(Qs[-1]))
        else:
            iq_vels = mlData.iq_vels[res_num, iAtten, start:end]
            Is = mlData.Is[res_num,iAtten,start:end]
            Qs = mlData.Qs[res_num,iAtten,start:end]
        #iq_vels = np.round(iq_vels * xWidth / max(mlData.iq_vels[res_num, iAtten, :]) )
        
        iq_vels = iq_vels / np.amax(mlData.iq_vels[res_num, :, :])
        res_mag = math.sqrt(np.amax(mlData.Is[res_num, :, :]**2 + mlData.Qs[res_num, :, :]**2))
        # print mlData.Is[res_num, :, :], res_mag
        Is = Is / res_mag
        Qs = Qs / res_mag

        if scale != 1:
            Is = Is * scale
            Qs = Qs * scale
            iq_vels = iq_vels * scale
        # Is = Is /np.amax(mlData.iq_vels[res_num, :, :])
        # Qs = Qs /np.amax(mlData.iq_vels[res_num, :, :])

        # Is = Is /np.amax(mlData.Is[res_num, :, :])
        # Qs = Qs /np.amax(mlData.Qs[res_num, :, :])

        # angle =math.pi/2 
        if angle != 0:
            #mags = Qs**2 + Is**2
            #mags = map(lambda x: math.sqrt(x), mags)#map(lambda x,y:x+y, a,b)

            #peak_idx = self.get_peak_idx(res_num,iAtten)
            # peak_idx =argmax(iq_vels)
            # #min_idx = argmin(mags)

            # phase_orig = math.atan2(Qs[peak_idx],Is[peak_idx])
            # #phase_orig = math.atan2(Qs[min_idx],Is[min_idx])

            # angle = -phase_orig

            rotMatrix = np.array([[np.cos(angle), -np.sin(angle)], 
                                     [np.sin(angle),  np.cos(angle)]])

            Is,Qs = np.dot(rotMatrix,[Is,Qs])

        image = np.zeros((len(Is),3))
        image[:,0] = Is
        image[:,1] = Qs
        image[:,2] = iq_vels

        loops.append(image)

    # if max_nClass != mlData.nClass:
    #     padding = np.zeros((max_nClass-len(loops),mlData.xWidth,3))
    #     loops = np.concatenate([loops,padding],axis=0)


    loops = loops * np.ones((len(loops),mlData.xWidth,3))
    if showFrames:
        plot_res(loops)

    return loops

def plot_res(loops):
    iAtten_view =  np.arange(0,np.shape(loops)[0],2)

    f, axarr = plt.subplots(len(iAtten_view),3,figsize=(5.0, 8.1))
    # axarr[0,1].set_title(iAtten)
    for i,av in enumerate(iAtten_view):
        axarr[i,0].plot(loops[av,:,2])
        axarr[i,0].set_ylabel(av)
        axarr[i,1].plot(loops[av,:,0])
        axarr[i,1].plot(loops[av,:,1])
        axarr[i,2].plot(loops[av,:,0],loops[av,:,1])
    plt.show()
    plt.close()

def get_peak_idx(mlData,res_num,iAtten):
    # if dataObj is None:
    #     if mlData.inferenceData is None:
    #         raise ValueError('Initialize dataObj first!')
    #     dataObj = mlData.inferenceData
    return argmax(mlData.iq_vels[res_num,iAtten,:])

def plotWeights(weights):
    '''creates a 2d map showing the positive and negative weights for each class'''
    # weights = [mlClass.sess.run(mlClass.W_conv1), mlClass.sess.run(mlClass.W_conv2)]
    print np.shape(weights)
    f, axarr = plt.subplots(3,3,figsize=(20.0, 4))
    axarr[2,1].set_xlabel('freq?')
    axarr[1,0].set_ylabel('atten?')
    for filt in range(3):
        for i, w in enumerate(weights):
            for out in range(np.shape(w)[2]):
                for a in range(np.shape(w)[0]):
                    axarr[i,filt].plot(w[a,:,out,filt])

    plt.show()

    f, axarr = plt.subplots(3,3,figsize=(20.0, 4))
    axarr[2,1].set_xlabel('freq?')
    axarr[1,0].set_ylabel('atten?')
    for filt in range(3):
        for i, w in enumerate(weights):
            # print np.shape(w)
            # plt.subplot(4,3,(i+1)*(row+1))
            im = axarr[i,filt].imshow(w[:,:,0,filt], cmap=cm.coolwarm, interpolation='none',aspect='auto')
            # plt.plot(weights[0,:,0, nc])
            plt.title(' %i' % i)


    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(im, cax=cbar_ax)

    plt.show()
    plt.close()

def plotActivations(activations):
    '''creates a 2d map showing the positive and negative weights for each class'''
    _,_,testImages,_ = mld.loadPkl(mldir+trainFile)

    # activations = [mlClass.sess.run(mlClass.x_image,feed_dict={mlClass.x: testImages}), 
    #             mlClass.sess.run(mlClass.h_conv1, feed_dict={mlClass.x: testImages}),
    #             mlClass.sess.run(mlClass.h_pool1, feed_dict={mlClass.x: testImages}),
    #             mlClass.sess.run(mlClass.h_conv2, feed_dict={mlClass.x: testImages}),
    #             mlClass.sess.run(mlClass.h_pool2, feed_dict={mlClass.x: testImages})]


    print np.shape(activations[0])
    for r in range(len(testImages)):
        # for p in range(max_nClass):
        #     print p
        f, axarr = plt.subplots(len(activations)+1,8,figsize=(16.0, 8.1))
        p=6
        axarr[0,0].plot(activations[0][r,p,:,0],activations[0][r,p,:,1])
        for ir, row in enumerate(range(1, 4)):
            axarr[0,row].plot(activations[0][r,p,:,ir])
        # print np.arange(0,max_nClass,2)
        for ir, row in enumerate(np.arange(0,max_nClass,2)):
            for i, a in enumerate(activations):
                im = axarr[i+1,ir].imshow(np.rot90(a[r,row,:,:]), cmap=cm.coolwarm,interpolation='none', aspect='auto')#aspect='auto'
            # if row==2:
            #     axarr[i+2,row].colorbar(cmap=cm.afmhot)
        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        f.colorbar(im, cax=cbar_ax)
        
        plt.show()
        plt.close()

def checkLoopAtten(inferenceData, res_num, iAtten, showFrames=False):
    '''A function to analytically check the properties of an IQ loop: saturatation and smoothness.

    To check for saturation if the ratio between the 1st and 2nd largest edge is > max_ratio_threshold.
    
    Another metric which is more of a proxy is if the angle on either side of the sides connected to the 
    longest edge is < min_theta or > max_theta the loop is considered saturated. 

    A True result means that the loop is unsaturated.

    Inputs:
    res_num: index of resonator in question
    iAtten: index of attenuation in question
    showLoop: pops up a window of the frame plotted 

    Output:
    Theta 1 & 2: used as a proxy for saturation
    Max ratio: the ratio of highest and 2nd highest v_iq - a more reliable indicator of saturation
    vels: angles every 3 points make around the loop. The closer each to ~ 160 deg the smoother the loop
    '''
    # vindx = (-inferenceData.iq_vels[res_num,iAtten,:]).argsort()[:1]
    # print vindx    
    vindx = np.argmax(inferenceData.iq_vels[res_num,iAtten,:])

    if vindx == 0:
        max_neighbor = inferenceData.iq_vels[res_num, iAtten,1]
    elif vindx == len(inferenceData.iq_vels[res_num,iAtten,:])-1:
        max_neighbor = inferenceData.iq_vels[res_num,iAtten,vindx-1]
    else:
        max_neighbor = maximum(inferenceData.iq_vels[res_num,iAtten,vindx-1],inferenceData.iq_vels[res_num, iAtten,vindx+1])

    max_theta_vel  = math.atan2(inferenceData.Qs[res_num,iAtten,vindx-1] - inferenceData.Qs[res_num,iAtten,vindx], 
                                inferenceData.Is[res_num,iAtten,vindx-1] - inferenceData.Is[res_num,iAtten,vindx])
    low_theta_vel = math.atan2(inferenceData.Qs[res_num,iAtten,vindx-2] - inferenceData.Qs[res_num,iAtten,vindx-1], 
                               inferenceData.Is[res_num,iAtten,vindx-2] - inferenceData.Is[res_num,iAtten,vindx-1])
    upp_theta_vel = math.atan2(inferenceData.Qs[res_num,iAtten,vindx] - inferenceData.Qs[res_num,iAtten,vindx+1], 
                               inferenceData.Is[res_num,iAtten,vindx] - inferenceData.Is[res_num,iAtten,vindx+1])


    # print (inferenceData.iq_vels[res_num,iAtten,vindx]/ max_neighbor)
    max_ratio = (inferenceData.iq_vels[res_num,iAtten,vindx]/ max_neighbor)

    theta1 = (math.pi + max_theta_vel - low_theta_vel)/math.pi * 180
    theta2 = (math.pi + upp_theta_vel - max_theta_vel)/math.pi * 180

    theta1 = abs(theta1)
    if theta1 > 360:
        theta1 = theta1-360
    theta2= abs(theta2)
    if theta2 > 360:
        theta2 = theta2-360

    if showFrames:
        plt.plot(inferenceData.Is[res_num,iAtten,:],inferenceData.Qs[res_num,iAtten,:], 'g.-')
        plt.show()
    
    vels = np.zeros((len(inferenceData.Is[res_num,iAtten,:])-2))
    # for i,_ in enumerate(vels[1:-1]):
    for i,_ in enumerate(vels, start=1):
        low_theta_vel = math.atan2(inferenceData.Qs[res_num,iAtten,i-1] - inferenceData.Qs[res_num,iAtten,i], 
                                   inferenceData.Is[res_num,iAtten,i-1] - inferenceData.Is[res_num,iAtten,i])
        if low_theta_vel < 0: 
            low_theta_vel = 2*math.pi+low_theta_vel
        upp_theta_vel = math.atan2(inferenceData.Qs[res_num,iAtten,i+1] - inferenceData.Qs[res_num,iAtten,i], 
                                   inferenceData.Is[res_num,iAtten,i+1] - inferenceData.Is[res_num,iAtten,i])
        if upp_theta_vel < 0: 
            upp_theta_vel = 2*math.pi+upp_theta_vel
        vels[i-1] = abs(upp_theta_vel- low_theta_vel)/math.pi * 180

    return [theta1, theta2, max_ratio, vels]


def checkResAtten(inferenceData, res_num, plotAngles=False, showResData=False, min_theta = 115, max_theta = 220, max_ratio_threshold = 2.5):
    '''
    Outputs useful properties about each resonator using checkLoopAtten.
    Figures out if a resonator is bad using the distribution of angles around the loop
    Analytically finds the attenuation values when the resonator is saturated using the max ratio metric and adjacent angles to max v_iq line metric 

    Inputs:
    min/max_theta: limits outside of which the loop is considered saturated
    max_ratio_threshold: maximum largest/ 2nd largest IQ velocity allowed before loop is considered saturated
    showFrames: plots all the useful information on one plot for the resonator

    Oututs:
    Angles non sat: array of bools (true is non sat)
    Ratio non sat: array of bools (true is non sat)
    Ratio: ratio in v_iq between 1st and next highest adjacent max
    Running ratio: Ratio but smoothed using a running average
    Bad res: using the distribution of angles bad resonators are identified (true is bad res)
    Angles mean center: the mean of the angles around the center of the distribution (should be ~ 160)
    Angles std center: the standard dev of the angles. In the center they should follow a gauss dist and the tighter the better 
    '''
    nattens = np.shape(inferenceData.attens)[1]
    max_ratio_threshold = np.linspace(0,max_ratio_threshold*7,int(nattens))
    # max_ratio = inferenceData.iq_vels[res_num,iAtten,vindx[0]]/ inferenceData.iq_vels[res_num,iAtten,vindx[1]]

    max_theta = np.linspace(max_theta,max_theta*1.2,int(nattens))
    min_theta = np.linspace(min_theta,min_theta/1.2,int(nattens))

    angles = np.zeros((nattens,2))
    ratio = np.zeros(nattens)

    angles_nonsat = np.ones(nattens)
    ratio_nonsat = np.zeros(nattens)


    running_ratio = np.zeros((nattens))  

    vels = np.zeros((np.shape(inferenceData.iq_vels[0])[0], np.shape(inferenceData.iq_vels[0])[1]-1))


    # for ia, _ in enumerate(inferenceData.attens):
    for ia in range(nattens):
        loop_sat_cube = checkLoopAtten(inferenceData,res_num,iAtten=ia, showFrames=False)
        angles[ia,0], angles[ia,1], ratio[ia], vels[ia] = loop_sat_cube
        ratio_nonsat[ia] = ratio[ia] < max_ratio_threshold[ia]
        
    angles_running =  np.ones((nattens,2))*angles[0,:]
    for ia in range(1,nattens):
        angles_running[ia,0] = (angles[ia,0] + angles_running[ia-1,0])/2
        angles_running[ia,1] = (angles[ia,1] + angles_running[ia-1,1])/2
        # running_ratio[-ia] = np.sum(ratio[-ia-1: -1])/ia
        running_ratio[-ia-1] = (running_ratio[-ia] + ratio[-ia-1])/2

    # for ia in range(1,nattens-2):
    #     diff_rr[ia] = sum(running_ratio[ia-1:ia])-sum(running_ratio[ia+1:ia])

    for ia in range(nattens/2):
        angles_nonsat[ia] = (max_theta[ia] > angles_running[ia,0] > min_theta[ia]) and (max_theta[ia] > angles_running[ia,1] > min_theta[ia])

    angles_mean_center = np.mean(vels[:,35:115], axis =1)
    angles_std_center = np.std(vels[:,35:115], axis=1)
    angles_mean = np.mean(vels,axis=1)
    angles_std = np.std(vels,axis=1)

    delim = np.shape(vels)[1]/3

    y, x = np.histogram(vels[:,50:100])
    x = x[:-1]
    
    angles_dist = np.zeros((nattens,len(y)))
    # angles_mean_correct=np.zeros(nattens))
    # angles_std_correct=np.zeros((nattens))

    for ia in range(nattens):
        angles_dist[ia],_ = np.histogram(vels[ia,:])
        tail = np.linspace(angles_dist[ia,0],angles_dist[ia,-1],len(angles_dist[ia]))
        angles_dist[ia] = abs(angles_dist[ia] - tail)
        # angles_mean_correct[ia] = mean(x,angles_dist[ia])
        # angles_std_correct[ia] = std(x,angles_dist[ia],angles_mean_correct[ia])

    tail = np.linspace(y[0],y[-1],len(y))
    y = y - tail
    mid_x = x[4:8]
    mid_y = y[4:8]
    
    # def mean(x, y):
    #     return sum(x*y) /sum(y)
    # def std(x,y,mean):
    #     return np.sqrt(sum(y * (x - mean)**2) / sum(y))
    # def Gauss(x, a, x0, sigma):
    #     return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    # from scipy.optimize import curve_fit
    # # correction for weighted arithmetic mean
    # mean = mean(x,y)
    # # mean = 153
    # sigma = std(x,y,mean)
    # varience = 5
    # print mean, sigma
    # popt,pcov = curve_fit(Gauss, mid_x, mid_y, p0=[max(y), mean, sigma])
    # chi2 = sum((mid_y- Gauss(mid_x, *popt))**2 / varience**2 )
    # dof = len(mid_y)-3

    if plotAngles:
        plt.title(res_num)
        plt.plot(x, y, 'b:', label='data')
        # plt.plot(mid_x, Gauss(mid_x, *popt), 'r--', label='fit')
        # plt.plot(mid_x,mid_y -  Gauss(mid_x, *popt), label='residual')
        # plt.legend()
        plt.xlabel('Angle')
        plt.show()
        # for ia,_ in enumerate(nattens):
        #     plt.plot(x, angles_dist[ia], 'b:', label='data')
        #     plt.show()

    # if chi2 < 100
    if np.all(angles_std_center > 80):
        print 'yes', angles_std_center
    if max(mid_y)<0:
            bad_res = True
    else: bad_res = False

    if showResData:
        fig, ax1 = plt.subplots()

        # ax1.plot(inferenceData.attens,angles[:,0],'b')
        # ax1.plot(inferenceData.attens,angles[:,1],'g')
        ax1.plot(inferenceData.attens,angles_running,'g')
        ax1.plot(inferenceData.attens,vels, 'bo',alpha=0.2)
        # ax1.plot(inferenceData.attens,angles_dist, 'bo',alpha=0.2)
        ax1.plot(inferenceData.attens,min_theta, 'k--')
        ax1.plot(inferenceData.attens,max_theta, 'k--')
        ax1.plot(inferenceData.attens,angles_mean)
        ax1.plot(inferenceData.attens,angles_std, 'r')
        ax1.plot(inferenceData.attens,angles_mean_center, 'b--')
        ax1.plot(inferenceData.attens,angles_std_center, 'r--')
        ax1.set_xlabel('Atten index')
        ax1.set_ylabel('Angles')
        ax1.set_title(res_num)
        for tl in ax1.get_yticklabels():
            tl.set_color('b')

        # ax2 = ax1.twinx()
        # ax2.plot(inferenceData.attens,ratio, 'r', label='ratio')
        # ax2.plot(inferenceData.attens,max_ratio_threshold, 'k--', label='thresh')
        # ax2.set_ylabel('Ratios')
        # for tl in ax2.get_yticklabels():
        #     tl.set_color('r')
        # plt.legend()

        # ax3 = ax1.twinx()
        # ax3.plot(inferenceData.attens,angles_nonsat, 'purple', label='angles_nonsat')
        # ax3.plot(inferenceData.attens,ratio_nonsat, 'crimson', label='ratio_nonsat')
        # fig.subplots_adjust(right=0.75)
        # ax3.spines['right'].set_position(('axes', 1.2))
       

        ax4 = ax1.twinx()
        
        from matplotlib import cm
        import matplotlib.colors
        # ax4.autoscale(False)
        ax4.imshow(angles_dist.T,interpolation='none',cmap=cm.coolwarm,alpha=0.5, origin='lower',extent=[inferenceData.attens[0],inferenceData.attens[-1],0,len(x)], aspect='auto')
        # ax4.legend()
        # plt.colorbar(cmap=cm.afmhot)
        # ax4.set_ylim(0,10)
        plt.show() 

    return [angles_nonsat,ratio_nonsat, ratio, running_ratio, bad_res, angles_mean_center, angles_std_center]

# def normalise_train():
def plot_confusion(independent, dependent):
    # def plotComparison(rawTrainData, atten_guess):
    # rawTrainData.opt_iAttens = rawTrainData.opt_iAttens[:len(atten_guess)]
    guesses_map = np.zeros((max_nClass,max_nClass))
    for ia,ao in enumerate(independent):   
        ag = dependent[ia]
        guesses_map[ag,ao] += 1

    from matplotlib import cm
    import matplotlib.colors
    plt.imshow(guesses_map,interpolation='none', origin='lower', cmap=cm.coolwarm) #,norm = matplotlib.colors.LogNorm())
    plt.xlabel('actual')
    plt.ylabel('estimate')

    plt.colorbar(cmap=cm.afmhot)
    # plt.show()
    plt.figure()
    plt.plot(np.sum(guesses_map, axis=0), label='actual')
    plt.plot(np.sum(guesses_map, axis=1), label='estimate')
    plt.legend(loc="upper left")
    plt.show()

def plot_accuracy(train_ce, test_ce, train_acc, test_acc):
    # if accuracy_plot == 'post':
    fig = plt.figure(frameon=False,figsize=(15.0, 5.0))
    fig.add_subplot(121)
    plt.plot(train_ce, label='train')
    plt.plot(test_ce, label='test')
    plt.legend(loc='upper right')
    fig.add_subplot(122)
    plt.plot(train_acc, label='train')
    plt.plot(test_acc, label='test')
    plt.legend(loc='lower right')
    plt.show()

def plot_missed(ys_true, ys_guess, testImages, get_true_ind=True):
    # plt.plot(np.histogram(np.argmax(testLabels,1), range(21))[0])
    # plt.plot(np.histogram(ys_guess, range(21))[0])
    # plt.show()
    if get_true_ind:
        h5File = rawTrainFiles[0]
        h5File = os.path.join(mdd,h5File)
        mlData = mld.PSFitMLData(h5File = h5File)
        mlData.loadRawTrainData()

        res_nums = len(mlData.resIDs)

        # recalculate test_ind for reference 
        train_ind = np.array(map(int,np.linspace(0,res_nums-1,res_nums*mlData.trainFrac)))
        test_ind=[]
        np.asarray([test_ind.append(el) for el in range(res_nums) if el not in train_ind])

    missed = []
    for i,y in enumerate(ys_true):
        if ys_guess[i] != y:
            missed.append(i)

    for f in range(int(np.ceil(len(missed)/res_per_win))+1):
        reduced_missed = np.asarray(missed[f*res_per_win:(f+1)*res_per_win])
        print reduced_missed
        if get_true_ind:
            try:
                test_ind = np.asarray(test_ind)
                good_missed = test_ind[reduced_missed]
                print mlData.good_res[good_missed]
            except IndexError:
                pass

        _, axarr = plt.subplots(2*res_per_win, max_nClass, figsize=(16.0, 8.1))
        for r in range(res_per_win):
            for ia in range(max_nClass):
                axarr[2*r,ia].axis('off')
                axarr[(2*r)+1,ia].axis('off')
                
                try: 
                    if ia == ys_true[missed[r+f*res_per_win]]: 
                        axarr[2*r,ia].plot(testImages[missed[r+f*res_per_win]][ia,:,2], 'g')
                    elif ia == ys_guess[missed[r+f*res_per_win]]: 
                        axarr[2*r,ia].plot(testImages[missed[r+f*res_per_win]][ia,:,2], 'r')
                    else: 
                        axarr[2*r,ia].plot(testImages[missed[r+f*res_per_win]][ia,:,2], 'b')
                    if ia == ys_true[missed[r+f*res_per_win]]: 
                        axarr[(2*r)+1,ia].plot(testImages[missed[r+f*res_per_win]][ia,:,0],testImages[missed[r+f*res_per_win]][ia,:,1], 'g-o')
                    elif ia == ys_guess[missed[r+f*res_per_win]]:
                        axarr[(2*r)+1,ia].plot(testImages[missed[r+f*res_per_win]][ia,:,0],testImages[missed[r+f*res_per_win]][ia,:,1], 'r-o')
                    else: 
                        axarr[(2*r)+1,ia].plot(testImages[missed[r+f*res_per_win]][ia,:,0],testImages[missed[r+f*res_per_win]][ia,:,1], 'b-o')
                except:
                    pass

        plt.show()
        plt.close()

def get_opt_atten_from_ind(mlData, atten_guess):
    # for i in range(12):
    #     print i, mlData.attens_orig[i], atten_guess[i]
    # print np.shape(mlData.attens_orig), np.shape(atten_guess)

    # print np.shape(mlData.attens_orig), np.shape(mlData.iq_vels_orig)
    mlData.opt_attens=np.zeros((len(atten_guess)))
    mlData.opt_freqs=np.zeros((len(atten_guess)))

    print np.shape(atten_guess), np.shape(mlData.good_res), 

    for r,a in enumerate(atten_guess):
        mlData.opt_attens[r] = mlData.attens_orig[r,a]
        mlData.opt_freqs[r] = mlData.freqs_orig[r, argmax(mlData.iq_vels_orig[r,a])]

    mlData.opt_attens = mlData.opt_attens[mlData.good_res]
    mlData.opt_freqs = mlData.opt_freqs[mlData.good_res]

    print np.shape(mlData.opt_attens), mlData.good_res
    # self.inferenceData.opt_attens[r] = self.inferenceData.attens[self.atten_guess[r]]
    # self.inferenceData.opt_freqs[r] = self.inferenceData.freqs[r,self.get_peak_idx(r,self.atten_guess[r])]
    return mlData

def reduce_PSFile(PSFile, good_res):
    print 'loading peak location data from %s' % PSFile
    PSFile = np.loadtxt(PSFile, skiprows=0)

    # opt_freqs = PSFile[:,1]
    # self.opt_attens = PSFile[:,2] #+ 1
    goodResIDs = PSFile[:,0]
    new_PSFile = np.zeros((len(good_res),3))
    for ir, r in enumerate(good_res):
        for c in range(3):
            new_PSFile[ir,c] = PSFile[np.where(goodResIDs == r)[0],c]

    return new_PSFile

def evaluateModel(mlClass, initialFile, showFrames=False, plot_missed=False, res_nums=50):
    '''
    The loopTrain() function evaluates true performance by running findAtten on the training dataset. The predictions 
    on the correct attenuation value for is resonator can then compared with what the human chose. Then you can see the 
    models accuracy and if their are any trends in the resonators it's missing
    '''
    print 'running model on test input data'
    mlClass.findPowers(inferenceFile=initialFile, searchAllRes=True, res_nums=res_nums)

    rawTrainData = mld.PSFitMLData(h5File = initialFile)
    rawTrainData.loadRawTrainData() # run again to get 

    # res remaining after manual flagging and cut to include up to res_nums
    # man_mask=rawTrainData.good_res[:np.where(rawTrainData.good_res<=res_nums)[0][-1]]
    man_mask=rawTrainData.good_res
    bad_man_mask = [] #res flagged by mlClass in man_mask 
    for item in mlClass.bad_res:
        try:
            bad_man_mask.append(list(man_mask).index(item) )
        except: pass
    rawTrainData.opt_iAttens =np.delete(rawTrainData.opt_iAttens, bad_man_mask)

    man_mask = np.delete(man_mask, bad_man_mask)

    atten_guess = mlClass.atten_guess[man_mask]
    # atten_guess_mode = mlClass.atten_guess_mode[man_mask]
    # atten_guess_mean = mlClass.atten_guess_mean[man_mask]
    # atten_guess_med = mlClass.atten_guess_med[man_mask]
    low_stds = mlClass.low_stds[man_mask]
    ratio_guess = mlClass.ratio_guesses[man_mask]

    correct_guesses = []
    bins = [5,3,1,0]

    def getMatch(bins=[5,3,1,0], metric=atten_guess, original=rawTrainData.opt_iAttens):
        matches = np.zeros((len(bins),len(atten_guess)))

        for ig, _ in enumerate(metric):
            for ib, b in enumerate(bins):
                if abs(metric[ig]-original[ig]) <=b: 
                    matches[ib,ig] = 1

        for ib, b in enumerate(bins):
            print 'within %s' % b, sum(matches[ib])/len(metric)

        return matches

    def plotCumAccuracy(matches, atten_guess, bins=[5,3,1,0]):
        cs = np.zeros((len(bins),len(atten_guess)))
        for ib,_ in enumerate(bins):
            cs[ib] = np.cumsum(matches[ib]/len(atten_guess))
        
        for ib, b in enumerate(bins):
            plt.plot(np.arange(len(atten_guess))/float(len(atten_guess))-cs[ib], label='within %i' % b)

        plt.legend(loc="upper left")
        plt.show()

    def getMissed(metric, original=rawTrainData.opt_iAttens):
        wrong_guesses=[]
        for ig, _ in enumerate(metric):            
            if abs(metric[ig]-original[ig]) >0:
                wrong_guesses.append(ig)

        return wrong_guesses

    # getMatch(bins, low_stds)
    # getMatch(bins, ratio_guess)
    matches = getMatch(bins, atten_guess)
    # matches = getMatch(bins, atten_guess_mean)
    # matches = getMatch(bins, atten_guess_med)

    wrong_guesses = getMissed(atten_guess,rawTrainData.opt_iAttens)

    # plotCumAccuracy(matches,atten_guess)

    def plotMissed(mlClass, man_mask, wrong_guesses, rawTrainData=None):
        for i, wg in enumerate(wrong_guesses):
            plotRes(mlClass, man_mask, wg, rawTrainData)

    # plotMissed(mlClass, man_mask, wrong_guesses, rawTrainData,)

    def plotComparison(rawTrainData, atten_guess):
        rawTrainData.opt_iAttens = rawTrainData.opt_iAttens[:len(atten_guess)]
        # guesses_map = np.zeros((np.shape(rawTrainData.attens)[1],np.shape(rawTrainData.attens)[1]))
        print rawTrainData.opt_iAttens,

        guesses_map = np.zeros((max_nClass,max_nClass))
        for ia,ao in enumerate(rawTrainData.opt_iAttens):   
            ag = np.int_(atten_guess[ia])
            print ag, ao
            guesses_map[ag,ao] += 1

        from matplotlib import cm
        import matplotlib.colors
        plt.imshow(guesses_map,interpolation='none', origin='lower', cmap=cm.coolwarm) #,norm = matplotlib.colors.LogNorm())
        plt.xlabel('actual')
        plt.ylabel('estimate')

        plt.colorbar(cmap=cm.afmhot)
        # plt.show()
        plt.figure()
        plt.plot(np.sum(guesses_map, axis=0), label='actual')
        plt.plot(np.sum(guesses_map, axis=1), label='estimate')
        plt.legend(loc="upper left")
        plt.show()

    plotComparison(rawTrainData, atten_guess)
    # plotComparison(rawTrainData, atten_guess_mean)
    # plotComparison(rawTrainData, atten_guess_med)

    def plotBinComparison(rawTrainData, atten_guess):
        good_res = rawTrainData.good_res[:len(atten_guess)-1]
        # guesses_map = np.zeros((np.shape(rawTrainData.attens)[1],np.shape(rawTrainData.attens)[1]))
        good_mask = np.zeros((len(good_res)))
        print np.shape(good_res), np.shape(good_mask), 
        good_mask[good_res]=1
        print good_res, good_mask, np.shape(good_mask)
        # good_res = np.any(self.good_res == rn)
        #     one_hot[good_res] = 1
        guesses_map = np.zeros((2,2))

        for ia,ao in enumerate(good_mask):   
            ag = atten_guess[ia]
            guesses_map[ag,ao] += 1

        from matplotlib import cm
        import matplotlib.colors
        plt.imshow(guesses_map,interpolation='none', origin='lower', cmap=cm.coolwarm) #,norm = matplotlib.colors.LogNorm())
        plt.xlabel('actual')
        plt.ylabel('estimate')

        plt.colorbar(cmap=cm.afmhot)
        plt.show()

        plt.plot(np.sum(guesses_map, axis=0), label='actual')
        plt.plot(np.sum(guesses_map, axis=1), label='estimate')
        plt.legend(loc="upper left")
        plt.show()
    # plotBinComparison(rawTrainData, atten_guess)
def plotRes(mlClass, man_mask, res=0, rawTrainData=None):
    # print res, man_mask[res], mlClass.atten_guess[man_mask[res]], '\t', rawTrainData.opt_iAttens[res], '\t', mlClass.low_stds[man_mask[res]] 
    fig, ax1 = plt.subplots()
    ax1.set_title(man_mask[res])
    ax1.plot(mlClass.inferenceLabels[man_mask[res],:], color='b',label='model')
    # ax1.plot(mlClass.max_2nd_vels[man_mask[res],:]/max(mlClass.max_2nd_vels[man_mask[res],:]),color='g', label='2nd vel')
    ax1.plot(mlClass.max_ratios[man_mask[res],:]/max(mlClass.max_ratios[man_mask[res]]),color='k', label='max vel')
    ax1.plot(mlClass.running_ratios[man_mask[res],:]/max(mlClass.running_ratios[man_mask[res],:]),color='g', label='running' )
    ax1.axvline(mlClass.atten_guess_mode[man_mask[res]], color='b', linestyle='--', label='machine')
    ax1.axvline(mlClass.ratio_guesses[man_mask[res]], color='g', linestyle='--', label='ratio')
    ax1.axvline(mlClass.low_stds[man_mask[res]], color='r', linestyle='--', label='angles')
    ax1.axvline(rawTrainData.opt_iAttens[res], color='k', linestyle='--', label='human')
    ax1.set_xlabel('Atten index')
    ax1.set_ylabel('Scores and 2nd vel')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    plt.legend()

    ax2 = ax1.twinx()
    ax2.plot(mlClass.ang_means[man_mask[res]], color='r', label='ang means')
    ax2.plot(mlClass.ang_stds[man_mask[res]], color='r', label='std means')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    plt.show()  

def compare_train_data(h5File, PSFile1, PSFile2, plot_agree=True):
    mlData1 = mld.PSFitMLData(h5File = h5File, PSFile = PSFile1)
    mlData1.loadRawTrainData()

    mlData2 = mld.PSFitMLData(h5File = h5File, PSFile = PSFile2)
    mlData2.loadRawTrainData()

    print np.shape(mlData1.opt_iAttens), np.shape(mlData2.opt_iAttens)
    print mlData1.resIDs, mlData2.resIDs
    final_res = min([max(mlData1.resIDs), max(mlData2.resIDs)])
    print final_res

    print np.where(mlData1.resIDs >= final_res)[0][0], np.where(mlData2.resIDs >= final_res)[0][0]

    # from collections import Counter
    # print list((Counter(mlData1.resIDs) & Counter(mlData1.resIDs)).elements())

    good_res = []
    for r in range(int(final_res)):
        print r
        if r in mlData1.resIDs and r in mlData2.resIDs:
            print 'yep'
            good_res.append(r)

    print good_res, len(good_res)
    # mlData1.resIDs = mlData1.resIDs[np.where(mlData1.resIDs == final_res)]
    opt_attens1 = np.zeros((len(good_res)))
    opt_attens2 = np.zeros((len(good_res)))
    
    for ir, r in enumerate(good_res):
        opt_attens1[ir] = mlData1.opt_iAttens[np.where(mlData1.resIDs == r)[0]]
        opt_attens2[ir] = mlData2.opt_iAttens[np.where(mlData2.resIDs == r)[0]]
    
    # mlData1.opt_iAttens = mlData1.opt_iAttens[:np.where(mlData1.resIDs >= final_res)[0][0]]
    # mlData2.opt_iAttens = mlData2.opt_iAttens[:np.where(mlData2.resIDs >= final_res)[0][0]]

    # mlData1.resIDs = mlData1.resIDs[:np.where(mlData1.resIDs >= final_res)[0][0]]
    # mlData2.resIDs = mlData1.resIDs[:np.where(mlData2.resIDs >= final_res)[0][0]]

    # print np.shape(mlData1.opt_iAttens), np.shape(mlData2.opt_iAttens)
    # print mlData1.resIDs, mlData2.resIDs

    print opt_attens1, opt_attens2
    diff = abs(opt_attens1 - opt_attens2)
    print diff[:10]
    mean = (opt_attens1 + opt_attens2)/2
    print mean[:10]
    var = np.sqrt(((opt_attens1-mean)**2 + (opt_attens2-mean)**2)/2)
    print var[:10]
    print argmax(var)

    # hist, bins = np.histogram(var)
    # plt.plot(hist, bins[:-1])
    # plt.show()

    def plot_agree_acc():
        # if not os.path.isfile(mldir + trainFile):
        #     print 'Could not find train file. Making new training images from initial h5File'
        #     mlData1.makeTrainData(res_per_class)

        mlClass = mlc.mlClassification(subdir='trained on x-reduced/')

        plot_missed = False
        kwargs = {'batches': batches, 
        'trainReps': trainReps, 
        'plot_missed': plot_missed,
        'plot_confusion': plot_confusion,
        'max_learning_rate': max_learning_rate, 
        'min_learning_rate': min_learning_rate, 
        'decay_speed': decay_speed}

        mlClass.train(**kwargs)
        mlClass.findPowers(inferenceFile=h5File)

        # print mlClass.inferenceLabels[:,5], np.shape(mlClass.inferenceLabels)
        # mlClass.inferenceLabels = mlClass.inferenceLabels[good_res]
        accuracies = np.zeros((len(good_res)))
        for ir, r in enumerate(good_res):
            accuracies[ir] = mlClass.inferenceLabels[r,opt_attens1[ir]]

        print accuracies
        var_bins = np.unique(var)
        print var_bins
        sum_accuracies = np.zeros((len(var_bins)))
        amount_accuracies = np.zeros((len(var_bins)))
        acc_binned = [[] for i in range(len(var_bins))]
        print acc_binned, np.shape(acc_binned)

        for ia, a in enumerate(accuracies):
            for ivb, vb in enumerate(var_bins):
                if var[ia] == vb:            
                    sum_accuracies[ivb] += a
                    amount_accuracies[ivb] += 1
                    acc_binned[ivb].append(a)

        av_accuracies = sum_accuracies/amount_accuracies
        std_acc_bin = np.zeros((len(var_bins)))
        for ivb, vb in enumerate(var_bins):
            for iab,ab in enumerate(acc_binned[ivb]):
                std_acc_bin[ivb] += (ab - av_accuracies[ivb])**2
            std_acc_bin[ivb] = np.sqrt(std_acc_bin[ivb]/len(acc_binned[ivb]))

        print len(std_acc_bin), len(var_bins), len(amount_accuracies)
        # plt.plot(var, accuracies, 'o')
        agg_bins = 1-var_bins/max(var_bins)
        # plt.errorbar(var_bins, av_accuracies, yerr=std_acc_bin)
        # plt.scatter(var_bins, av_accuracies, s=amount_accuracies)
        plt.figure(figsize=(6, 5))

        # plt.scatter(agg_bins, av_accuracies, c='m', marker='s', s=10*amount_accuracies)
        cm = plt.cm.get_cmap('RdYlBu')
        sc = plt.scatter(agg_bins, av_accuracies, c=amount_accuracies, s=50, marker='s', cmap=cm)
        plt.errorbar(agg_bins, av_accuracies,c='k', yerr=std_acc_bin)
        plt.axhline(0.68,linestyle='--')
        plt.axhline(0.88,linestyle='--')
        plt.xlabel('Manual Agreement')
        plt.ylabel('Model Accuracy')
        plt.colorbar(sc)
        # plt.plot(var_bins, std_acc_bin)
        plt.show()

    if plot_agree:
        plot_agree_acc()

    agreed_res=[]
    for ir, r in enumerate(good_res):
        if diff[ir]==0:
            agreed_res.append(r)

    print len(agreed_res)        
    return agreed_res




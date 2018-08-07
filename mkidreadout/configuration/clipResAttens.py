"""
Author: Ben Mazin
Date: June 20, 2018

This code clips the attenuation values for resonators in a frequency list
This is useful so that a single high powered resonator doesn't take up the entire DAC dynamic range
It histograms the attenuations and then asks the user for a max and min attenuation
Resonators above the max are removed
Resonators below the min are clipped to the min

It outputs a new frequency file called *filename*_clip.txt

Usage:
$python clipResAttens.py ps_freq_FL6a.txt ps_freq_FL6b.txt
"""


import sys,os
import numpy as np
import matplotlib.pyplot as plt

def clipHist(filename):
    data = np.loadtxt(filename)
    maxAtten = np.amax(data[:,2])
    minAtten = np.amin(data[:,2])
    nBins = int(round(maxAtten - minAtten))

    # the histogram of the data
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(data[:,2], nBins, facecolor='g', alpha=0.75)
    ax.set_xlabel('Attenuation')
    ax.set_ylabel('Number')
    ax.set_title('Histogram of Resonator Attenuations\n'+os.path.basename(filename))
    ax.grid(True)

    atten = {'max':maxAtten, 'min':minAtten}
    ln_max = ax.axvline(x=maxAtten, c='b')
    ln_min = ax.axvline(x=minAtten, c='b')

    def onclick(event):
        if event.inaxes!=ax: return
        if fig.canvas.manager.toolbar._active is not None: return
        if event.xdata >= (atten['max']+atten['min'])/2.:
            atten['max'] = round(event.xdata)
            print atten['max']
            ln_max.set_xdata(atten['max'])
        else:
            atten['min'] = round(event.xdata)
            print atten['min']
            ln_min.set_xdata(atten['min'])
        plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    print 'Max Atten = ' + str(atten['max'])
    print 'Min Atten = ' + str(atten['min'])

    data[np.where(data[:,2]<atten['min'])[0],2]=atten['min']
    data = data[np.where(data[:,2]<atten['max'])]
    outFN = filename.rsplit('.',1)[0]+'_clip.txt'
    np.savetxt(outFN, data, fmt="%6i %10.9e %4i")

if __name__=='__main__':

    if( len(sys.argv) < 2 ):
        print 'Usage: python clipResAttens.py freqFilename'
    for fn in sys.argv[1:]:
        clipHist(fn)




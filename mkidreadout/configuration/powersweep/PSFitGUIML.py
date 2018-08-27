#-----------------------------------
# PSFitAuto.py
#
# Given IQ sweeps at various powers of resonators, this program chooses the best resonant frequency and power
# ----------------------------------
#
# Chris S:
#
# select_atten was being called multiple times after clicking on a point in plot_1.
# inside of select_atten, the call to self.ui.atten.setValue(self.atten) triggered
# another call to select_atten since it is a slot.
#
# So, instead of calling select_atten directly, call self.ui.atten.setValue(round(attenuation)) when
# you want to call select_atten, and do not call this setValue inside select_atten.
#
# Near line 68, set the frequency instead of the index to the frequency.
#
# Implemented a v2 gui that fits in a small screen.  
# Run this program with no arguments to get the "classic" look.
# Run this progam with any variable to get the "new" look.  The values of arguments are ignored.
#!/bin/env python
import ConfigParser
from numpy import *
import numpy
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from mkidreadout.utils.iqsweep import *
from matplotlib.backends.backend_qt4 import NavigationToolbar2QT as NavigationToolbar
import argparse

class StartQt4(QMainWindow):
    def __init__(self, cfgfile, feedline, Ui, parent=None):
        QWidget.__init__(self, parent)
        self.ui = Ui()
        self.ui.setupUi(self)
    
        self.atten = -1
        self.ui.atten.setValue(self.atten)  
        self.resnum = 0
        self.indx=0
        
        QObject.connect(self.ui.open_browse, SIGNAL("clicked()"), self.open_dialog)
        QObject.connect(self.ui.save_browse, SIGNAL("clicked()"), self.save_dialog)
        QObject.connect(self.ui.atten, SIGNAL("valueChanged(int)"), self.setnewatten)
        QObject.connect(self.ui.savevalues, SIGNAL("clicked()"), self.savevalues)
        QObject.connect(self.ui.jumptores, SIGNAL("clicked()"), self.jumptores)

        self.widesweep=None
        self.h5resID_offset=0
        self.wsresID_offset=0
        try:
            config = ConfigParser.ConfigParser()
            config.read(cfgfile)
            fl = 'FL' + feedline.lower()
            ws_FN = config.get(fl, 'widesweep_FN')
            ws_freqs_all_FN = config.get(fl, 'ws_freqs_all_FN')
            ws_freqs_good_FN = config.get(fl, 'ws_freqs_good_FN')
            self.widesweep=numpy.loadtxt(ws_FN)  #freq, I, Q
            self.widesweep_goodFreqs = numpy.loadtxt(ws_freqs_good_FN ,usecols=[2])
            self.widesweep_allResIDs,self.widesweep_allFreqs = numpy.loadtxt(ws_freqs_all_FN,usecols=[0,2],unpack=True)
            self.h5resID_offset=config.getint(fl, 'h5resID_offset')
            self.wsresID_offset=config.getint(fl, 'wsresID_offset')
            self.widesweep_allResIDs+=self.wsresID_offset
        except (IOError, ConfigParser.NoSectionError):
            print 'Could not load widewseep data :-('
            self.widesweep=None
            self.h5resID_offset=0
            self.wsresID_offset=0
            raise

        self.navi_toolbar = NavigationToolbar(self.ui.plot_3.canvas, self)
        self.ui.plot_3.canvas.setFocusPolicy( Qt.ClickFocus )
        cid=self.ui.plot_3.canvas.mpl_connect('key_press_event', self.zoom_plot_3)
        
    def open_dialog(self):
        self.openfile = QFileDialog.getOpenFileName(parent=None, caption=QString(str("Choose PS File")),directory = ".",filter=QString(str("H5 (*.h5)")))
        self.ui.open_filename.setText(str(self.openfile))
        self.loadps()
        
    def save_dialog(self):
        self.savefile = QFileDialog.getOpenFileName(parent=None, caption=QString(str("Choose Save File")),directory = ".")
        self.ui.save_filename.setText(str(self.savefile))
        self.f = open(str(self.savefile), 'a')
        self.f.close() 
        self.mlResIDs, self.mlFreqs, self.mlAttens = np.loadtxt(str(self.savefile), unpack=True)
    
    def loadres(self):
        self.Res1=IQsweep()
        self.Res1.LoadPowers(str(self.openfile), 'r0', self.freq[self.resnum])
        self.ui.res_num.setText(str(self.resnum))
        self.resfreq = self.freq[self.resnum]
        self.ui.jumptonum.setValue(self.resnum)

        try: self.Res1.resID+=self.h5resID_offset    #sometimes the resID in the h5 file is wrong...        
        except: self.Res1.resID = self.resnum+self.h5resID_offset   #or the h5 has no resID column
        
        self.resID =self.Res1.resID
        print "Res: "+str(self.resnum)+' --> ID: '+str(self.resID)
        
        self.ui.frequency.setText(str(self.resfreq))
        self.NAttens = len(self.Res1.atten1s)
        self.res1_iq_vels=numpy.zeros((self.NAttens,self.Res1.fsteps-1))
        self.res1_iq_amps=numpy.zeros((self.NAttens,self.Res1.fsteps))
        for iAtt in range(self.NAttens):
            for i in range(1,self.Res1.fsteps-1):
                self.res1_iq_vels[iAtt,i]=sqrt((self.Res1.Qs[iAtt][i]-self.Res1.Qs[iAtt][i-1])**2+(self.Res1.Is[iAtt][i]-self.Res1.Is[iAtt][i-1])**2)
                self.res1_iq_amps[iAtt,:]=sqrt((self.Res1.Qs[iAtt])**2+(self.Res1.Is[iAtt])**2)
        #Sort the IQ velocities for each attenuation, to pick out the maximums
        sorted_vels = numpy.sort(self.res1_iq_vels,axis=1)
        #Last column is maximum values for each atten (row)
        self.res1_max_vels = sorted_vels[:,-1]
        #Second to last column has second highest value
        self.res1_max2_vels = sorted_vels[:,-2]
        #Also get indices for maximum of each atten, and second highest
        sort_indices = numpy.argsort(self.res1_iq_vels,axis=1)
        max_indices = sort_indices[:,-1]
        max2_indices = sort_indices[:,-2]
        max_neighbor = max_indices.copy()

        #for each attenuation find the ratio of the maximum velocity to the second highest velocity
        self.res1_max_ratio = self.res1_max_vels.copy()
        max_neighbors = zeros(self.NAttens)
        max2_neighbors = zeros(self.NAttens)
        self.res1_max2_ratio = self.res1_max2_vels.copy()
        for iAtt in range(self.NAttens):
            if max_indices[iAtt] == 0:
                max_neighbor = self.res1_iq_vels[iAtt,max_indices[iAtt]+1]
            elif max_indices[iAtt] == len(self.res1_iq_vels[iAtt,:])-1:
                max_neighbor = self.res1_iq_vels[iAtt,max_indices[iAtt]-1]
            else:
                max_neighbor = maximum(self.res1_iq_vels[iAtt,max_indices[iAtt]-1],self.res1_iq_vels[iAtt,max_indices[iAtt]+1])
            max_neighbors[iAtt]=max_neighbor
            self.res1_max_ratio[iAtt] = self.res1_max_vels[iAtt]/max_neighbor
            if max2_indices[iAtt] == 0:
                max2_neighbor = self.res1_iq_vels[iAtt,max2_indices[iAtt]+1]
            elif max2_indices[iAtt] == len(self.res1_iq_vels[iAtt,:])-1:
                max2_neighbor = self.res1_iq_vels[iAtt,max2_indices[iAtt]-1]
            else:
                max2_neighbor = maximum(self.res1_iq_vels[iAtt,max2_indices[iAtt]-1],self.res1_iq_vels[iAtt,max2_indices[iAtt]+1])
            max2_neighbors[iAtt]=max2_neighbor
            self.res1_max2_ratio[iAtt] = self.res1_max2_vels[iAtt]/max2_neighbor
        #normalize the new arrays
        self.res1_max_vels /= numpy.max(self.res1_max_vels)
        self.res1_max_vels *= numpy.max(self.res1_max_ratio)
        self.res1_max2_vels /= numpy.max(self.res1_max2_vels)
        #self.res1_relative_max_vels /= numpy.max(self.res1_relative_max_vels)
        self.ui.plot_1.canvas.ax.clear()
        print 'file', self.openfile
        print 'attens', self.Res1.atten1s
        self.ui.plot_1.canvas.ax.plot(self.Res1.atten1s,self.res1_max_vels,'b.-',label='Max IQ velocity')
        #self.ui.plot_1.canvas.ax.plot(self.Res1.atten1s,max_neighbors,'r.-')
        self.ui.plot_1.canvas.ax.plot(self.Res1.atten1s,self.res1_max_ratio,'k.-',label='Ratio (Max Vel)/(2nd Max Vel)')
        self.ui.plot_1.canvas.ax.legend()
        self.ui.plot_1.canvas.ax.set_xlabel('attenuation')
        
        #self.ui.plot_1.canvas.ax.plot(self.Res1.atten1s,self.res1_max2_vels-1,'b.-')
        #self.ui.plot_1.canvas.ax.plot(self.Res1.atten1s,max2_neighbors-1,'b.-')
        #self.ui.plot_1.canvas.ax.plot(self.Res1.atten1s,self.res1_max2_ratio-1,'g.-')


        # Chris S:  seems that button_press_event causes click_plot_1 to be called more than once sometimes.
        cid=self.ui.plot_1.canvas.mpl_connect('button_press_event', self.click_plot_1)
        #cid=self.ui.plot_1.canvas.mpl_connect('button_release_event', self.click_plot_1)
        #self.ui.plot_1.canvas.format_labels()
        self.ui.plot_1.canvas.draw()

        max_ratio_threshold = 1.5
        rule_of_thumb_offset = 2

        # require ROTO adjacent elements to be all below the MRT
        bool_remove = np.ones(len(self.res1_max_ratio))
        for ri in range(len(self.res1_max_ratio)-rule_of_thumb_offset-2):
            bool_remove[ri] = bool((self.res1_max_ratio[ri:ri+rule_of_thumb_offset+1]< max_ratio_threshold).all())
        
        if(np.any(self.resID==self.mlResIDs)):
            resInd = np.where(self.resID==self.mlResIDs)[0]
            self.select_atten(self.mlAttens[resInd])
            self.ui.atten.setValue(self.mlAttens[resInd])
        
        else:
            guess_atten_idx = np.extract(bool_remove,np.arange(len(self.res1_max_ratio)))

            # require the attenuation value to be past the initial peak in MRT
            guess_atten_idx = guess_atten_idx[where(guess_atten_idx > argmax(self.res1_max_ratio) )[0]]

            if size(guess_atten_idx) >= 1:
                if guess_atten_idx[0]+rule_of_thumb_offset < len(self.Res1.atten1s):
                    guess_atten_idx[0] += rule_of_thumb_offset
                guess_atten = self.Res1.atten1s[guess_atten_idx[0]]
                self.select_atten(guess_atten)
                self.ui.atten.setValue(round(guess_atten))
            else:
                self.select_atten(self.Res1.atten1s[self.NAttens/2])
                self.ui.atten.setValue(round(self.Res1.atten1s[self.NAttens/2]))

    def guess_res_freq(self):
        if(np.any(self.resID==self.mlResIDs)):
            resInd = np.where(self.resID==self.mlResIDs)[0]
            self.select_freq(self.mlFreqs[resInd])
        else:
            guess_idx = argmax(self.res1_iq_vels[self.iAtten])
            #The longest edge is identified, choose which vertex of the edge
            #is the resonant frequency by checking the neighboring edges            
            #len(IQ_vels[ch]) == len(f_span)-1, so guess_idx is the index
            #of the lower frequency vertex of the longest edge            
            if guess_idx-1 < 0 or self.res1_iq_vel[guess_idx-1] < self.res1_iq_vel[guess_idx+1]:
                iNewResFreq = guess_idx
            else:
                iNewResFreq = guess_idx-1
            guess = self.Res1.freq[iNewResFreq]
            print 'Guessing resonant freq at ',guess,' for self.iAtten=',self.iAtten
            self.select_freq(guess)

    def loadps(self):
        #### IN HERE NEED TO ADD LOADING OF ID FIELD FROM PS
    
        hd5file=open_file(str(self.openfile),mode='r')
        group = hd5file.get_node('/','r0')
        self.freq=empty(0,dtype='float32')
        self.idList=empty(0,dtype='int32')
        for sweep in group._f_walknodes('Leaf'):
            k=sweep.read()
            self.scale = k['scale'][0]
            #print "Scale factor is ", self.scale
            self.freq=append(self.freq,[k['f0'][0]])
            self.idList=append(self.idList,[k['resID'][0]])
        self.idList+=self.h5resID_offset
        #self.freqList = np.zeros(len(k['f0']))
        #self.attenList = np.zeros(len(self.freqList)) - 1
        self.freqList = np.zeros(2000)
        #self.idList = np.zeros(len(self.freqList)) - 1
        self.attenList = np.zeros(len(self.freqList)) - 1
        hd5file.close()
        self.loadres()
    
    def on_press(self, event):
        self.select_freq(event.xdata)

    def zoom_plot_3(self,event):
        if str(event.key) == str('z'):
            print 'zoooom'
            self.navi_toolbar.zoom()

    def click_plot_1(self, event):
        #Chris. self.select_atten(event.xdata)
        self.ui.atten.setValue(round(event.xdata))
    def select_freq(self,freq):
        self.resfreq = freq
        self.ui.frequency.setText(str(self.resfreq))
        self.ui.plot_2.canvas.ax.plot(self.Res1.freq[self.indx],self.res1_iq_vel[self.indx],'bo')
        self.ui.plot_3.canvas.ax.plot(self.Res1.I[self.indx],self.Res1.Q[self.indx],'bo')
        self.indx=min(where(self.Res1.freq >= self.resfreq)[0][0], self.Res1.freq.size-1)
        self.ui.plot_2.canvas.ax.plot(self.Res1.freq[self.indx],self.res1_iq_vel[self.indx],'ro')
        self.ui.plot_2.canvas.draw()
        self.ui.plot_3.canvas.ax.plot(self.Res1.I[self.indx],self.Res1.Q[self.indx],'ro')
        self.ui.plot_3.canvas.draw()
       
        
    def select_atten(self,attenuation):
        if self.atten != -1:
            attenIndex = where(self.Res1.atten1s == self.atten)
            if size(attenIndex) >= 1:
                self.iAtten = attenIndex[0][0]
                self.ui.plot_1.canvas.ax.plot(self.atten,self.res1_max_ratio[self.iAtten],'ko')
                self.ui.plot_1.canvas.ax.plot(self.atten,self.res1_max_vels[self.iAtten],'bo')
        self.atten = round(attenuation)
        attenIndex = where(self.Res1.atten1s == self.atten)
        if size(attenIndex) != 1:
            print "Atten value is not in file"
            return
        self.iAtten = attenIndex[0][0]
        self.res1_iq_vel = self.res1_iq_vels[self.iAtten,:]
        self.Res1.I=self.Res1.Is[self.iAtten]
        self.Res1.Q=self.Res1.Qs[self.iAtten]
        self.Res1.Icen=self.Res1.Icens[self.iAtten]
        self.Res1.Qcen=self.Res1.Qcens[self.iAtten]
        self.ui.plot_1.canvas.ax.plot(self.atten,self.res1_max_ratio[self.iAtten],'ro')
        self.ui.plot_1.canvas.ax.plot(self.atten,self.res1_max_vels[self.iAtten],'ro')
        self.ui.plot_1.canvas.draw()
        #Chris S self.ui.atten.setValue(self.atten)
        self.makeplots()
        self.guess_res_freq()


    
    def makeplots(self):
        try:

            #Plot transmission magnitudeds as a function of frequency for this resonator
            #self.ui.plot_1.canvas.ax.clear()
            #self.ui.plot_1.canvas.ax.semilogy(self.Res1.freq,res1_iq_amp,'.-')
            #self.ui.plot_1.canvas.format_labels()
            #self.ui.plot_1.canvas.draw()
            
            self.ui.plot_2.canvas.ax.clear()
            self.ui.plot_2.canvas.ax.set_xlabel('frequency (GHz)')
            self.ui.plot_2.canvas.ax.set_ylabel('IQ velocity')
            self.ui.plot_2.canvas.ax.plot(self.Res1.freq[:-1],self.res1_iq_vel,'b.-')
            if self.iAtten > 0:
                self.ui.plot_2.canvas.ax.plot(self.Res1.freq[:-1],self.res1_iq_vels[self.iAtten-1],'g.-')
                self.ui.plot_2.canvas.ax.lines[-1].set_alpha(.7)
            if self.iAtten > 1:
                self.ui.plot_2.canvas.ax.plot(self.Res1.freq[:-1],self.res1_iq_vels[self.iAtten-2],'g.-')
                self.ui.plot_2.canvas.ax.lines[-1].set_alpha(.3)
            cid=self.ui.plot_2.canvas.mpl_connect('button_press_event', self.on_press)

            if self.widesweep is not None:
                freq_start = self.Res1.freq[0]/1.E9
                freq_stop = self.Res1.freq[-1]/1.E9
                widesweep_inds = np.where(np.logical_and(self.widesweep[:,0]>=freq_start, self.widesweep[:,0]<=freq_stop))
                iqVel_med = numpy.median(self.res1_iq_vel)
                ws_amp = (self.widesweep[widesweep_inds,1]**2. + self.widesweep[widesweep_inds,2]**2.)**0.5
                #ws_amp = 10.*numpy.log10(ws_amp)
                ws_amp_med = numpy.median(ws_amp)
                ws_amp*=1.0*iqVel_med/ws_amp_med

                self.ui.plot_2.canvas.ax.plot(self.widesweep[widesweep_inds,0]*1.E9, ws_amp,'k.-')
                self.ui.plot_2.canvas.ax.lines[-1].set_alpha(.5)
                
                #ws_goodFreqs = self.widesweep_goodFreqs[numpy.where(numpy.logical_and(self.widesweep_goodFreqs>=freq_start,self.widesweep_goodFreqs<=freq_stop))]
                ws_allFreqs = self.widesweep_allFreqs[numpy.where(numpy.logical_and(self.widesweep_allFreqs>=freq_start,self.widesweep_allFreqs<=freq_stop))]
                ws_allResIDs = self.widesweep_allResIDs[numpy.where(numpy.logical_and(self.widesweep_allFreqs>=freq_start,self.widesweep_allFreqs<=freq_stop))]
                for ws_resID_i, ws_freq_i in zip(ws_allResIDs,ws_allFreqs):
                    ws_color = 'k'
                    if ws_freq_i in self.widesweep_goodFreqs:
                        ws_color='r'
                    self.ui.plot_2.canvas.ax.axvline(ws_freq_i*1.E9, c=ws_color, alpha=0.5)
                    ws_ymax=self.ui.plot_2.canvas.ax.yaxis.get_data_interval()[1]
                    self.ui.plot_2.canvas.ax.text(x=ws_freq_i*1.E9, y=ws_ymax,s=str(int(ws_resID_i)), color=ws_color, alpha=0.5)
            self.ui.plot_2.canvas.draw()

            self.ui.plot_3.canvas.ax.clear()
            if self.iAtten >0:
                self.ui.plot_3.canvas.ax.plot(self.Res1.Is[self.iAtten-1],self.Res1.Qs[self.iAtten-1],'g.-')
                self.ui.plot_3.canvas.ax.lines[0].set_alpha(.6)
            if self.iAtten > 1:
                self.ui.plot_3.canvas.ax.plot(self.Res1.Is[self.iAtten-2],self.Res1.Qs[self.iAtten-2],'g.-')
                self.ui.plot_3.canvas.ax.lines[-1].set_alpha(.3)
            self.ui.plot_3.canvas.ax.plot(self.Res1.I,self.Res1.Q,'.-')
            if self.widesweep is not None:
                ws_I = self.widesweep[widesweep_inds,1]
                ws_Q = self.widesweep[widesweep_inds,2]
                ws_dataRange_I = self.ui.plot_3.canvas.ax.xaxis.get_data_interval()
                ws_dataRange_Q = self.ui.plot_3.canvas.ax.yaxis.get_data_interval()
                ws_I -= numpy.median(ws_I) - numpy.median(ws_dataRange_I)
                ws_Q -= numpy.median(ws_Q) - numpy.median(ws_dataRange_Q)
                #self.ui.plot_3.canvas.ax.plot(ws_I, ws_Q,'k.-')
                #self.ui.plot_3.canvas.ax.lines[-1].set_alpha(.5)
            #self.ui.plot_3.canvas.format_labels()
            print 'makeplots'
            self.ui.plot_3.canvas.draw()
        except IndexError:
            self.f.close()
            print "reached end of resonator list, saving file"
            print "closing GUI"
            sys.exit()

    def jumptores(self):
        try:
            self.atten = -1
            self.resnum = self.ui.jumptonum.value()
            self.resfreq = self.resnum
            self.loadres()
        except IndexError:
            print "Res value out of bounds."
            self.ui.plot_1.canvas.ax.clear()
            self.ui.plot_2.canvas.ax.clear()
            self.ui.plot_3.canvas.ax.clear()
            self.ui.plot_1.canvas.draw()
            self.ui.plot_2.canvas.draw()
            self.ui.plot_3.canvas.draw()
            

    def setnewatten(self):
        #Chris S.:  this is the only place that select_atten should be called.
        self.select_atten(self.ui.atten.value())

    def savevalues(self):
        #if self.resnum == 0:
        #    self.f = open(str(self.savefile), 'a')
        #    self.f.write(str(self.scale)+'\t'+str(self.scale)+'\t'+str(self.scale)+'\t'+str(self.scale)+'\n')
        #    self.f.close()
        #Icen=0
        #Qcen=0
        
        #self.idList[self.resnum] = self.id
        assert self.idList[self.resnum]==self.resID, "Something is wrong. Need to debug resID"
        self.freqList[self.resnum] = self.resfreq
        self.attenList[self.resnum] = self.atten
        data = np.transpose([self.idList[np.where(self.attenList >=0)], self.freqList[np.where(self.attenList >=0)], self.attenList[np.where(self.attenList >=0)]])
        
        print data
        
        try:
            #grab and append any data already saved
            dataCurrent = np.atleast_2d(np.loadtxt(str(self.savefile)))
            if len(dataCurrent)<1 or len(dataCurrent[0])<1:
                raise IOError('')
            
            # remove duplicates
            #notDuplicates = np.where(np.in1d(dataCurrent[:,0], data[:,0],invert=True))      # freqs aren't duplicated at these indices
            #dataCurrent = dataCurrent[notDuplicates,:]
            
            #remove doubles based on freq
            #doubleThreshold = 0.05e6     # 100 kHz
            #closestIdx = [(np.abs(data[:,1] - i)).argmin() for i in dataCurrent[:,1]]
            #notDoubles = np.where(np.abs(dataCurrent[:,1] - data[closestIdx,1]) > doubleThreshold)[0]
            #dataCurrent = dataCurrent[notDoubles,:]

            #remove doubles based on ID (seems to be bug in code that writes some lines twice)
            #doubleThreshold = 0.1e6
            #closestIdx = [(np.abs(data[:,0] - i)).argmin() for i in dataCurrent[:,0]]
            #notDoubles = np.where(np.abs(dataCurrent[:,0] - data[closestIdx,0]) > doubleThreshold)[0]
            #dataCurrent = dataCurrent[notDoubles,:]

            print dataCurrent

            dataNew = np.append(dataCurrent,data,axis=0)
            
            #sort the frequencies and remove duplicate IDs
            ids = np.copy(dataNew[:,0][::-1])
            freqs = np.copy(dataNew[:,1][::-1])
            attens = np.copy(dataNew[:,2][::-1])
            newIDs, indx = np.unique(ids, return_index=True)
            newFreqs = freqs[indx]
            newAttens = attens[indx]
            
            #sort the frequencies
            #ids = np.copy(dataNew[:,0][::-1])
            #freqs = np.copy(dataNew[:,1][::-1])
            #attens = np.copy(dataNew[:,2][::-1])
            #newFreqs, indx = np.unique(freqs, return_index=True)
            #newIDs = ids[indx]
            #newAttens = attens[indx]
            
            '''
            #Remove any doubles
            doubleThreshold = 0.5e6     # 500 kHz
            freqDiff = newFreqs[1:] - newFreqs[:-1]
            freqDiff = np.append(freqDiff, [doubleThreshold+1])     # last frequency is never a double
            notDoubles = np.where(freqDiff > doubleThreshold)
            newFreqs = newFreqs[notDoubles]
            newAttens = newAttens[notDoubles]
            '''
            
            data = np.transpose([newIDs,newFreqs,newAttens])
        except IOError:
            print 'IOError'
            pass
        
        np.savetxt(str(self.savefile), data, "%6i %10.9e %4i")
        
        #self.f = open(str(self.savefile), 'a')
        #self.f.write(str(self.resfreq)+'\t'+str(Icen)+'\t'+str(Qcen)+'\t'+str(self.atten)+'\n')
        #self.f.write(str(self.resfreq)+'\t'+str(self.atten)+'\n')
        #self.f.close()
        print " ....... Saved to file:  resnum=",self.resnum," resID=",self.resID, " resfreq=",self.resfreq," atten=",self.atten
        self.resnum += 1
        self.atten = -1
        self.mlResIDs, self.mlFreqs, self.mlAttens = np.loadtxt(str(self.savefile), unpack=True)
        self.loadres()
                
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='MKID Powersweep GUI')
    parser.add_argument('cfgfile', type=str, help='The config file', default='./wsData.cfg')
    parser.add_argument('feedline', type=str, help='Feedline number/band (e.g. 7a)')
    parser.add_argument('--small', action='store_true', dest='smallui', default=False,
                        help='Use small GUI')
    args = parser.parse_args()

    if args.smallui:
        from PSFit_GUI_v2 import Ui_MainWindow
    else:
        from PSFit_GUI import Ui_MainWindow


    app = QApplication(sys.argv)
    myapp = StartQt4(args.cfgfile, args.feedline, Ui_MainWindow)
    myapp.show()
    app.exec_()

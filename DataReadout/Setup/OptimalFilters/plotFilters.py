import sys
import os
from PyQt4 import QtGui
from PyQt4 import QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
try:
	from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
except ImportError: #Named changed in some newer matplotlib versions
	from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle




class Window(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.initUI()          
         
    def initUI(self):  
        #make the folder finder button and display      
        self.btn = QtGui.QPushButton('Find Folder', self)
        self.btn.clicked.connect(self.loadData)
        self.btn.setAutoDefault(False)

        self.file_display=QtGui.QLineEdit(self)
        self.file_display.setReadOnly(True)
        
        #make the checkbox
        self.check=QtGui.QCheckBox("Plot Data Used to Fit Template")
        self.check.stateChanged.connect(lambda:self.plot())
        
        #make third checkbox
        self.check3=QtGui.QCheckBox("Plot Zero Noise Filter")
        self.check3.stateChanged.connect(lambda:self.plot())
        
        #make second checkbox
        self.check2=QtGui.QCheckBox("Plot Scaled Fourier Transform of Filter")
        self.check2.stateChanged.connect(lambda:self.plot())

        #make the slider and pixel input
        self.slider_label=QtGui.QLabel("Select Pixel Number")

        self.slider=QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMaximum(1)
        self.slider.setValue(0)
        self.slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.sliderchange)

        self.pixel_input=QtGui.QLineEdit(self)
        self.pixel_input.setText('0')
        self.pixel_input.setFixedWidth(40)
        self.pixel_input.returnPressed.connect(self.inputchange)

        #make file name display
        self.file_label=QtGui.QLabel("File Name:")

        self.name_display=QtGui.QLineEdit(self)
        self.name_display.setReadOnly(True)

        #make log file display
        self.log_label=QtGui.QLabel("Log File:")
        
        self.log_display=QtGui.QTextEdit(self)
        self.log_display.setReadOnly(True)
        self.log_display.setFixedHeight(100)

        #define options panel geometry
        filebox = QtGui.QHBoxLayout()
        filebox.addWidget(self.btn)
        filebox.addWidget(self.file_display)
        
        checkbox = QtGui.QHBoxLayout()
        checkbox.addWidget(self.check)
        checkbox.addStretch()
        
        checkbox3=QtGui.QHBoxLayout()
        checkbox3.addWidget(self.check3)
        checkbox3.addStretch()

        checkbox2=QtGui.QHBoxLayout()
        checkbox2.addWidget(self.check2)
        checkbox2.addStretch()
        
        labelbox1=QtGui.QHBoxLayout()
        labelbox1.addWidget(self.slider_label)
        labelbox1.addStretch()

        sliderbox = QtGui.QHBoxLayout()
        sliderbox.addWidget(self.slider)
        sliderbox.addWidget(self.pixel_input)

        labelbox2=QtGui.QHBoxLayout()
        labelbox2.addWidget(self.file_label)
        labelbox2.addStretch()
        
        labelbox3=QtGui.QHBoxLayout()
        labelbox3.addWidget(self.log_label)
        labelbox3.addStretch()
        
        self.options = QtGui.QVBoxLayout()
        self.options.addLayout(filebox)
        self.options.addLayout(checkbox)
        self.options.addLayout(checkbox3)
        self.options.addLayout(checkbox2)
        self.options.addLayout(labelbox1)
        self.options.addLayout(sliderbox)
        self.options.addLayout(labelbox2)
        self.options.addWidget(self.name_display)
        self.options.addStretch()
        self.options.addLayout(labelbox3)
        self.options.addWidget(self.log_display)
        self.options.addStretch()
        

        # a figure instance to plot on
        self.figure1 = plt.figure()
        self.figure2 = plt.figure()
        self.figure3 = plt.figure()

        self.ax1=self.figure1.add_subplot(111)
        self.ax2=self.figure2.add_subplot(111)
        self.ax3=self.figure3.add_subplot(111)

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas1 = FigureCanvas(self.figure1)
        self.canvas2 = FigureCanvas(self.figure2)
        self.canvas3 = FigureCanvas(self.figure3)
  
        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar1 = NavigationToolbar(self.canvas1, self)
        self.toolbar2 = NavigationToolbar(self.canvas2, self)
        self.toolbar3 = NavigationToolbar(self.canvas3, self)

        # Just some button connected to `plot` method
        self.button = QtGui.QPushButton('Plot')
        self.button.clicked.connect(self.plot)

        # set the layout
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.toolbar1,1,1)
        grid.addWidget(self.toolbar2,1,2)
        grid.addWidget(self.toolbar3,3,1)
        grid.addWidget(self.canvas1,2,1)
        grid.addWidget(self.canvas2,2,2)
        grid.addWidget(self.canvas3,4,1)
        grid.addLayout(self.options,3,2,4,2)

        self.setGeometry(300, 30, 1000, 900)
        self.setWindowTitle('Plot Filters')
    
        self.show()
    def loadData(self):
        reformated=False
        self.directory = QtGui.QFileDialog.getExistingDirectory(self, 'Select a folder:', 
                '/mnt/data0/Darkness',QtGui.QFileDialog.ShowDirsOnly)
        self.f_template_coef = os.path.join(str(self.directory),'template_coefficients.txt')
        self.f_filter_coef= os.path.join(str(self.directory),'filter_coefficients.txt')
        self.f_noise= os.path.join(str(self.directory),'noise_data.txt')
        self.f_rough_templates= os.path.join(str(self.directory),'rough_templates.txt')
        self.f_type= os.path.join(str(self.directory),'filter_type.txt')
        self.f_list= os.path.join(str(self.directory),'file_list.txt')
        self.f_log= os.path.join(str(self.directory),'log_file.txt')
        self.f_filters_fourier=os.path.join(str(self.directory),'filters_fourier.txt')

        self.template_coefficients = np.loadtxt(self.f_template_coef)
        self.filter_coefficients=np.loadtxt(self.f_filter_coef)
        self.noise_data=np.loadtxt(self.f_noise)
        self.filter_type=np.loadtxt(self.f_type)
        self.rough_templates=np.loadtxt(self.f_rough_templates)
        self.filters_fourier=np.loadtxt(self.f_filters_fourier)
        
        with open (self.f_list, 'rb') as fp:
            self.file_list = pickle.load(fp)
        with open(self.f_log,'rU') as f:
            self.log_file=f.readlines()
        
        #calculate filter that would be used there was no noise won't work if only one set
        try:
            self.template_filter=-self.template_coefficients/(np.dot(np.atleast_2d(np.einsum('ij,ij->i', self.template_coefficients, self.template_coefficients)).transpose(),np.ones((1,np.shape(self.template_coefficients)[1]))))
            self.template_fft=np.abs(np.fft.fft(self.template_filter))**2
        except:
            pass
        #reformat arrays if 1D
        if len(np.shape(self.template_coefficients))==1:
            #recalculate filter
            self.template_filter=np.row_stack((-self.template_coefficients/(np.dot(self.template_coefficients,self.template_coefficients)),np.zeros(len(self.filter_coefficients))))            
            self.template_fft=np.row_stack((np.abs(np.fft.fft(self.template_filter))**2,np.zeros(len(self.filter_coefficients))))

            self.template_coefficients=np.row_stack((self.template_coefficients,np.zeros(len(self.template_coefficients))))
            self.filter_coefficients=np.row_stack((self.filter_coefficients,np.zeros(len(self.filter_coefficients))))
            self.noise_data=np.row_stack((self.noise_data,np.zeros(len(self.noise_data))))
            self.rough_templates=np.row_stack((self.rough_templates,np.zeros(len(self.rough_templates))))
            self.filter_type=np.array([self.filter_type,0])
            self.filters_fourier=np.row_stack((self.filters_fourier,np.zeros(len(self.filters_fourier))))
            reformated=True

        self.log_display.setText("".join(self.log_file[-4:]))
        self.noise_freq=np.fft.fftfreq(np.size(self.noise_data[0,:]),d=1e-6)
        self.filters_freq=np.fft.fftfreq(np.size(self.filters_fourier[0,:]),d=1e-6)
        self.file_display.setText(self.directory)
        self.pixel=0
        if reformated:
            self.slider.setMaximum(len(self.filter_type)-2)
        else:
            self.slider.setMaximum(len(self.filter_type)-1)
        self.name_display.setText(str(self.file_list[self.pixel]))

        self.plot()

    def plot(self):
        self.ax1.cla()
        self.ax2.cla()        
        self.ax3.cla()

        self.ax1.plot(self.template_coefficients[self.pixel,:],'b',label='final template')
        if self.check.isChecked() == True:
            self.ax1.plot(self.rough_templates[self.pixel,:],'r',label='data fitted')

        self.ax2.plot(self.filter_coefficients[self.pixel,:],'b',label='final filter')  

        if self.check3.isChecked()==True:
            self.ax2.plot(self.template_filter[self.pixel,],'k',label='zero noise filter')
  
        if np.sum(self.noise_data[self.pixel,:]>0)>0:       
                self.ax3.loglog(self.noise_freq,self.noise_data[self.pixel,:],'b',label='noise PSD')
                self.ax3.legend(fontsize=10,loc='upper center', bbox_to_anchor=(0.5, 1.13))
        if self.check2.isChecked() == True and np.sum(self.filters_fourier[self.pixel,:]>0)>0 and np.sum(self.template_fft[self.pixel,:]>0)>0:
            scale=self.noise_data[self.pixel,:][-1]/self.template_fft[self.pixel,:][-1]
            if scale==0:
                scale=1
            self.ax3.loglog(self.filters_freq,self.filters_fourier[self.pixel,:]*scale,'r',label='final filter fft squared')
            self.ax3.loglog(self.filters_freq,self.template_fft[self.pixel,:]*scale,'k',label='zero noise filter fft squared')
            self.ax3.legend(fontsize=10,loc='upper center', bbox_to_anchor=(0.5, 1.13))

        self.ax1.legend(loc='right',fontsize=10)
        self.ax2.legend(loc='upper right',fontsize=10)


        self.check_type()
        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()

    def check_type(self):
        if self.filter_type[self.pixel]==0:
            self.ax1.text(0.5, 0.01, 'Default Template Used',
                          verticalalignment='bottom', horizontalalignment='center',
                          transform=self.ax1.transAxes,
                          color='red', fontsize=15)
            self.ax2.text(0.5, 0.01, 'Default Filter Used',
                          verticalalignment='bottom', horizontalalignment='center',
                          transform=self.ax2.transAxes,
                          color='red', fontsize=15)
            self.ax3.text(0.5, 0.01, 'Noise Spectrum Could Not Be Generated',
                          verticalalignment='bottom', horizontalalignment='center',
                          transform=self.ax3.transAxes,
                          color='red', fontsize=15)
        elif self.filter_type[self.pixel]==1:
            self.ax2.text(0.5, 0.01, 'Using Calculated Template as Filter',
                          verticalalignment='bottom', horizontalalignment='center',
                          transform=self.self.ax2.transAxes,
                          color='red', fontsize=15)
            self.ax3.text(0.5, 0.01, 'Noise Spectrum Could Not Be Generated',
                          verticalalignment='bottom', horizontalalignment='center',
                          transform=self.ax3.transAxes,
                          color='red', fontsize=15)
        elif self.filter_type[self.pixel]==2:
            self.ax1.text(0.5, 0.01, 'Default Template Used',
                          verticalalignment='bottom', horizontalalignment='center',
                          transform=self.ax1.transAxes,
                          color='red', fontsize=15)
    def sliderchange(self):
        self.pixel=self.slider.value()
        self.pixel_input.setText(str(self.pixel))
        self.name_display.setText(str(self.file_list[self.pixel]))
        self.plot()
    def inputchange(self):
        self.pixel=int(self.pixel_input.text())
        self.slider.setValue(self.pixel)
        self.name_display.setText(str(self.file_list[self.pixel]))
        self.plot()

        


        
 
def main(*args, **kwargs):
    app = QtGui.QApplication(sys.argv)
    ex = Window()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

import glob
import inspect
import os
import sys
import time
import traceback
from functools import partial

import numpy as np
from PyQt4 import QtCore, QtGui

import makeFilters as mF
import processData as pD

reload(pD)

class WorkerSignals(QtCore.QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        `tuple` (integer for dataset, timing info, bool for killed or not, percent complete)

    error
        `tuple` (exctype, value, traceback.format_exc() )

    result
        `object` data returned from processing, anything

    progress
        `tuple`  (float indicating % progress, index for dataset)

    '''
    finished = QtCore.pyqtSignal(tuple)
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object) 
    progress = QtCore.pyqtSignal(tuple)
    interrupt=QtCore.pyqtSignal(bool) 


class Worker(QtCore.QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and 
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        kwargs['progress_callback'] = self.signals.progress

    @QtCore.pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
            worked=True
        except:
            #traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((self.kwargs['dataset'], value,exctype, traceback.format_exc()))
            worked=False
        else:
            pass
            #self.signals.result.emit(result)  # Return the result of the processing
        finally:
            if worked:
                self.signals.finished.emit((result,worked))  # Done
            else:
                self.signals.finished.emit(((self.kwargs['dataset'], 0, False, 0),worked))



class MainWindow(QtGui.QMainWindow):


    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        #make the folder finder button, display, filter type and plotting tool      
        self.btn = QtGui.QPushButton('Find Folder', self)
        self.btn.clicked.connect(self.getDirectory)
        self.btn.setAutoDefault(False)

        self.file_display=QtGui.QLineEdit(self)
        self.file_display.setReadOnly(True)

        self.comboBox=QtGui.QComboBox(self)
        self.filterTypes=zip(*inspect.getmembers(mF, inspect.isfunction))
        index0=0
        for ind, filterType in enumerate(self.filterTypes[0]):
            self.comboBox.addItem(filterType)
            try:
                smalldoc=self.filterTypes[1][ind].__doc__.split('\n')
                smalldoc=map(str.strip,smalldoc[1:])
                smalldoc='\n'.join(smalldoc)
                smalldoc=smalldoc.split('\n\n')[0]
                self.comboBox.setItemData(ind,smalldoc,QtCore.Qt.ToolTipRole)
            except: pass
            if filterType=='wienerFilter':
                index0=ind
        self.filterFunction=self.filterTypes[1][index0]
        self.filterName=self.filterTypes[0][index0]
        self.comboBox.setCurrentIndex(index0)
        self.comboBox.currentIndexChanged.connect(self.change_filter)
        
        #plotting tool button
        self.btn2=QtGui.QPushButton('Open Plotting Tool',self)
        self.btn2.clicked.connect(self.openPlotting)
        self.btn.setAutoDefault(False)
        
        #define filebar panel geometry
        self.filebar = QtGui.QHBoxLayout()
        self.filebar.addWidget(self.btn)
        self.filebar.addWidget(self.file_display)
        self.filebar.addWidget(self.comboBox)
        self.filebar.addWidget(self.btn2)

               
        #define control panel geometry    
        spacer=QtGui.QSpacerItem(350, 0)
        
        self.control=QtGui.QVBoxLayout()
        self.control.addItem(spacer)


        # set the layout
        grid = QtGui.QGridLayout()
        grid.addLayout(self.filebar,1,1,1,2)
        grid.addLayout(self.control,2,1,3,2)

        w=QtGui.QWidget()
        w.setLayout(grid)
        self.setCentralWidget(w)
    
        self.setGeometry(300, 30, 800, 400)
        self.setWindowTitle('Optimal Filter GUI')


        self.show()

        self.threadpool = QtCore.QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())


    def getDirectory(self):
        #remove previous kill files if changing directories
        if hasattr(self,'directory'):
            for f in glob.glob(os.path.join(self.directory,'kill_processes*.txt')):
                os.remove(f)
        
        #get directory
        self.directory = str(QtGui.QFileDialog.getExistingDirectory(self, 'Select a folder:', 
                '/mnt/data0/Darkness',QtGui.QFileDialog.ShowDirsOnly))
        self.file_display.setText(self.directory)
        
        #update control pannel with directories
        self.subdirectories=[d for d in os.listdir(self.directory) if os.path.isdir(os.path.join(self.directory,d))]
        
        if len(self.subdirectories)>25:
            raise(ValueError('Too many directories in chosen folder'))

        #remove any previously created rows
        if hasattr(self,'row'):
            self.clearLayout(self.control)
        #initialize row, label, check, run_button, kill_button, and progress
        self.row=[]
        self.row_label=[]
        self.check=[]
        self.run_button=[]
        self.kill_button=[]
        self.progress=[]
                
        #loop through folders
        for index, directory in enumerate(self.subdirectories):            
            #make row elements
            self.row_label.append(QtGui.QLabel(directory+':'))

            self.check.append(QtGui.QCheckBox("Continuing?"))
            
            self.run_button.append(QtGui.QPushButton('Run', self))
            self.run_button[index].clicked.connect(partial(self.run_filter,index))
            self.run_button[index].setAutoDefault(False)
            
            self.kill_button.append(QtGui.QPushButton('Kill', self))
            self.kill_button[index].clicked.connect(partial(self.kill_filter,index))
            self.kill_button[index].setDisabled(True)

            self.progress.append(QtGui.QLineEdit(self))
            self.progress[index].setReadOnly(True)
            
            spacer=QtGui.QSpacerItem(50, 0)
    
            #add elements to row
            self.row.append(QtGui.QHBoxLayout())
            self.row[index].addWidget(self.row_label[index])
            self.row[index].addItem(spacer)
            self.row[index].addWidget(self.check[index])
            self.row[index].addWidget(self.run_button[index])
            self.row[index].addWidget(self.kill_button[index])
            self.row[index].addWidget(self.progress[index])

            #add row to control pannel
            self.control.addLayout(self.row[index])
            
            #set warning text if filters already exist
            if os.path.isfile(os.path.join(os.path.join(self.directory,directory),"log_file.txt")):
                self.progress[index].setText("WARNING: Filters already exist for this dataset")
    
            #create kill file for row
            filename=os.path.join(self.directory,'kill_processes'+str(index)+'.txt')
            np.savetxt(filename,np.array([0]))

        #add last row for running everything
        self.check_all=QtGui.QCheckBox("Check All")
        self.check_all.stateChanged.connect(self.check_all_boxes)

        self.run_all=QtGui.QPushButton("Run All",self)
        self.run_all.clicked.connect(self.run_all_filters)
        self.run_all.setAutoDefault(False)

        self.kill_all=QtGui.QPushButton("Kill All",self)
        self.kill_all.clicked.connect(self.kill_all_filters)
        self.kill_all.setAutoDefault(False)

        self.recalculate_filters=QtGui.QPushButton("Recalculate Filters Only")
        self.recalculate_filters.clicked.connect(self.run_all_recalculations)
        self.recalculate_filters.setAutoDefault(False)
        
        self.last_row=QtGui.QHBoxLayout()
        self.last_row.addStretch()
        self.last_row.addWidget(self.check_all)
        self.last_row.addWidget(self.run_all)
        self.last_row.addWidget(self.kill_all)
        self.last_row.addWidget(self.recalculate_filters)
        self.last_row.addStretch()
        
        self.control.addLayout(self.last_row)

        #add final stretch        
        self.control.addStretch()     
        
        #create is running attribute
        self.isrunning=np.zeros(len(self.row),dtype=bool)

    def kill_filter(self,index):
        filename=os.path.join(self.directory,'kill_processes'+str(index)+'.txt')
        working=False
        while not working:
            try:           
                np.savetxt(filename,np.array([1]))
                working=True
            except:
                time.sleep(0.1)

    def progress_fn(self, n):
        self.progress[n[1]].setText("{0}% done".format(round(n[0],1)))
    
    def error_fn(self,n):
        self.progress[n[0]].setText('error: ' + str(n[1]))

    def thread_complete(self,results):
        worked=results[1]
        results=results[0]        
    
        #unlock run button
        self.run_button[results[0]].setEnabled(True)
        #lock kill button
        self.kill_button[results[0]].setDisabled(True)       
        
        #print completion status
        if results[2] and worked:
            self.progress[results[0]].setText("Filters Complete after {0} minutes!".format(round(results[1]/60.0,1)))
        elif worked and not results[2]:
            self.progress[results[0]].setText("Calculation Stopped at {0}% after {1} minutes!".format(round(results[3],1),round(results[1]/60.0,1)))

        #remove kill setting
        filename=os.path.join(self.directory,'kill_processes'+str(results[0])+'.txt')
        working=False
        while not working:
            try:         
                np.savetxt(filename,np.array([0]))
                working=True
            except:
                time.sleep(0.1)

        self.isrunning[results[0]]=False 
        time.sleep(0.1)
          
    def run_filter(self,index,isAll=False):
        #check if files exist
        if (not isAll) and os.path.isfile(os.path.join(self.directory,self.subdirectories[index],self.filterName +'_coefficients.txt')) and (not self.check[index].isChecked()):
            question="Filters of type '" + self.filterName + "' have already been calculated. Are you sure you would like to delete and restart?"
            answer=self.query_yes_no(question)
            if not answer:
                return

        #clear text box
        self.progress[index].setText('')
        #grey out run button and make unresponsive       
        self.run_button[index].setDisabled(True)
        #unlock kill button
        self.kill_button[index].setEnabled(True)        
        
        #set as running
        self.isrunning[index]=True

        # Pass the function to execute
        path=os.path.join(self.directory,self.subdirectories[index])
        worker = Worker(execute_filter_calcs,dataset=index, mainDirectory=self.directory, continuing=self.check[index].isChecked(), directory=path,filterMethod=self.filterFunction) 
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)
        
        # Execute
        self.threadpool.start(worker)

    def run_recalculation(self,index):
        #clear text box
        self.progress[index].setText('')
        #grey out run button and make unresponsive       
        self.run_button[index].setDisabled(True)
        #unlock kill button
        self.kill_button[index].setEnabled(True) 

        #set as running
        self.isrunning[index]=True

        # Pass the function to execute
        path=os.path.join(self.directory,self.subdirectories[index])
        worker = Worker(execute_filters_only,dataset=index, mainDirectory=self.directory, directory=path,filterMethod=self.filterFunction) 
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)
        worker.signals.error.connect(self.error_fn)
        
        # Execute
        self.threadpool.start(worker)

    def run_all_recalculations(self):
        for index, row in enumerate(self.row):
            if not self.isrunning[index]:
                self.run_recalculation(index) 

    def run_all_filters(self):
        #check if files exist
        logic=False
        for index,subdirectory in enumerate(self.subdirectories):
            logic=logic or (os.path.isfile(os.path.join(self.directory,subdirectory,self.filterName +'_coefficients.txt')) and (not self.check[index].isChecked()))
        if logic:
            question="Filters of type '" + self.filterName + "' have already been calculated for some folders. Are you sure you would like to delete and restart?"
            answer=self.query_yes_no(question)
            if not answer:
                return
        for index, row in enumerate(self.row):
            if not self.isrunning[index]:
                self.run_filter(index, isAll=True)

    def check_all_boxes(self):
        for index, row in enumerate(self.row):
            if self.check_all.checkState():
                self.check[index].setChecked(True)
            else:
                self.check[index].setChecked(False)

    def kill_all_filters(self):
        for index, row in enumerate(self.row):
            if self.isrunning[index]:
                self.kill_filter(index)
                time.sleep(0.1)

    def closeEvent(self,event):
        if hasattr(self,'directory'):
            for f in glob.glob(os.path.join(self.directory,'kill_processes*.txt')):
                os.remove(f)
        
    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())
    def openPlotting(self):
        os.system("python plotFiltersGUI.py &")
    def change_filter(self,index):
        self.filterFunction=self.filterTypes[1][index]
        self.filterName=self.filterTypes[0][index]
    def query_yes_no(self,question):
        choice = QtGui.QMessageBox.question(self, 'Answer before continuing',
                                            question,
                                            QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if choice == QtGui.QMessageBox.Yes:
            return True
        else:
            return False
 
def execute_filter_calcs(progress_callback=[],dataset=[],mainDirectory=[],directory=[],continuing=False,filterMethod=mF.wienerFilter):
    defaultTemplate=np.loadtxt('template200_15us.txt') #change default template here
    result=pD.processData(directory,defaultTemplate,GUI=True, progress_callback=progress_callback, dataset=dataset, mainDirectory=mainDirectory, continuing=continuing,filterMethod=filterMethod)    
    return result   

def execute_filters_only(filterCode=[],progress_callback=[],dataset=[],mainDirectory=[],directory=[],filterMethod=mF.wienerFilter): 
    result=pD.recalculate_filters(directory,filterMethod,GUI=True, progress_callback=progress_callback, dataset=dataset, mainDirectory=mainDirectory)    
    return result   

app = QtGui.QApplication([])
window = MainWindow()
app.exec_()


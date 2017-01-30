import os
import numpy as np


dataDirectory='/mnt/data0/Darkness/20161118'
folder='112_filter_data'
folder='114_filter_data'
folder='115_filter_data'
#folder='116_filter_data' #bad
folder='117_filter_data'
#folder='118_filter_data' #bad
folder='121_filter_data'
#folder='122_filter_data' #bad
folders=['112_filter_data','115_filter_data','117_filter_data','121_filter_data']

finalTemplate=np.zeros((50,len(folders)))
for index, folder in enumerate(folders):

    directory=os.path.join(dataDirectory,folder)

    templates=np.loadtxt(os.path.join(directory,'template_coefficients.txt'))

    temp=templates[:,np.trapz(templates,axis=0)<-10]
    oscillating = np.abs(np.trapz(np.diff(np.transpose(temp))))
    meaned=np.mean(temp[:,oscillating<.15],axis=1)
    meaned /= np.max(np.abs(meaned))
    finalTemplate[:,index]=meaned

finalTemplate=np.mean(finalTemplate,axis=1)    
np.savetxt('template.txt',finalTemplate)

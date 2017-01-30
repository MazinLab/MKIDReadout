from multiprocessing import Pool
import os
import numpy as np
import processDataOneTemplate as pD

directory='/mnt/data0/Darkness/20161119/'
folders=['112_filter_data','115_filter_data','117_filter_data','121_filter_data']
template=np.loadtxt('template.txt')
defaultFilter=np.loadtxt('matched50_20.0us.txt')
def func(direct):
    pD.processData(direct,defaultFilter,template)
                
dirList=[]
for folder in folders:
    dirList.append(os.path.join(directory,folder))
workerPool = Pool()
workerPool.map_async(func, args=dirList, chunkSize=1)
workerPool.close()
workerPool.join()

import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit

start_time = time.time()

def doubleGaussian(x, s1, s2, c, a1, a2, x1, x2):
    return (c+ a1 * np.exp(-(((x-x1)**2)/(2*s1**2))) + a2*np.exp(-(((x-x2)**2)/(2*s2**2))))

def gaussian(x, s1, c, a1, x1):
    return (c+ a1 * np.exp(-(((x-x1)**2)/(2*s1**2))))




path="/mnt/data0/Darkness/20170917/Beammap/Beamlist_Master.txt"
outPath="/mnt/data0/MkidDigitalReadout/DataReadout/Setup/DARKNESS-Beammapping/giulia/"
neighFile="neighbours.p"
allDFile="distances.p"
angleFile="angle.p"
neighFileName=outPath+neighFile
allDFileName=outPath+allDFile
angleFileName=outPath+angleFile

runAll=0
if runAll:
    data=np.loadtxt(path)
    pairs=np.array([np.array([d[2],d[3]]) for d in data ])
    nearestNeigh=[]
    allD=[]
    angle=[]
    for ip,pair in enumerate(pairs):
        if ip%1000==0:
            print "calculated neighbours for ", ip, "locations" 
        if  np.sum(pair)!=0:
            distance=[]
            angRad=[]
            if ip<2000:
                for p in pairs[:2000]:
                    d=np.sqrt(np.sum((pair-p)**2))
                    distance.append(d) 
                    if (p[0]-pair[0])>=(p[1]-pair[1]):
                        angRad.append(np.arctan((p[1]-pair[1])/(p[0]-pair[0])))
                    else:
                        angRad.append(-np.arctan((p[0]-pair[0])/(p[1]-pair[1])))
            if np.logical_and(ip>=2000,ip<4000):
                for p in pairs[2000:4000]:
                    d=np.sqrt(np.sum((pair-p)**2))
                    distance.append(d) 
                    if (p[0]-pair[0])>=(p[1]-pair[1]):
                        angRad.append(np.arctan((p[1]-pair[1])/(p[0]-pair[0])))
                    else:
                        angRad.append(-np.arctan((p[0]-pair[0])/(p[1]-pair[1])))
            if np.logical_and(ip>=4000,ip<6000):
                for p in pairs[4000:6000]:
                    d=np.sqrt(np.sum((pair-p)**2))
                    distance.append(d) 
                    if (p[0]-pair[0])>=(p[1]-pair[1]):
                        angRad.append(np.arctan((p[1]-pair[1])/(p[0]-pair[0])))
                    else:
                        angRad.append(-np.arctan((p[0]-pair[0])/(p[1]-pair[1])))
            if np.logical_and(ip>=6000,ip<8000):
                for p in pairs[6000:8000]:
                    d=np.sqrt(np.sum((pair-p)**2))
                    distance.append(d) 
                    if (p[0]-pair[0])>=(p[1]-pair[1]):
                        angRad.append(np.arctan((p[1]-pair[1])/(p[0]-pair[0])))
                    else:
                        angRad.append(-np.arctan((p[0]-pair[0])/(p[1]-pair[1])))
            if ip>8000:
                for p in pairs[8000:]:
                    d=np.sqrt(np.sum((pair-p)**2))
                    distance.append(d) 
                    if (p[0]-pair[0])>=(p[1]-pair[1]):
                        angRad.append(np.arctan((p[1]-pair[1])/(p[0]-pair[0])))
                    else:
                        angRad.append(-np.arctan((p[0]-pair[0])/(p[1]-pair[1])))
            




            
            distance=np.array(distance)
            neighbours=distance[np.argsort(distance)[1:5]]
            angRad=np.array(angRad)
            angRad=angRad[np.argsort(distance)[1:5]]
            if ip<10:
                print neighbours
                print angRad
            nearestNeigh.append(neighbours)
            allD.append(distance)
            angle.append(angRad)
            #print "neighbours for pair", pair

    ###flattens the array
    nearestNeigh=np.array([n for neigh in nearestNeigh for n in neigh])
    allD=np.array([d for dist in allD for d in dist])
    angle=np.array([a for an in angle for a in an])
    pickle.dump(nearestNeigh,open(neighFileName,'wb'))
    pickle.dump(allD,open(allDFileName,'wb'))
    pickle.dump(angle,open(angleFileName,'wb'))
    print "--- %s seconds ---" % (time.time() - start_time)
else:
    nearestNeigh=pickle.load(open(neighFileName,'rb'))
    allD=pickle.load(open(allDFileName,'rb'))
    angle=pickle.load(open(angleFileName,'rb'))

print "max distance:", np.max(allD)
print "max neighbour distance:", np.max(nearestNeigh)
print "neighbours > 25", nearestNeigh[nearestNeigh>25]
hist, bin_edges= np.histogram(nearestNeigh[nearestNeigh<25],5000)
angHist, ang_bin_edges = np.histogram(angle,5000)

plt.figure(1000)
plt.plot(ang_bin_edges[:-1],angHist)
plt.show()


maxC=np.max(hist)
maxL=bin_edges[:-1][hist==maxC]
print maxC, maxL
partialHistX=bin_edges[bin_edges<1.8*maxL]
partialHistY=hist[bin_edges<1.8*maxL]


guess=([0,0,0,maxC*0.8,maxC/2,maxL*0.8,maxL],[3.5, 3.5, np.mean(partialHistY), maxC*1.5, maxC, maxL*1.3, maxL*1.6])
fitDGauss, pcov=curve_fit(doubleGaussian, partialHistX, partialHistY,bounds=guess)

angguess=([0,0,0,-0.08],[3.5, np.mean(angHist), np.max(angHist), 0.08])
fitAngGauss, pcovAng=curve_fit(gaussian, ang_bin_edges[:-1], angHist,bounds=angguess)

print "angle fit:", fitAngGauss,"angle", fitAngGauss[-1]

plt.figure(90)
plt.plot(ang_bin_edges[:-1],angHist,linestyle=":")

plt.plot(ang_bin_edges,gaussian(ang_bin_edges, *fitAngGauss))

plt.show()


print fitDGauss, 'peak loc:', fitDGauss[-2]

plt.figure(0)
plt.plot(bin_edges[:-1],hist,linestyle=":")
plt.figure(9)
plt.plot(partialHistX,partialHistY,linestyle=":")
locMax=np.array(argrelextrema(hist,np.greater))

plt.plot(partialHistX, doubleGaussian(partialHistX, *fitDGauss))

#print hist[locMax]
#print bin_edges[locMax]
plt.show()
"""
plt.hist(allD,bins=100)
plt.figure(1)
plt.hist(allD,bins=1000)
plt.figure(2)

plt.figure(3)
plt.hist(nearestNeigh[nearestNeigh<25],bins=5000)
plt.show()
"""





"""    for i in range(1,5):
        print "neighbour", i, pairs[neighbours[i]]
"""

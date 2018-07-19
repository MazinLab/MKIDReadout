from MKIDDigitalReadout.DataReadout.Setup.Beammap.mapcheckertesting import mapchecker
from MKIDDigitalReadout.DataReadout.Setup.Beammap.mapcheckertesting import feedline
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.optimize as opt



noah_design_feedline_path = r"C:\Users\njswi\PycharmProjects\BeammapPredictor\predictor\mec_feedline.txt"
design_feedline=np.loadtxt(noah_design_feedline_path)
noah_beammap_path = r"C:\Users\njswi\PycharmProjects\BeammapPredictor\predictor\beammapTestData\test\finalMap_20180605.txt"
noah_freqsweeps_path = r"C:\Users\njswi\PycharmProjects\BeammapPredictor\predictor\beammapTestData\test\ps_*"


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


feedlinearray=np.array([feedline1, feedline5, feedline6, feedline7, feedline8, feedline9, feedline10])
bestfitparameters=np.zeros((len(feedlinearray),2))
bestfitparametererrors=np.zeros((len(feedlinearray),2))

'''
analysisstarttime = time.time()
for i in range(len(feedlinearray)):
     params, errors = mapchecker.runfeedlinemcmc(feedlinearray[i])
     print(params,errors)
     bestfitparameters[i][0], bestfitparameters[i][1] = params
     bestfitparametererrors[i][0], bestfitparametererrors[i][1] = errors
analysisendtime = time.time()


print("Your analysis took {0:1.1f}".format((analysisendtime-analysisstarttime)/60),"minutes.")


residsbf = np.empty_like(feedlinearray)
residsu = np.empty_like(feedlinearray)
stddevbfarray = np.zeros(len(feedlinearray))
stddevuarray = np.zeros(len(feedlinearray))
avgbfarray = np.zeros(len(feedlinearray))
avguarray = np.zeros(len(feedlinearray))

for i in range(len(feedlinearray)):
    residsbf[i], residsu[i], stddevbfarray[i], stddevuarray[i], avgbfarray[i], \
    avguarray[i] = mapchecker.feedlineprocessor(bestfitparameters[i], feedlinearray[i])

for i in range(len(feedlinearray)):
    plt.figure(i+1)
    plt.hist(residsbf[i],bins=25,alpha=0.5,label='Data Modified')
    plt.hist(residsu[i], bins=25, alpha=0.5, label='Data Unmodified')
    plt.xlabel("Residual Distance (MHz)")
    plt.ylabel("Counts")
    plt.title(feedlinearray[i].name)
    plt.legend()
    plt.show()
'''

designfreqs = design_feedline.flatten()
measfreqs8 = feedline8.normfreqs.flatten()

designfreqs = designfreqs[~np.isnan(measfreqs8)]
measfreqs8 = measfreqs8[~np.isnan(measfreqs8)]

'''Based on a model for our data fit to a polynomial find the residuals between the measured data and the model'''
def residuals(parameters, feedlineobject, model):
    p = np.poly1d(parameters)
    ydata, modeldata = flattendata(feedlineobject,model)
    err = ydata - p(modeldata)
    return err


'''Because we are working in a space where we do not know the ideal number of parameters to fit, create a function
where we can specify the order of the polynomial we want to describe our model with, which will then let us find out
what order polynomial allows us to minimize our residuals by nonlinear least squares regression (maximimizing likelihood)'''
def initialparamguesser(feedlineobject, model, order):
    ydata, modeldata = flattendata(feedlineobject, model)
    params = np.polyfit(modeldata, ydata, order, full=True)[0]
    return params


''' Create a function which flattens the data to conveniently work in frequency space, ensuring that the model data and
measured data match each other, which is to say that at a given index in the 1D array, the ydata array shows the frequency
that was measured at a given position, will at the same index, the modeldata array gives the design frequency at the same
coordinate. '''
def flattendata(feedlineobject, model):
    ydata = feedlineobject.normfreqs.flatten()
    modeldata = model.flatten()
    modeldata = modeldata[~np.isnan(ydata)]
    ydata = ydata[~np.isnan(ydata)]
    return ydata, modeldata


def feedlinefitter(feedlineobj,modelfeedline):
    chisquarevals = np.zeros(70)
    for i in range(len(chisquarevals)):
        pguess = initialparamguesser(feedlineobj, modelfeedline, i)
        lssqsol = opt.least_squares(residuals, pguess, args=(feedlineobj, modelfeedline))
        chisquarevals[i] = np.sum((lssqsol.fun)**2)**(1/2)
    minchisquare = np.min(chisquarevals)
    bestfitorder = np.where(chisquarevals == minchisquare)[0]

    bestguess = initialparamguesser(feedlineobj, modelfeedline, bestfitorder)
    leastsquaressolution = opt.least_squares(residuals, bestguess, args=(feedlineobj, modelfeedline))
    return leastsquaressolution, bestfitorder


try1, order = feedlinefitter(feedline1, design_feedline)
print("The order of best fit to the data is:", order)
rd, md=flattendata(feedline1, design_feedline)

plt.subplots(2,1)
plt.subplot(211)
plt.scatter(rd, md, label='measured data', marker='.')
plt.scatter(rd, rd, label='ideal data', marker='.')
plt.legend()
plt.ylabel('Measured Frequency (MHz)')
plt.subplot(212)
plt.scatter(md, rd-md, label='unfit data', marker='.')
plt.scatter(md, (-1)*try1.fun, label='fitted data', marker='.')
plt.legend()
plt.xlabel('Design Frequency (MHz)')
plt.ylabel('Residual Distance (MHz)')
plt.title('Feedline 1')

plt.figure(2)
plt.hist((rd-md), bins=30, label='unfit data', alpha=0.7)
plt.hist((-1)*try1.fun, bins=30, label='fitted data', alpha=0.7)
plt.legend()
plt.xlabel('Residual Distance (MHz)')
plt.ylabel('Counts')
plt.title('Feedline 1')
plt.show()
''' This block of code was used in development to see what order polynomial we would have to go to get our best fit 
csvals=np.zeros((len(feedlinearray),30))
stnl=time.time()
for i in range(len(csvals)):
    print(i,'!!!')
    for j in range(len(csvals[i])):
        pguess = initialparamguesser(feedlinearray[i],design_feedline,j)
        firstls = opt.least_squares(residuals,pguess,args=(feedlinearray[i],design_feedline))
        coeff = firstls.x
        csvals[i][j] = np.sum((firstls.fun)**2)**(1/2)
etnl=time.time()
print("Non-linear least squares fitting took {0:1.1f}".format((etnl-stnl)/60),"minutes")

mins = np.amin(csvals,axis=1)
idealorder = np.zeros(len(mins))
for i in range(len(mins)):
    idealorder[i] = np.where(csvals[i] == mins[i])[0]

print(idealorder)

plt.scatter(range(len(csvals[0])), csvals[0], label='feedline 1', marker='.')
plt.scatter(range(len(csvals[0])), csvals[1], label='feedline 5', marker='.')
plt.scatter(range(len(csvals[0])), csvals[2], label='feedline 6', marker='.')
plt.scatter(range(len(csvals[0])), csvals[3], label='feedline 7', marker='.')
plt.scatter(range(len(csvals[0])), csvals[4], label='feedline 8', marker='.')
plt.scatter(range(len(csvals[0])), csvals[5], label='feedline 9', marker='.')
plt.scatter(range(len(csvals[0])), csvals[6], label='feedline 10', marker='.')
plt.legend()
plt.xlabel("Order of fit")
plt.ylabel("Chi Square")
plt.show()'''
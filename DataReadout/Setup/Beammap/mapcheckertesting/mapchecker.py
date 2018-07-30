
# coding: utf-8
'''
 FLAG DEFINITIONS (Consider moving elsewhere?)
 0: Good
 1: Pixel not read out
 2: Beammap failed to place pixel
 3: Succeeded in x, failed in y
 4: Succeeded in y, failed in x
 5: Multiple locations found for pixel
 6: Beammap placed the pixel in the wrong feedline
'''
import numpy as np
import emcee # Used in early development, don't delete until guaranteed not to use this package
import scipy.optimize as opt


design_feedline_path = r"C:\Users\njswi\PycharmProjects\BeammapPredictor\predictor\mec_feedline.txt"
design_feedline = np.loadtxt(design_feedline_path)


'''Based on a model for our data fit to a polynomial find the residuals between the measured data and the model.
This is formatted to be used in the non-linear least squares regression code block'''
def residuals(parameters, feedlineobject, model):
    p = np.poly1d(parameters)
    ydata, modeldata = flattendata(feedlineobject,model)
    err = ydata - p(modeldata)
    return err


'''Because we are working in a space where we do not know the ideal number of parameters to fit, create a function
where we can specify the order of the polynomial we want to describe our model with, which will then let us find out
what order polynomial allows us to minimize our residuals by nonlinear least squares regression (maximimizing likelihood)
NOTE: This works in conjunction with the feedlinefitter function based on the order it is given, which can be specific
or run through a large number of orders to see which fits best'''
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

'''This returns the nonlinear least squares object from the Scipy optimize package, and we use two of the return
values: .x is the array of coefficients for the best fit and .fun is the actual array of residuals. More often we 
use the .fun method so we can see how well we were able to fit our data'''
def feedlinefitter(feedlineobj, modelfeedline, order = None):
    if order == None:
        chisquarevals = np.zeros(30)
        for i in range(len(chisquarevals)):
            pguess = initialparamguesser(feedlineobj, modelfeedline, i)
            lssqsol = opt.least_squares(residuals, pguess, args=(feedlineobj, modelfeedline))
            chisquarevals[i] = np.sum((lssqsol.fun)**2)**(1/2)
        minchisquare = np.min(chisquarevals)
        bestfitorder = np.where(chisquarevals == minchisquare)[0]

        bestguess = initialparamguesser(feedlineobj, modelfeedline, bestfitorder)
        leastsquaressolution = opt.least_squares(residuals, bestguess, args=(feedlineobj, modelfeedline))
        return leastsquaressolution, bestfitorder
    else :
        bestguess = initialparamguesser(feedlineobj, modelfeedline, order)
        leastsquaressolution = opt.least_squares(residuals, bestguess, args=(feedlineobj, modelfeedline))
        return leastsquaressolution, order


# Given a feedline, design feedline, and whatever order you wish your least squares to be (None gives the best fit
# below order 30) and returns the measured frequencies, the frequencies fit by the least squares regression, the model
# frequencies, and the residuals
def leastsquaremethod(feedlineobj, modelfeedline, order = None):
    results, best_order = feedlinefitter(feedlineobj, modelfeedline,order)
    fitted_coeffs = np.poly1d(results.x) # Consider if this is necessary
    realdata, modeldata = flattendata(feedlineobj, design_feedline)
    fitteddata = fitted_coeffs(modeldata)
    residualvalues = results.fun
    return realdata, fitteddata, modeldata, residualvalues


# MCMC framework
'''
Abandoned MCMC attempt - Keep here for the foreseeable future, although it is likely obsolete

def residualmaker(params, feedlineobject):

    designmap=design_feedline
    frequencyarray=frequency_modifier(params,feedlineobject)
    residuals=frequencyarray-designmap
    flags=np.copy(feedlineobject.data[:, :, 2])
    for i in range(len(residuals)):
        for j in range(len(residuals[i])):
            if flags[i][j] != 0 :
                residuals[i][j]=float('NaN')
    return residuals


def frequency_modifier (params, feedlineobject):
    stretchparam = abs(params[0])
    shiftparam = params[1]
    norm = np.copy(feedlineobject.normfreqs)
    stretchedfreqs = norm * stretchparam
    shiftedfreqs = stretchedfreqs + shiftparam
    return shiftedfreqs


def residualstddev (residualarray):
    residuals = residualarray[~np.isnan(residualarray)]
    return np.std(residuals)


def residualavg (residualarray):
    residuals = residualarray[~np.isnan(residualarray)]
    return np.average(residuals)


def feedlineprocessor (modifyingparameters, feedlineobject):
    bestfitresiduals = residualmaker(modifyingparameters, feedlineobject)
    unmodifiedresiduals = residualmaker([1, 0], feedlineobject)
    stddevfrombestfit = residualstddev(bestfitresiduals)
    stddevunmodified = residualstddev(unmodifiedresiduals)
    avgfrombestfit = residualavg(bestfitresiduals)
    avgunmodified = residualavg(unmodifiedresiduals)
    return bestfitresiduals[~np.isnan(bestfitresiduals)], unmodifiedresiduals[~np.isnan(unmodifiedresiduals)],\
           stddevfrombestfit, stddevunmodified, avgfrombestfit, avgunmodified


# Params should be of the form [stretch parameter, shift parameter]
def chisquare (params, feedlineobject):
    residuals = residualmaker(params, feedlineobject)
    chisqu = (np.sum(residuals[~np.isnan(residuals)]**2))**(1/2)
    return chisqu


def ln_like (params, feedlineobject):
    return -2*chisquare(params, feedlineobject)


def ln_prior (params):
    if params[0] > 0:
        return 0.0
    return -np.inf


def ln_prob (pvals, feedlineobject):
    lp = ln_prior(pvals)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_like(pvals, feedlineobject)


def runfeedlinemcmc(feedlineobject):
    print("Starting new feedline...")
    ndim=2
    nwalkers=10
    initialguess=[1,0]
    p0 = [initialguess + 1e-4 * np.random.rand(ndim) for i in range(nwalkers)]
    sampler=emcee.EnsembleSampler(nwalkers, ndim, ln_prob, args=[feedlineobject])

    # Run a burn-in.
    print("Burning-in ...")
    t1 = time.time()
    pos, prob, state = sampler.run_mcmc(p0, 600)
    t2= time.time()
    print("Burn-in took {0:1.2}".format((t2-t1)/60),"minutes.")

    # Reset the chain to remove the burn-in samples.
    sampler.reset()

    # Starting from the final position in the burn-in chain, sample for 1500
    # steps. (rstate0 is the state of the internal random number generator)
    print("Running MCMC...")
    t3=time.time()
    pos, prob, state = sampler.run_mcmc(pos, 5000, rstate0=state)
    t4=time.time()
    print("MCMC took {0:1.2f}".format((t4-t3)/60),"minutes.")

    # Get the index with the highest probability
    maxprob_index = np.argmax(prob)

    # Get the best parameters and their respective errors
    params_fit = pos[maxprob_index]
    errors_fit = [sampler.flatchain[:, i].std() for i in range(ndim)]
    print("Done with feedline!")
    return params_fit,errors_fit
'''


# MCMC execution
'''
This was part of the MCMC attempt, where the first block was the code itself, this was the block used to
run the full MCMC which only took into account a global shift and global stretch (non-localized)

bestfitparameters=np.zeros((len(feedlinearray),2))
bestfitparametererrors=np.zeros((len(feedlinearray),2))

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


# Design array from design feedline (could be useful, probably not, keep it here until definitely not needed)
'''
Create an array map where each element is the design freqency at a given pixel coordinate
Essentially, copy the 14-by-146 design feedline 10 times side-by-side into a 140-by-146 array

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


# Development of non-linear regression analysis
''' 
This block of code was used in development to see what order polynomial we would have to go to get our best fit.
This is part of what we are now using in the non-linear least squares code, but clunkier, we've streamlined it
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
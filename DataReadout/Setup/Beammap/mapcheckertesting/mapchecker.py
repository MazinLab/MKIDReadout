
# coding: utf-8
'''
 FLAG DEFINITIONS
 0: Good
 1: Pixel not read out
 2: Beammap failed to place pixel
 3: Succeeded in x, failed in y
 4: Succeeded in y, failed in x
 5: Multiple locations found for pixel
 6: Beammap placed the pixel in the wrong feedline
'''
import numpy as np
import emcee
import time
import matplotlib.pyplot as plt

noah_design_feedline_path = r"C:\Users\njswi\PycharmProjects\BeammapPredictor\predictor\mec_feedline.txt"
design_feedline=np.loadtxt(noah_design_feedline_path)

'''
Abandoned MCMC package - keep until better process is found

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
    return -0.5*chisquare(params, feedlineobject)


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


'''
Come back to this later if necessary, used during development to analyze a full array (as opposed to only feedlines)

Create an array map where each element is the design freqency at a given pixel coordinate
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
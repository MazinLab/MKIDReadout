
# coding: utf-8
"""
 FLAG DEFINITIONS (Consider moving elsewhere?)
 0: Good
 1: Pixel not read out
 2: Beammap failed to place pixel
 3: Succeeded in x, failed in y
 4: Succeeded in y, failed in x
 5: Multiple locations found for pixel
 6: Beammap placed the pixel in the wrong feedline
"""
import numpy as np
import scipy.optimize as opt


design_feedline_path = r"mec_feedline.txt"
design_feedline = np.loadtxt(design_feedline_path)


def residuals(parameters, feedlineobject, model):
    """
    Based on a model for our data fit to a polynomial find the residuals between the measured data and the model.
    This is formatted to be used in the non-linear least squares regression code block
    """
    p = np.poly1d(parameters)
    ydata, modeldata = flattendata(feedlineobject,model)
    err = ydata - p(modeldata)
    return err


def initialparamguesser(feedlineobject, model, order):
    """
    Because we are working in a space where we do not know the ideal number of parameters to fit, create a function
    where we can specify the order of the polynomial we want to describe our model with, which will then let us find out
    what order polynomial allows us to minimize our residuals by nonlinear least squares regression (maximimizing
    likelihood)
    NOTE: This works in conjunction with the feedlinefitter function based on the order it is given, which can be specific
    or run through a large number of orders to see which fits best
    """
    ydata, modeldata = flattendata(feedlineobject, model)
    params = np.polyfit(modeldata, ydata, order, full=True)[0]
    return params


def flattendata(feedlineobject, model):
    """
    Create a function which flattens the data to conveniently work in frequency space, ensuring that the model data and
    measured data match each other, which is to say that at a given index in the 1D array, the ydata array shows the frequency
    that was measured at a given position, will at the same index, the modeldata array gives the design frequency at the same
    coordinate.
    """
    ydata = feedlineobject.normfreqs.flatten()
    modeldata = model.flatten()
    modeldata = modeldata[~np.isnan(ydata)]
    ydata = ydata[~np.isnan(ydata)]
    return ydata, modeldata


def feedlinefitter(feedlineobj, modelfeedline, order=None):
    """
    This returns the nonlinear least squares object from the Scipy optimize package, and we use two of the return
    values: .x is the array of coefficients for the best fit and .fun is the actual array of residuals. More often we
    use the .fun method so we can see how well we were able to fit our data
    """
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


def leastsquaremethod(feedlineobj, modelfeedline, order = None):
    """
    Given a feedline, design feedline, and whatever order you wish your least squares to be (None gives the best fit
    below order 30) and returns the measured frequencies, the frequencies fit by the least squares regression, the model
    frequencies, and the residuals
    """
    results, best_order = feedlinefitter(feedlineobj, modelfeedline,order)
    fitted_coeffs = np.poly1d(results.x) # Consider if this is necessary
    realdata, modeldata = flattendata(feedlineobj, design_feedline)
    fitteddata = fitted_coeffs(modeldata)
    residualvalues = results.fun
    return realdata, fitteddata, modeldata, residualvalues

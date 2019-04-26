import numpy as np

import makeNoiseSpectrum as mNS


def wienerFilter(template, noiseSpectrum,nTaps=50):
    """
    Default Filter. Calculate acausal Wiener Filter coefficients (roll off frequencies above 250 kHz)     
 
    INPUTS:
    noiseSpectrum - noise spectrum same length as template
    template - template of pulse shape
    
    OUTPUTS:
    wienerFilter - list of Wiener Filter coefficients
    """
    template /= np.max(np.abs(template)) #should be redundant

    #simulate anti-aliasing filter roll off
    templateFft=np.fft.rfft(template)
    spectrum=1/(1+(np.fft.rfftfreq(len(template),d=1e-6)/250000.0)**8.0)
    templateFft=templateFft*spectrum
    
    #set up so that filter works with a coorelation, not a convolution. 
    #Take the conjugate of templateFft for the other case
    wienerFilter= np.fft.irfft(np.conj(templateFft)/noiseSpectrum)[-nTaps:]
    filterNorm = np.convolve(template[:nTaps],wienerFilter, mode='same').max()
    wienerFilter /= filterNorm
    
    return -wienerFilter

def wienerFilter250(template, noiseSpectrum,nTaps=50):
    """
    Calculate acausal Wiener Filter coefficients (discard frequencies above 250 kHz)

    INPUTS:
    noiseSpectrum - noise spectrum same length as template
    template - template of pulse shape
    
    OUTPUTS:
    wienerFilter - list of Wiener Filter coefficients
    """
    noiseFreqs=np.fft.rfftfreq(len(template),d=1e-6)
    logic=(noiseFreqs>250000)
    template /= np.max(np.abs(template)) #should be redundant

    templateFft=np.fft.rfft(template)
    templateFft[logic]=0     

    #set up so that filter works with a coorelation, not a convolution. 
    #Take the conjugate of templateFft for the other case
    wienerFilter= np.fft.irfft(np.conj(templateFft)/noiseSpectrum)[-nTaps:]
    filterNorm = np.convolve(template[:nTaps], wienerFilter, mode='same').max()
    wienerFilter /= filterNorm

    return -wienerFilter

def wienerFilter250s(template, noiseSpectrum,nTaps=50):
    """
    Calculate acausal Wiener Filter coefficients (roll off frequencies above 250 kHz)     
 
    INPUTS:
    noiseSpectrum - noise spectrum same length as template
    template - template of pulse shape
    
    OUTPUTS:
    wienerFilter - list of Wiener Filter coefficients
    """
    template /= np.max(np.abs(template)) #should be redundant

    #simulate anti-aliasing filter roll off
    templateFft=np.fft.rfft(template)
    spectrum=1/(1+(np.fft.rfftfreq(len(template),d=1e-6)/250000.0)**8.0)
    templateFft=templateFft*spectrum
    
    #set up so that filter works with a coorelation, not a convolution. 
    #Take the conjugate of templateFft for the other case
    wienerFilter= np.fft.irfft(np.conj(templateFft)/noiseSpectrum)[-nTaps:]
    filterNorm = np.convolve(template[:nTaps], wienerFilter, mode='same').max()
    wienerFilter /= filterNorm
    
    return -wienerFilter

def wienerFilter200(template, noiseSpectrum,nTaps=50):
    """
    Calculate acausal Wiener Filter coefficients (discard frequencies above 200 kHz)

    INPUTS:
    noiseSpectrum - noise spectrum same length as template
    template - template of pulse shape
    
    OUTPUTS:
    wienerFilter - list of Wiener Filter coefficients
    """
    noiseFreqs=np.fft.rfftfreq(len(template),d=1e-6)
    logic=(noiseFreqs>200000)
    template /= np.max(np.abs(template)) #should be redundant

    templateFft=np.fft.rfft(template)
    templateFft[logic]=0     

    #set up so that filter works with a coorelation, not a convolution. 
    #Take the conjugate of templateFft for the other case
    wienerFilter= np.fft.irfft(np.conj(templateFft)/noiseSpectrum)[-nTaps:]
    filterNorm = np.convolve(template[:nTaps], wienerFilter, mode='same').max()
    wienerFilter /= filterNorm

    return -wienerFilter

def wienerFilter200s(template, noiseSpectrum,nTaps=50):
    """
    Calculate acausal Wiener Filter coefficients (roll off frequencies above 200 kHz)     
 
    INPUTS:
    noiseSpectrum - noise spectrum same length as template
    template - template of pulse shape
    
    OUTPUTS:
    wienerFilter - list of Wiener Filter coefficients
    """
    template /= np.max(np.abs(template)) #should be redundant

    #simulate anti-aliasing filter roll off
    templateFft=np.fft.rfft(template)
    spectrum=1/(1+(np.fft.rfftfreq(len(template),d=1e-6)/200000.0)**8.0)
    templateFft=templateFft*spectrum
    
    #set up so that filter works with a coorelation, not a convolution. 
    #Take the conjugate of templateFft for the other case
    wienerFilter= np.fft.irfft(np.conj(templateFft)/noiseSpectrum)[-nTaps:]
    filterNorm = np.convolve(template[:nTaps], wienerFilter, mode='same').max()
    wienerFilter /= filterNorm
    
    return -wienerFilter

def wienerFilter150(template, noiseSpectrum,nTaps=50):
    """
    Calculate acausal Wiener Filter coefficients (discard frequencies above 150 kHz)

    INPUTS:
    noiseSpectrum - noise spectrum same length as template
    template - template of pulse shape
    
    OUTPUTS:
    wienerFilter - list of Wiener Filter coefficients
    """
    noiseFreqs=np.fft.rfftfreq(len(template),d=1e-6)
    logic=(noiseFreqs>150000)
    template /= np.max(np.abs(template)) #should be redundant

    templateFft=np.fft.rfft(template)
    templateFft[logic]=0     

    #set up so that filter works with a coorelation, not a convolution. 
    #Take the conjugate of templateFft for the other case
    wienerFilter= np.fft.irfft(np.conj(templateFft)/noiseSpectrum)[-nTaps:]
    filterNorm = np.convolve(template[:nTaps], wienerFilter, mode='same').max()
    wienerFilter /= filterNorm

    return -wienerFilter

def wienerFilter150s(template, noiseSpectrum,nTaps=50):
    """
    Calculate acausal Wiener Filter coefficients (roll off frequencies above 150 kHz)     
 
    INPUTS:
    noiseSpectrum - noise spectrum same length as template
    template - template of pulse shape
    
    OUTPUTS:
    wienerFilter - list of Wiener Filter coefficients
    """
    template /= np.max(np.abs(template)) #should be redundant

    #simulate anti-aliasing filter roll off
    templateFft=np.fft.rfft(template)
    spectrum=1/(1+(np.fft.rfftfreq(len(template),d=1e-6)/150000.0)**8.0)
    templateFft=templateFft*spectrum
    
    #set up so that filter works with a coorelation, not a convolution. 
    #Take the conjugate of templateFft for the other case
    wienerFilter= np.fft.irfft(np.conj(templateFft)/noiseSpectrum)[-nTaps:]
    filterNorm = np.convolve(template[:nTaps], wienerFilter, mode='same').max()
    wienerFilter /= filterNorm
    
    return -wienerFilter

def wienerFilter100(template, noiseSpectrum,nTaps=50):
    """
    Calculate acausal Wiener Filter coefficients (discard frequencies above 100 kHz)

    INPUTS:
    noiseSpectrum - noise spectrum same length as template
    template - template of pulse shape
    
    OUTPUTS:
    wienerFilter - list of Wiener Filter coefficients
    """
    noiseFreqs=np.fft.rfftfreq(len(template),d=1e-6)
    logic=(noiseFreqs>100000)
    template /= np.max(np.abs(template)) #should be redundant

    templateFft=np.fft.rfft(template)
    templateFft[logic]=0     

    # set up so that filter works with a convolution, not a correlation.
    # Remove the conjugate of templateFft for the other case
    wienerFilter= np.fft.irfft(np.conj(templateFft)/noiseSpectrum)[-nTaps:]
    filterNorm = np.convolve(template[:nTaps], wienerFilter, mode='same').max()
    wienerFilter /= filterNorm

    return -wienerFilter

def wienerFilter100s(template, noiseSpectrum,nTaps=50):
    """
    Calculate acausal Wiener Filter coefficients (roll off frequencies above 100 kHz)     
 
    INPUTS:
    noiseSpectrum - noise spectrum same length as template
    template - template of pulse shape
    
    OUTPUTS:
    wienerFilter - list of Wiener Filter coefficients
    """
    template /= np.max(np.abs(template)) #should be redundant

    # simulate anti-aliasing filter roll off
    templateFft=np.fft.rfft(template)
    spectrum=1/(1+(np.fft.rfftfreq(len(template),d=1e-6)/100000.0)**8.0)
    templateFft=templateFft*spectrum
    
    # set up so that filter works with a convolution, not a correlation.
    # Remove the conjugate of templateFft for the other case
    wienerFilter= np.fft.irfft(np.conj(templateFft)/noiseSpectrum)[-nTaps:]
    filterNorm = np.convolve(template[:nTaps], wienerFilter, mode='same').max()
    wienerFilter /= filterNorm
    
    return -wienerFilter

def wienerFilter0(template, noiseSpectrum,nTaps=50):
    """
    Calculate acausal Wiener Filter coefficients. All frequencies are used.

    INPUTS:
    noiseSpectrum - noise spectrum same length as template
    template - template of pulse shape
    
    OUTPUTS:
    wienerFilter - list of Wiener Filter coefficients
    """

    template /= np.max(np.abs(template))  # should be redundant

    # set up so that filter works with a convolution, not a correlation.
    # Remove the conjugate of templateFft for the other case
    templateFft = np.fft.rfft(template)
    wienerFilter= np.fft.irfft(np.conj(templateFft)/noiseSpectrum)[-nTaps:]

    filterNorm = np.convolve(template[:nTaps], wienerFilter, mode='same').max()
    wienerFilter /= filterNorm
    return -wienerFilter

def matchedFilter(template, noiseSpectrum, nTaps=50):
    """
    Make a matched filter using a template and noise PSD. Rolls off template above 250kHz.
    (same as 250 roll off Wiener filter but calculated with the covariance matrix)

    INPUTS:
    template - array containing pulse template
    noiseSpectrum - noise PSD
    nTaps - number of filter coefficients
    
    OUTPUTS
    matchedFilt - matched filter that should be convolved with the data
                  to get the pulse heights 
    """
    # check normalized to 1
    template/=np.max(np.abs(template))
    
    # mimic antialiasing filter
    fft=np.fft.rfft(template)
    spectrum=1/(1+(np.fft.rfftfreq(len(template),d=1e-6)/250000.0)**8.0)
    fft=fft*spectrum
    template1=np.fft.irfft(fft)
    template1 /= np.max(np.abs(template1)) 

    noiseCov = mNS.covFromPsd(noiseSpectrum, nTaps)['covMatrix']
   
    template1 = template1[:nTaps]  #shorten template to length nTaps
    template = template[:nTaps]

    filterNorm= np.dot(template,np.linalg.solve(noiseCov,template1))
    matchedFilt=np.linalg.solve(noiseCov,template1)/filterNorm

    # flip so that the result works with a convolution
    matchedFilt = matchedFilt[::-1]

    return -matchedFilt

def superMatchedFilter(template, noiseSpectrum, nTaps=50):
    """
    Make a matched filter that is robust against pulse pileup using prescription from
    Alpert 2013 Rev. of Sci. Inst. 84. (Untested)

    INPUTS:
    template - array containing pulse template
    noiseSpectrum - noise PSD
    nTaps - number of filter coefficients
    
    OUTPUTS
    superMatchedFilt - super matched filter that should be convolved with 
                       the data to get the pulse heights 
    """
    # get the fall time for the end of the pulse
    # (only a good idea to use this formula if using a fitted template)
    fallTime=(template[-1]-template[-2])/np.log(template[-2]/template[-1])

    # determine pulse direction
    if np.min(template)>np.max(template):
        pos_neg=-1.
    else:
        pos_neg=1
    # check normalized to 1
    template/=np.abs(template[np.argmax(np.abs(template))])    
    # create covariance inverse matrix
    noiseCovInv = mNS.covFromPsd(noiseSpectrum, nTaps)['covMatrixInv']
    # shorten template to length nTaps
    template = template[:nTaps]  
    # create exponential to be orthogonal to
    exponential=pos_neg*np.exp(-np.arange(0,len(template))/fallTime)
    
    # create filter
    orthMat=np.array([template,exponential])
    orthMat=orthMat.T
    e1=np.array([1,0])
    norm=np.linalg.inv(np.dot(orthMat.T,np.dot(noiseCovInv,orthMat)))
    superMatchedFilter=np.dot(noiseCovInv,np.dot(orthMat,np.dot(norm,e1)))

    # Flip if you want the filter to work with a correlation and not convolution
    # superMatchedFilter=superMatchedFilter[::-1]
    return superMatchedFilter  

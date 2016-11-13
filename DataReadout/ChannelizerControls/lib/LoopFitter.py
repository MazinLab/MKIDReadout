"""
LoopFitter.py
Paul Szypryt 
July 8, 2016

Takes the frequency, I, and Q values around a single microwave resonator as inputs.  Adjusts the loop for the cable delay and fits the position of the center.
"""

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def RemoveDelay(IData, QData, frequencyData, tau):
    delayAngle = 2.0*np.pi*frequencyData*tau

    IRemoved = IData*np.cos(delayAngle) - QData*np.sin(delayAngle)
    QRemoved = IData*np.sin(delayAngle) + QData*np.cos(delayAngle)
    
    return IRemoved, QRemoved

def LoopFitter(IData, QData):

    # Using the circle parameterization A(x^2 + y^2) + Bx + Cy + D = 0
    # Let w = x^2 + y^2
    x = IData
    y = QData
    w = x**2 + y**2

    # Number of data points to include in the fit
    numPoints = w.size

    # Minimizing the function F(xc, yc, r) = Sum(i=1:n)[A*w_i^2 + B*x_i + C*y_i + D]^2
    # Constraints are B^2 + C^2 -4AD = 1
    # In matrix form, we have F = A' * M * A, with constraint A' * B * A = 1
    # M and B are devined below, A = [A, B, C, D]

    # Calculate moments
    # First row
    Mww = np.dot(w, w)
    Mxw = np.dot(x, w)
    Myw = np.dot(y, w)
    Mw = sum(w)
    # Second row
    # Mxw = dot(x, w)
    Mxx = np.dot(x, x)
    Mxy = np.dot(x, y)
    Mx = sum(x)
    # Third row
    # Myw = dot(y, w)
    # Mxy = dot(x, y)
    Myy = np.dot(y, y)
    My = sum(y)
    # Fourth row
    # Mw = sum(w)
    # Mx = sum(x)
    # My = sum(y)
    n = numPoints


    # Define M, and B matrices 
    MMat= np.matrix([[Mww, Mxw, Myw, Mw],
                    [Mxw, Mxx, Mxy, Mx],
                    [Myw, Mxy, Myy, My],
                    [Mw,  Mx,  My,  n ]])
    BMat = np.matrix([[0.0, 0.0, 0.0, -2.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [-2.0, 0.0, 0.0, 0.0]])
    
    # Find the generalized eigenvalue solution for matrix pair (M, B)
    # Solves equation M*A=eta*B*A
    generalEigenSolution = scipy.linalg.eig(MMat, BMat)

    # Separate out eigenvectors and eigenvalues from solution
    generalizedEigenvalues = np.real(generalEigenSolution[0])
    generalizedEigenvectors = generalEigenSolution[1]
    
    # We want the smallested position eigenvalue
    # Account for 0 being represented as extremely small negative value...
    eta = min(generalizedEigenvalues[np.where(generalizedEigenvalues >= -1/n)])
    
    # Take the eigenvector associated with smallest positive eta.
    # This represents the vector A, before normalization.
    AMat = generalizedEigenvectors[:,generalizedEigenvalues == eta]

    # Extract components of the vector, A.
    ARaw = AMat[0]
    BRaw = AMat[1]
    CRaw = AMat[2]
    DRaw = AMat[3]

    # Subject to the constraint B^2 + C^2 - 4AD = 1, need to renormalize.
    # Basically just defining the constant in front of the eigenvector.
    normFactor = 1/np.sqrt(BRaw**2+CRaw**2-4*ARaw*DRaw)
    A=ARaw*normFactor
    B=BRaw*normFactor
    C=CRaw*normFactor
    D=DRaw*normFactor

    # Reparameterize to go back to the physical parameters of interested
    xCen = -B/(2*A)
    yCen = -C/(2*A)
    radius = 1/(2*abs(A))

    return xCen, yCen, radius





#....................................................................................
# Test cases for above function

# Load an IQ sweep file, separate out I, Q and frequency
testFile = 'iqdata_jul_8_veruna_4.620303GHz.npz'
testData = np.load(testFile)
testI = testData['i']
testQ = testData['q']
testFrequencies = testData['freqs']*10.0**6


# Synthetic circle test case, no cable delay
syntheticTheta = np.linspace(0.0,2.0*np.pi,101)
syntheticError = np.random.randn(101)/10
syntheticICenter = np.random.randn()
syntheticQCenter = np.random.randn()

syntheticFrequencyData = np.linspace(4.0,4.2,101)

syntheticIData = syntheticICenter + (np.cos(syntheticTheta) + syntheticError)
syntheticQData = syntheticQCenter + (np.sin(syntheticTheta) + syntheticError)

#print 'Expected I Center: ' + str(syntheticICenter)
#print 'Expected Q Center: ' + str(syntheticQCenter)

#LoopFitter(syntheticIData, syntheticQData, syntheticFrequencyData)

testTau = 40.0*10.0**-9

print testFrequencies

IRemoved, QRemoved = RemoveDelay(testI, testQ, testFrequencies, testTau)


ICenter, QCenter, circleRadius = LoopFitter(testI, testQ)
ICenterRemoved, QCenterRemoved, circleRadiusRemoved = LoopFitter(IRemoved, QRemoved)

plt.plot(testI, testQ, 'b.', ICenter, QCenter, 'bo', IRemoved, QRemoved, 'r.', ICenterRemoved, QCenterRemoved, 'r.')
plt.show()

#plt.plot(testI, testQ, 'b.', ICenter, QCenter, 'ro')
#plt.show()



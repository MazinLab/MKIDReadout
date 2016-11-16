

from scipy.interpolate import griddata 
import numpy as np
import matplotlib.pyplot as plt



def interpolateImage(image, method='cubic'):
    """
    This function interpolates the dead pixels in the image
    
    INPUTS:
        image - 2D array. Nan values for dead pixels. 0 indicates that pixel is good and has 0 counts
        method - scipy griddata option. Can be nearest, linear, cubic.
        
    OUTPUTS:
        interpolatedImage - Same shape as input image. Dead pixels are interpolated over. 
    """
    shape = np.shape(image)
    goodPoints = np.where(np.isfinite(image))
    values = np.asarray(image)[goodPoints]
    interpPoints = (np.repeat(range(shape[0]),shape[1]), np.tile(range(shape[1]),shape[0]))

    interpValues = griddata(goodPoints, values, interpPoints, method)
    interpImage = np.reshape(interpValues, shape)
    
    return interpImage


if __name__=="__main__":
    image = [[1,1,1],[1,np.nan,5],[1,np.nan,1],[1,1,1]]
    interpImage = interpolateImage(image)
    
    plt.matshow(image)
    plt.matshow(interpImage)
    plt.show()

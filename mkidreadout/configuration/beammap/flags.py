"""
Author: Alex Walter
Date: June 5, 2018

This file contains the flags used in the beammap file
"""


beamMapFlags = {
                'good':0,           #No flagging
                'noDacTone':1,      #Pixel not read out
                'failed':2,         #Beammap failed to place pixel
                'yFailed':3,        #Beammap succeeded in x, failed in y
                'xFailed':4,        #Beammap succeeded in y, failed in x
                'double':5,         #Multiple locations found for pixel
                'wrongFeedline':6,  #Beammap placed pixel in wrong feedline
                'duplicatePixel':7,  #Beammap placed pixel on top of another one, and no neighbor could be found
                }

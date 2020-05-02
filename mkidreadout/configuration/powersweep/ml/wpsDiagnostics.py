import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import mkidreadout.configuration.sweepdata as sd
import mkidreadout.configuration.powersweep.ml.tools as mlt

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--sweep', default=None)
    parser.add_argument('--wpsmap', default=None)
    parser.add_argument('-f', '--freq', type=float, default=None)
    parser.add_argument('-a', '--atten', type=float, default=None)
    parser.add_argument('-w', '--window', type=int, default=100)
    parser.add_argument('-t', '--threshold', type=float, default=0)
    parser.add_argument('--sat', action='store_true')
    parser.add_argument('--up', action='store_true')
    parser.add_argument('--mag', action='store_true')
    parser.add_argument('--image', action='store_true')
    args = parser.parse_args()

    if args.wpsmap is not None:
        wpsdata = np.load(args.wpsmap)
        freqInd = np.argmin(np.abs(args.freq - wpsdata['freqs']))
        print freqInd
        if args.sat:
            plt.imshow(wpsdata['wpsmap'][:, freqInd-args.window/2:freqInd+args.window/2, 1], vmin=args.threshold)
        elif args.up:
            plt.imshow(wpsdata['wpsmap'][:, freqInd-args.window/2:freqInd+args.window/2, 2], vmin=args.threshold)
        elif args.mag: 
            plt.imshow(wpsdata['wpsmap'][:, freqInd-args.window/2:freqInd+args.window/2, -1], vmin=args.threshold)
        else:
            plt.imshow(wpsdata['wpsmap'][:, freqInd-args.window/2:freqInd+args.window/2, 0], vmin=args.threshold)
        plt.show()

    if args.model is not None:
        mlDict, sess, graph, x_input, y_output, keep_prob, is_training = mlt.get_ml_model(args.model)

    if args.sweep  is not None:
        freqSweep = sd.FreqSweep(args.sweep)

    if args.image:
        if args.sweep is None:
            raise Exception('Must specify model and freq sweep')
        fig0 = plt.figure()
        fig1 = plt.figure()
        ax0 = fig0.add_subplot(111)
        ax1 = fig1.add_subplot(111)

        if args.model is None:
            #image = mlt.makeWPSImageList(freqSweep, args.freq, args.atten, 30, 7, True, False,
            #        normalizeBeforeCenter=True)[0][0]
            image = mlt.makeWPSImage(freqSweep, args.freq, args.atten, 30, 7, True, False,
                    normalizeBeforeCenter=True)[0]

        else:
            image = mlt.makeWPSImageList(freqSweep, args.freq, args.atten, mlDict['freqWinSize'], 
                    1+mlDict['attenWinBelow']+mlDict['attenWinAbove'], mlDict['useIQV'], mlDict['useVectIQV'],
                    normalizeBeforeCenter=mlDict['normalizeBeforeCenter'])[0][0]

        ax0.plot(image[:, :, 0].T, image[:, :, 1].T)
        if image.shape[2] > 2:
            ax1.plot(image[:, :, 2].T)
        plt.show()

        

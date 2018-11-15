from mkidreadout.configuration.widesweep.widesweepfile import WideSweepFile as WSF

file='/Users/one/ucsb/Packages/MKIDReadout/tests/raw_digWS_FL5.txt'
f=WSF(file)
for s in f.tone_slices:
    plot(f.x[s],f.mag[s])

from skimage.feature import peak_local_max

def continuum_peak_anchor_spacing(sweep):
    # Only look for the highest points in a tone
    return 2*sweep.resonator_width/sweep.sampling

# def find_peaks(im, min_sep=10, minv=25, maxv=5000):
#     """find points to use for psf measurements"""
#     points = peak_local_max(im, min_distance=min_sep,
#                             threshold_abs=minv,
#                             threshold_rel=0, indices=False)
#     points[((im > maxv) & points)] = False
#
#     return zip(*np.where(points))


class FrequencySweep(object):
    def __init__(self):
        self.resonator_width = 25
        self.sampling = 1

sweep = FrequencySweep()

for s in f.tone_slices:
    plot(f.x[s],f.mag[s])

    points = peak_local_max(f.mag[s], min_distance=continuum_peak_anchor_spacing(sweep),
                            indices=False)
    plot(f.x[s][points], f.mag[s][points],'o')
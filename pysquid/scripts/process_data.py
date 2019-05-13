import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pysquid.util.datatools import smooth_streaks

def makefilenames(directory):
    paths = list(os.walk(directory))
    return [os.path.join(paths[0][0], p) for p in paths[0][2]]
   
def loadrawscan(filename):
    dat = loadmat(filename)
    scan = dat['scans'][0][0][0][:,:,2][::-1]
    voltage = dat['gate'][0][0]
    return [voltage, scan]

def load_analyzed_scans(filename):
    dat = loadmat(filename)
    voltage = dat['GateVoltage'][0][0]
    images = dat['im_data'][0][0]
    x, y = images[0], images[1]
    scan = images[2][::-1]
    jx = images[5][::-1]
    jy = images[8][::-1]
    return voltage, {'x': x, 'y': y, 'scan': scan, 
                     'jx': jx, 'jy': jy}

def loaddata(directory):
    filenames = makefilenames(directory)
    datalist = [load_analyzed_scans(f) for f in filenames]
    vorder = np.argsort([d[0] for d in datalist])
    return [datalist[v] for v in vorder]

if __name__ == '__main__':
    loc = os.path.dirname(os.path.realpath(__file__))
    dy, dx = [0.16, 0.73] #Pixel spacing in micrometers

    #Largest voltage (0.1) is thresholded so we throw away the last element!
    #datarec = loaddata(os.path.join(loc,
    #                                '../../../data/data_incl_fourier_reconstruction'))[:-1]
    datarec = loaddata("/home/colin/Dropbox/Katja_current_reconstruction_share/ImageData/data/data_incl_fourier_reconstruction")

    #Smooth streaks in data
    for dd in datarec:
        sm = smooth_streaks(dd[1]['scan'], dx, dy)
        dd[1]['scan'][:,:] = sm.copy()

    stds = np.array([d[1]['scan'].std() for d in datarec])
    maxstd = stds.max()
    scale = 3*maxstd
    for d in datarec: 
        d[1]['scan'] /= -1*scale
    np.savez(os.path.join(loc, "../gendata/flux_and_reconstructions.npz"), 
             data = datarec, scale = scale)


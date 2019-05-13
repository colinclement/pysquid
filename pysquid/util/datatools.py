import numpy as np
import scipy.ndimage as ndi
from pysquid.util.helpers import makeD2_operators, makeD_operators

def smooth_streaks(image, dx, dy, curve_cutoff = 4., 
                   eps = 1E-10, itnlim = 2000, binit = 5): 
    D2x, D2y = makeD2_operators(image.shape, dx, dy)
    D2xim = np.abs(D2x.dot(image.ravel()).reshape(image.shape))
    D2yim = np.abs(D2y.dot(image.ravel()).reshape(image.shape))
    md2x, md2y = D2xim.mean(), D2yim.mean()
    mask = ((D2xim > curve_cutoff*D2xim.std()) * 
            (D2yim > curve_cutoff*D2yim.std())) 
    sy, sx = (int(np.ceil(dx/dy)), 0) if dx > dy else (0, int(np.ceil(dy/dx)))
    smax = max(sy, sx)
    bmask = np.zeros((2*smax+1, 2*smax+1))
    bmask[smax-sy:smax+sy+1, smax] = 1.
    bmask[smax, smax-sx:smax+sx+1] = 1.

    expanded = ndi.binary_dilation(mask, structure = bmask, iterations = binit)
    flatexpand = expanded.ravel()
    Ly, Lx = image.shape
    coordinates = [(i/Lx, i%Lx) for i, e in enumerate(flatexpand) if e]
    smoothed = image.copy()
    norm = 2.*(dx**2+dy**2)
    for itn in range(itnlim):
        oldsmoothed = smoothed.copy()
        for (i, j) in coordinates:
            up = smoothed[i-1, j] if i > 0 else 0.
            down = smoothed[i+1, j] if i < Ly-1 else 0.
            left = smoothed[i, j-1] if j > 0 else 0.
            right = smoothed[i, j+1] if j < Lx-1 else 0.
            smoothed[i,j] = ((up + down)*dy**2 + (left + right)*dx**2)/norm
        diff = (smoothed - oldsmoothed)**2
        if diff.sum()/len(coordinates) < eps:
            break
    return smoothed
                 


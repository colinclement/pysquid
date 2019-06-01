import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import numexpr as nu
from scipy.ndimage import mean

try:
    from numba import jit, float64, complex128, int64, void

    hasnumba = True
except ImportError as er:
    print("Missing numba will effect performance")
    hasnumba = False
try:
    from itertools import imap
except ImportError as er:
    imap = map  # python 3
try:
    import png

    haspng = True
except ImportError as er:
    haspng = False




def changeUnits(params, dx=1.0, dy=1.0):
    """
    Rescale psf_params for Mixture of Gaussians
    input:
        params: (number of gaussians, 5)-shaped array
        dx, dy : floats to divide x, y by in params
    output:
        rescaled params
    """
    params = params.reshape(-1, 5)
    x, y = params[:, 1], params[:, 2]
    sx, sy = params[:, 3], params[:, 4]
    return np.vstack([params[:, 0], x / dx, y / dy, sx / dx, sy / dy]).T


def makeM(kernel, padding=[0, 0]):
    """
    Construct matrix M so that \phi = M.dot(g)
    Explicit convolution matrix. WARNING! You should only use this for very
    small image sizes, it requires ~N^2 memory where N is the number of pixels    
    input:
        kernel: (2*Ly_pad, 2*Lx_pad)-shaped array representing flux field
                due to a single g point source
                It should be centered so that the point source is in the
                middle of the image.

    output:
        M: float array of shape (N,N)
    """
    py, px = padding
    twoLy_pad, twoLx_pad = kernel.shape
    Ly_pad, Lx_pad = twoLy_pad // 2, twoLx_pad // 2
    Ly, Lx = Ly_pad - 2 * py, Lx_pad - 2 * px
    M = np.zeros((Lx * Ly, Lx_pad * Ly_pad))
    for j in range(Ly_pad):
        for i in range(Lx_pad):
            yc, xc = Ly_pad - j + py, Lx_pad - i + px  # upper left
            M[:, j * Lx_pad + i] = kernel[yc : yc + Ly, xc : xc + Lx].ravel()
    return M


def noise_estimate(image, w=2):
    """
    Estimate the noise in an image by looking at the variance of
    the difference between image and image smoothed by a boxcar
    filter of width w.
    input:
        image : (Ly, Lx)-shaped array
        w : (int) width in each direction of boxcar filter
    returns:
        sigma : (float) noise estimate of image
    """
    Ly, Lx = image.shape
    boxcar = np.zeros_like(image)
    boxcar[Ly // 2 - w : Ly // 2 + w, Lx // 2 - w : Lx // 2 + w] = 1.0
    boxcar = np.fft.fftshift(boxcar) / boxcar.sum()
    smoothed = np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(boxcar)).real
    return (image - smoothed).std()


def autoCorrByDistance(image, n_bins):
    Ly, Lx = image.shape
    x = np.arange(Lx) - Lx // 2
    y = np.arange(Ly) - Ly // 2
    xg, yg = np.meshgrid(x, y)
    r = np.roll(np.roll(np.hypot(xg, yg), Lx // 2, 1), Ly // 2, 0)
    bins = np.linspace(0, r.max(), num=n_bins + 1, endpoint=True)
    binlabels = np.digitize(r.ravel(), bins, right=True)
    image_k = np.fft.fftn(image)
    acorr = np.fft.ifftn(image_k * image_k.conj()).real / (Lx * Ly)
    a = mean(acorr, labels=binlabels.reshape(Ly, Lx), index=bins)
    return bins, a


def loadpng(filename):
    """
    input : filename of a .png image
    returns : RGB or RGBA array
    """
    data = png.Reader(filename).read()
    width, height = data[0], data[1]
    imarr = np.vstack(imap(np.uint16, data[2]))
    return imarr.reshape(height, width, -1)


if not haspng:

    def loadpng(filename):
        print("Module png not found; pip install pypng")


def plotrgba(rgba):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = ["Red", "Green", "Blue"]
    for ax, lab, chan in zip(axes, labels, range(3)):
        ax.matshow(rgba[:, :, chan], cmap="Greys")
        # ax.colorbar()
        ax.set_title(lab)


def matshow(mat, aspect=0.2, figsize=(10, 10)):
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    cax = axes.matshow(mat, aspect=aspect)
    plt.colorbar(cax, fraction=0.037)
    plt.show()
    return fig, axes


################ Optimized helper functions ##################

if hasnumba:

    @jit(void(complex128[:, :], complex128[:, :], complex128[:, :]), nopython=True)
    def _mult(store, a, b):
        Ly, Lx = store.shape
        for y in range(Ly):
            for x in range(Lx):
                store[y, x] = a[y, x] * b[y, x]


else:

    def _mult(store, a, b):
        store[:, :] = a * b

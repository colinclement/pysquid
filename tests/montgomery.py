import os
import string
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import ImageGrid

from pysquid.rnet import ResistorNetworkModel
from pysquid.kernels.magpsf import GaussianKernel
from pysquid.util.linear_operators import curl, makeD2
from pysquid.util.datatools import estimate_noise, match_edge

from tester import Tester
from lithographic_mask import makedata, makedata2

mpl.rcParams['axes.titlesize'] = 15
mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['font.family'] = 'serif'

def j_density(g):
    return np.hypot(*curl(g))

def schematic(mask, ext_mask, window, sl):
    fig, axe = plt.subplots()
    ext_model = 1 * mask
    ext_model[sl] = ext_mask

    extent = [0, mask.shape[1], mask.shape[0], 0]
    img = np.ones((mask.shape[0], mask.shape[1], 3))
    removed = np.array([242, 56, 127])/255
    sample = np.array([138, 138, 138])/255
    inside = np.array([4, 204, 0])/255
    img[(mask == 1) * (ext_model == 1),:] = removed
    img[(mask == 1) * (ext_model != 1),:] = sample
    img[(mask != 1) * (ext_model == 1),:] = inside

    axe.imshow(img, extent=extent, interpolation='gaussian')
    axe.axis('off')
    lines = [Line2D([0], [0], color=removed, lw=10.), 
             Line2D([0], [0], color=sample, lw=10.),
             Line2D([0], [0], color=inside, lw=10.)]
    axe.legend(lines, 
               ['Subtracted by $\mathbf{g}_\mathrm{ext}$', 
                'Sample region of interest', 
                'Bridge connecting model leads'],
               bbox_to_anchor=(-.03, .05),
               loc='upper left')

    axins = axe.inset_axes([0.2, 0.3, .6, .6])

    insetimg = np.zeros_like(img)
    insetimg[sl][window] = img[sl][window]
    axins.imshow(insetimg, extent=extent, interpolation='bilinear')
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    axins.set_title("Field of view")
    x0 = sl[1].start + window[1].start
    x1 = sl[1].start + window[1].stop
    y0 = sl[0].start + window[0].start
    y1 = sl[0].start + window[0].stop
    axins.set_xlim([x0, x1])
    axins.set_ylim([y1, y0])

    axe.indicate_inset_zoom(axins)
    return fig, axe
    

def compare(true_g, results, mirror_results,
            labels=['(b) Gaussian\nwith mirror',
                    '(c) TV\nwith mirror',
                    '(d) TV with\nexternal model']):
    letters = string.ascii_lowercase
    fig = plt.figure(figsize=(10., 4.5))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 4), axes_pad=0.01, 
                     share_all=True, cbar_location="right", 
                     cbar_mode="each",)
    for i in range(len(grid)):
        if not i%4 == 3:
            grid.cbar_axes[i].axis('off')
        grid[i].axis('off')
    true_j = j_density(true_g)
    lim = true_j.max()
    gsols = [mirror_results['gaussian']['gsol'],
             mirror_results['TV prior']['gsol'],
             results['TV prior']['gsol']]
    for g in gsols:
        j = j_density(g)
        lim = max(lim, j.max())
    kwargs = {'cmap': 'gray_r', 'vmin': 0, 'vmax': lim}

    grid[0].matshow(true_j, **kwargs)
    grid[0].set_title("(a) Ground truth\nsample $|\mathbf{j}|$", pad=0)
    for i, (lab, g) in enumerate(zip(labels, gsols)):
        im = grid[i+1].matshow(j_density(g), **kwargs)
        grid[i+1].set_title(lab, pad=0)
    grid.cbar_axes[i+1].colorbar(im)

    for g in gsols:
        j = j_density(g) - true_j
        lim = max(lim, max(abs(j.min()), j.max()))
    kwargs = {'cmap': 'RdBu_r', 'vmin': -lim, 'vmax': lim}
    for i, g in enumerate(gsols):
        im = grid[i+5].matshow(j_density(g) - true_j, **kwargs)
    grid[5].set_title("Reconstruction\nerror", pad=0)
    grid.cbar_axes[i+5].colorbar(im)

    grid.set_axes_pad((0.01, 0.5))
    return fig, grid

DIR = os.path.dirname(os.path.realpath(__file__))

GDAT = os.path.join(DIR, 'data', 'g-field-litho-2.npz')
if not os.path.exists(GDAT):
    #makedata()
    makedata2()

MASK = os.path.join(DIR, 'data', 'litho-mask.png') 
mask = 1 * (plt.imread(MASK)[:,:,3] > 0)[:1000]

dat = np.load(GDAT)

g = dat['g']
sl = tuple(dat['model_slice'])
window = tuple(dat['window'])
g_ext = dat['g_ext']

sigma = 0.05

params = np.array([4.0, 1e-3, 1e-3])  # very small gaussian so no PSF
kernel = GaussianKernel(g[window].shape, params)
mirror_kernel = GaussianKernel(g[window].shape, params, mirror=True)

# edges = False turns off the top current loop
ext_kernel = GaussianKernel(mask[sl].shape, params, edges=False)

g_scale = ext_kernel.applyM(g)[window].ptp()
true_g = g / g_scale
g_ext = g_ext / g_scale

# True flux data
phi = ext_kernel.applyM(true_g)[window]

ext_flux = ext_kernel.applyM(g_ext)

tester = Tester(true_g[window], phi, kernel, sigma, g_ext=g_ext[window].ravel(), 
                phi_ext=ext_flux[window].ravel())
mirror_tester = Tester(true_g[window], phi, mirror_kernel, sigma)
control_tester = Tester(true_g[window], phi, kernel, sigma)

admm_kwargs = {'iprint': 1, 'eps_rel': 1e-8, 'eps_abs': 1e-8, 'itnlim': 300,
               'rho': 1e-4}
L_factor = 0.8
TV_factor = 0.9

protocol = []
protocol.append(
    dict(label="gaussian", decon="LinearDeconvolver",
         sigma=L_factor * sigma)
)
protocol.append(
    dict(label="TV prior", decon="TVDeconvolver", sigma=TV_factor * sigma,
         deconv_kwargs=admm_kwargs)
)

#results = tester.test_protocols(protocol)
#mirror_results = mirror_tester.test_protocols(protocol)
#control_results = control_tester.test_protocols(protocol)

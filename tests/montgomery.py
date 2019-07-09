import os
import numpy as np
import matplotlib.pyplot as plt

from pysquid.rnet import ResistorNetworkModel
from pysquid.kernels.magpsf import GaussianKernel
from pysquid.util.linear_operators import curl, makeD2
from pysquid.util.datatools import estimate_noise, match_edge

from tester import Tester
from lithographic_mask import makedata, makedata2

DIR = os.path.dirname(os.path.realpath(__file__))

GDAT = os.path.join(DIR, 'data', 'g-field-litho.npz')
if not os.path.exists(GDAT):
    makedata()
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

admm_kwargs = {'iprint': 1, 'eps_rel': 1e-8, 'eps_abs': 1e-8, 'itnlim': 200,
               'rho': 1e-2}
L_factor = 0.7
TV_factor = 0.7

protocol = []
protocol.append(
    dict(label="gaussian", decon="LinearDeconvolver",
         sigma=L_factor * sigma)
)
protocol.append(
    dict(label="TV prior", decon="TVDeconvolver", sigma=TV_factor * sigma,
         deconv_kwargs=admm_kwargs)
)

results = tester.test_protocols(protocol)
mirror_results = mirror_tester.test_protocols(protocol)
control_results = control_tester.test_protocols(protocol)

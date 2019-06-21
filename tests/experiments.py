import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from scipy.ndimage import binary_erosion
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from pysquid.kernels.magpsf import GaussianKernel
from pysquid.util.linear_operators import curl
from pysquid.util.datatools import estimate_noise 

from annular_currents import annular_gfield
from tester import Tester

mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['font.family'] = 'serif'

def j_density(g):
    return np.hypot(*curl(g))

def get_j_density_range(true_g, results):
    j = j_density(true_g)
    vmin, vmax = j.min(), j.max()
    for v in results.values():
        j = j_density(v['gsol'])
        vmin = min(vmin, j.min())
        vmax = max(vmax, j.max())
    return dict(vmin=vmin, vmax=vmax)

def addline(axe, xy, xytext):
    axe.annotate('', xy=xy, xycoords='data', xytext=xytext, 
                 textcoords='figure fraction',
                 arrowprops=dict(arrowstyle='-'))

def plot_regularization(results, tester, protocol):
    sigma = tester.sigma
    vkwargs = dict(vmin=-3*sigma, vmax=3*sigma)

    res = [v['residual'] for k, v in results.items()]
    err = np.array([v['residual'].std() for k, v in results.items()])
    best = np.argmin(np.abs(err - sigma))

    gamma = np.array([p['sigma']/sigma for p in protocol])
    fig, axe = plt.subplots(figsize=(6.4, 4.))
    axe.plot(gamma, err)
    axe.axhline(sigma, c='k', label=r'True $\sigma$', lw=0.8)
    axe.set_ylabel(r'$\mathrm{std}~||Mg - \phi||^2$')
    axe.set_xlabel(r'Regularization strength $\gamma$')
    axe.legend(loc='upper left')

    im = OffsetImage(res[0], zoom=0.8)
    ab = AnnotationBbox(im, xy=[gamma[0], err[0]],
                        xybox=(.31, .33), boxcoords='figure fraction',
                        xycoords='data',
                        pad=0.,
                        arrowprops=dict(arrowstyle="->", lw=1.))
    axe.add_artist(ab)

    im = OffsetImage(res[best], zoom=0.8)
    ab = AnnotationBbox(im, xy=[gamma[best], err[best]],
                        xybox=(.55, .42), boxcoords='figure fraction',
                        xycoords='data',
                        pad=0.,
                        arrowprops=dict(arrowstyle="->", lw=1.))
    axe.add_artist(ab)

    im = OffsetImage(res[-1], zoom=0.8)
    ab = AnnotationBbox(im, xy=[gamma[-1], err[-1]],
                        xybox=(.8, .55), boxcoords='figure fraction',
                        xycoords='data',
                        pad=0.,
                        arrowprops=dict(arrowstyle="->", lw=1.))
    axe.add_artist(ab)

    fig.subplots_adjust(bottom=0.15)

    return fig, axe 


def compare_truth(uni_result, uni_tester, para_result, para_tester):
    fig = plt.figure(1, (8., 8.))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.4, 
                     share_all=True, cbar_location="right", 
                     cbar_mode="single",)
    vlim = get_j_density_range(uni_tester.g, uni_result)
    grid[0].matshow(j_density(uni_tester.g), cmap='gray_r', **vlim)
    grid[0].set_title("True uniform current ring", pad=0)

    for label, res in uni_result.items():
        grid[1].matshow(j_density(res['gsol']), cmap='gray_r', **vlim)
        grid[1].set_title(label, pad=0)

    vlim = get_j_density_range(para_tester.g, para_result)
    grid[2].matshow(j_density(para_tester.g), cmap='gray_r', **vlim)
    grid[2].set_title("True parabolic current ring", pad=0)

    for label, res in para_result.items():
        im = grid[3].matshow(j_density(res['gsol']), cmap='gray_r', **vlim)
        grid[3].set_title(label, pad=0)

    for i in range(len(grid)):
        grid[i].axis('off')
    grid.cbar_axes[0].colorbar(im)
    return fig, grid

def diagnostic(results, tester, protocols):
    s = tester.sigma
    g = tester.g
    fig, axes = plt.subplots(2, len(results))
    for ax, pro, (k, v) in zip(axes[0], protocols, results.items()):
        ax.matshow(results[k]['residual'])
        ax.set_title(r"$\gamma\sigma$={}, ".format(pro['sigma']) + k)
        ax.axis('off')
    for ax, (k, v) in zip(axes[1], results.items()):
        res = results[k]['residual'].ravel()
        r, b, _ = ax.hist(res, bins=50, density=True)
        x = 0.5 * (b[1:] + b[:-1])
        ax.plot(x, np.exp(-0.5 * (x / s) ** 2)/np.sqrt(2 * np.pi * s ** 2))
        ax.set_title("Standard Deviation = {:.3f}".format(res.std()))
        ax.set_yscale('log')
    fig.suptitle('Noise sigma = {}'.format(s))

    fig2, axes2 = plt.subplots()
    axes2.plot(curl(g)[1][len(g)//2], label='Ground truth')
    for k, v in results.items():
        axes2.plot(curl(results[k]['gsol'])[1][len(g)//2], label=k)
    plt.legend()
    return (fig, axes), (fig2, axes2)

def show_current_density(results, tester, cmap='gray_r'):
    fig, axes = plt.subplots(1, len(results)+1)
    fig.suptitle(
        'Reconstructed Current Density, sigma = {}'.format(tester.sigma)
    )
    vlim = get_range(j_density(tester.g), results)
    axes[0].matshow(j_density(tester.g), cmap=cmap, **vlim)
    axes[0].set_title('Ground truth')
    axes[0].axis('off')
    for ax, (k, v) in zip(axes[1:], results.items()):
        ax.matshow(j_density(v['gsol']), cmap=cmap, **vlim)
        ax.set_title(k)
        ax.axis('off')
    return fig, axes

L = 100
inner_rad = 10
width = 20
sigma = 0.05

g_uniform = annular_gfield(L, inner_rad, width)
g_parabolic = annular_gfield(L, inner_rad, width, profile="parabolic")

mask = np.zeros_like(g_uniform)
mask[g_uniform == 0.] = 1.
mask[g_uniform == g_uniform.max()] = 1.
mask = 1 * binary_erosion(mask == 1., border_value=True)

params = np.array([4.0, 1e-3, 1e-3])  # very small gaussian basically no PSF
kernel = GaussianKernel((L, L), params)

# rescale g-field so flux is of order 1
g_uniform /= kernel.applyM(g_uniform).ptp()
g_parabolic /= kernel.applyM(g_parabolic).ptp()

admm_kwargs = {'iprint': 0, 'eps_rel': 1e-7, 'eps_abs': 1e-5, 'itnlim': 100,
               'rho': 1e-1}
TV_factor = 2.
L_factor = 2.7

protocols = []
protocols.append(
    dict(label="annulus TV prior", decon="TVDeconvolver", sigma=TV_factor * sigma,
         deconv_kwargs=admm_kwargs)
)
protocols.append(
    dict(label="annulus TV and finite support prior",
         decon="TVDeconvolver", sigma=TV_factor * sigma, support_mask=mask,
         deconv_kwargs=admm_kwargs)
)
protocols.append(
    dict(label="annulus gaussian prior", decon="LinearDeconvolver",
         sigma=L_factor * sigma)
)
protocols.append(
    dict(label="annulus gaussian and finite support prior",
         decon="LinearDeconvolver", 
         sigma=L_factor * sigma, support_mask=mask)
)


uniform_tester = Tester(g_uniform, kernel, sigma)
parabolic_tester = Tester(g_parabolic, kernel, sigma)

sigma = estimate_noise(uniform_tester.phi.reshape(kernel._shape))
reg_protocol = []
factors = np.linspace(0.01, 10, 50)
for fac in factors:
    reg_protocol.append(
        dict(label='factor = {}'.format(fac), decon='LinearDeconvolver',
             sigma=fac * sigma)
    )

#print("Performing uniform annulus tests")
#uniform_results = uniform_tester.test_protocols(protocols)
#
#print("Performing parabolic annulus tests")
#parabolic_results = parabolic_tester.test_protocols(protocols)

#fig, grid = compare_truth(
#    {'Gaussian prior reconstruction': uniform_results['annulus gaussian prior']}
#    uniform_tester, 
#    {'Gaussian prior reconstruction': parabolic_results['annulus gaussian prior']}, 
#    parabolic_tester
#)

#fig, grid = compare_truth(
#    {'TV prior reconstruction': uniform_results['annulus TV prior']}
#    uniform_tester, 
#    {'TV prior reconstruction': parabolic_results['annulus TV prior']}, 
#    parabolic_tester
#)

#print("Performing regularization test")
#reg_results = uniform_tester.test_protocols(reg_protocol)

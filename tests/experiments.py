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
mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['font.family'] = 'serif'

def absfft(img):
    imgk = np.fft.fftn(img)
    imgk[0,0] = 0.
    return np.fft.fftshift(np.abs(imgk))

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

def plot_regularization(results, tester, protocol, opt_index=5, yshift_b=0.,
                        yshift_t=0.):
    sigma = tester.sigma
    vkwargs = dict(vmin=-3*sigma, vmax=3*sigma)

    res = [v['residual'] for k, v in results.items()]
    err = np.array([v['residual'].std() for k, v in results.items()])
    best = np.argmin(np.abs(err - sigma))

    lamb = np.array([p['sigma']/sigma for p in protocol])
    fig, axe = plt.subplots(figsize=(6.4, 4.))
    axe.plot(lamb, err)
    axe.axhline(sigma, c='k', label=r'True $\sigma$', lw=0.8)
    axe.set_ylabel(r'$\mathrm{std}~||Mg_\lambda - \phi||^2$')
    axe.set_xlabel(r'Regularization strength $\lambda$')
    axe.legend(loc='lower right')
   
    axe.plot(lamb[0], err[0], '.', c='k')
    im = OffsetImage(res[0], zoom=0.8)
    ab = AnnotationBbox(im, xy=[lamb[0], err[0]],
                        xybox=(.31, .33-yshift_b), boxcoords='figure fraction',
                        xycoords='data',
                        pad=0.,
                        arrowprops=dict(arrowstyle="->", lw=1.))
    axe.add_artist(ab)
    im = OffsetImage(absfft(res[0]), zoom=0.8, cmap='gray_r')
    ab = AnnotationBbox(im, xy=[lamb[0], err[0]],
                        xybox=(0.35, 0.4-yshift_b), boxcoords='figure fraction',
                        arrowprops=dict(alpha=0), pad=0)
    axe.add_artist(ab)

    axe.plot(lamb[best], err[best], '.', c='k')
    im = OffsetImage(res[best], zoom=0.8)
    ab = AnnotationBbox(im, xy=[lamb[best], err[best]],
                        xybox=(.535, .38-yshift_b), boxcoords='figure fraction',
                        xycoords='data',
                        pad=0.,
                        arrowprops=dict(arrowstyle="->", lw=1.))
    axe.add_artist(ab)
    im = OffsetImage(absfft(res[best]), zoom=0.8, cmap='gray_r')
    ab = AnnotationBbox(im, xy=[lamb[0], err[0]],
                        xybox=(0.575, 0.45-yshift_b), boxcoords='figure fraction',
                        arrowprops=dict(alpha=0), pad=0)
    axe.add_artist(ab)

    axe.plot(lamb[-1], err[-1], '.', c='k')
    im = OffsetImage(res[-1], zoom=0.8)
    ab = AnnotationBbox(im, xy=[lamb[-1], err[-1]],
                        xybox=(.7675, .43-yshift_b), boxcoords='figure fraction',
                        xycoords='data',
                        pad=0.,
                        arrowprops=dict(arrowstyle="->", lw=1.))
    axe.add_artist(ab)
    im = OffsetImage(absfft(res[-1]), zoom=0.8, cmap='gray_r')
    ab = AnnotationBbox(im, xy=[lamb[0], err[0]],
                        xybox=(0.8075, 0.5-yshift_b), boxcoords='figure fraction',
                        arrowprops=dict(alpha=0), pad=0)
    axe.add_artist(ab)

    axe.plot(lamb[opt_index], err[opt_index], '.', c='k')
    im = OffsetImage(res[opt_index], zoom=0.8)
    ab = AnnotationBbox(im, xy=[lamb[opt_index], err[opt_index]],
                        xybox=(.245, .75-yshift_t), boxcoords='figure fraction',
                        xycoords='data',
                        pad=0.,
                        arrowprops=dict(alpha=0))
    axe.add_artist(ab)
    im = OffsetImage(absfft(res[opt_index]), zoom=0.8, cmap='gray_r')
    ab = AnnotationBbox(im, xy=[lamb[opt_index], err[opt_index]],
                        xybox=(0.295, 0.83-yshift_t), boxcoords='figure fraction',
                        arrowprops=dict(arrowstyle='->', lw=1.), pad=0)
    axe.add_artist(ab)

    axe.set_yticks([0.04, 0.045, 0.05, 0.055])
    fig.subplots_adjust(bottom=0.15)
    fig.subplots_adjust(left=0.15)

    return fig, axe 

def compare_truth(uni_result, uni_tester, para_result, para_tester):
    fig = plt.figure(1, (17.1, 8.25))
    L = len(uni_result)
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 1 + L), axes_pad=0.4, 
                     share_all=True, cbar_location="right", 
                     cbar_mode="single",)
    vlim = get_j_density_range(uni_tester.g, uni_result)
    grid[0].matshow(j_density(uni_tester.g), cmap='gray_r', **vlim)
    grid[0].set_title("(a) Uniform ground truth", pad=0)

    for i, (label, res) in enumerate(uni_result.items()):
        grid[i+1].matshow(j_density(res['gsol']), cmap='gray_r', **vlim)
        grid[i+1].set_title(label, pad=0)

    vlim = get_j_density_range(para_tester.g, para_result)
    grid[L+1].matshow(j_density(para_tester.g), cmap='gray_r', **vlim)
    grid[L+1].set_title("(f) Parabolic ground truth", pad=0)

    for i, (label, res) in enumerate(para_result.items()):
        im = grid[L+2+i].matshow(j_density(res['gsol']), cmap='gray_r', **vlim)
        grid[L+2+i].set_title(label, pad=0)

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
        ax.set_title(r"$\lambda\sigma$={}, ".format(pro['sigma']) + k)
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
    dict(label="TV prior", decon="TVDeconvolver", sigma=TV_factor * sigma,
         deconv_kwargs=admm_kwargs)
)
protocols.append(
    dict(label="TV and finite support prior",
         decon="TVDeconvolver", sigma=TV_factor * sigma, support_mask=mask,
         deconv_kwargs=admm_kwargs)
)
protocols.append(
    dict(label="gaussian prior", decon="LinearDeconvolver",
         sigma=L_factor * sigma)
)
protocols.append(
    dict(label="gaussian and finite support prior",
         decon="LinearDeconvolver", 
         sigma=L_factor * sigma, support_mask=mask)
)

uniform_tester = Tester(g_uniform, kernel, sigma)
parabolic_tester = Tester(g_parabolic, kernel, sigma)

sigma = estimate_noise(uniform_tester.phi.reshape(kernel._shape))
reg_protocol = []
factors = np.linspace(0.0001, 10, 50)
for fac in factors:
    reg_protocol.append(
        dict(label='factor = {}'.format(fac), decon='LinearDeconvolver',
             sigma=fac * sigma, support_mask=mask)
    )

#print("Performing uniform annulus tests")
#uniform_results = uniform_tester.test_protocols(protocols)
#
#print("Performing parabolic annulus tests")
#parabolic_results = parabolic_tester.test_protocols(protocols)
#
#fig, grid = compare_truth(
#    {
#        '(b) Gaussian prior': uniform_results['gaussian prior'],
#        '(c) TV prior': uniform_results['TV prior'],
#        '(d) Gaussian and FS prior': 
#          uniform_results['gaussian and finite support prior'],
#        '(e) TV and FS prior': uniform_results['TV and finite support prior'],
#    },
#    uniform_tester, 
#    {
#        '(g) Gaussian prior': parabolic_results['gaussian prior'],
#        '(h) TV prior': parabolic_results['TV prior'],
#        '(i) Gaussian and FS prior': 
#          parabolic_results['gaussian and finite support prior'],
#        '(j) TV and FS prior': parabolic_results['TV and finite support prior'],
#    },
#    parabolic_tester
#)

#print("Performing parabolic regularization test")
#uniform_reg_results = parabolic_tester.test_protocols(reg_protocol)

#print("Performing uniform regularization test")
#parabolic_reg_results = uniform_tester.test_protocols(reg_protocol)
#fig, axe = plot_regularization(parabolic_reg_results, parabolic_tester,
#                               reg_protocol, 9, yshift_b=.03)

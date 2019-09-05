import string
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.gridspec as gridspec
from scipy.ndimage import binary_erosion
from mpl_toolkits.axes_grid1 import ImageGrid

from pysquid.kernels.magpsf import GaussianKernel
from pysquid.util.linear_operators import curl, makeD2
from pysquid.util.datatools import estimate_noise 

from annular_currents import annular_gfield
from tester import Tester

mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['font.family'] = 'serif'

def absfft(img):
    imgk = np.fft.fftn(img)
    imgk[0,0] = 0.
    return np.fft.fftshift(np.abs(imgk))

def j_density(g):
    return np.hypot(*curl(g))

def get_j_density_range(true_g, results, truth=None):
    jtruth = truth if truth is not None else np.zeros_like(true_g)
    j = j_density(true_g) - jtruth
    vmin, vmax = j.min(), j.max()
    for v in results.values():
        j = j_density(v['gsol']) - jtruth
        vmin = min(vmin, j.min())
        vmax = max(vmax, j.max())
    if truth is not None:
        lim = max(abs(vmin), abs(vmax))
        return dict(vmin=-lim, vmax=lim)
    else:
        return dict(vmin=vmin, vmax=vmax)

def addline(axe, xy, xytext):
    axe.annotate('', xy=xy, xycoords='data', xytext=xytext, 
                 textcoords='figure fraction',
                 arrowprops=dict(arrowstyle='-'))

def plot_regularization(results, tester, protocol, opt_index=5):
    sigma = tester.sigma
    vkwargs = dict(vmin=-3*sigma, vmax=3*sigma)

    res = [v['residual'] for k, v in results.items()]
    vlim = {'vmin': np.min(res), 'vmax': np.max(res)}
    err = np.array([v['residual'].std() for k, v in results.items()])
    best = np.argmin(np.abs(err - sigma))
    lamb = np.array([p['sigma']/sigma for p in protocol])

    L = res[0].shape[0]
    gs = gridspec.GridSpec(3, 4, height_ratios=[2, 1., 1.])

    axe = plt.subplot(gs[0,:])
    axe.plot(lamb, err)
    axe.axhline(sigma, c='k', label=r'True $\sigma$', lw=0.8, dashes=[4,3])
    axe.set_ylabel(r'$\mathrm{std}(||Mg_\lambda - \phi||^2)$', fontsize=12)
    axe.set_xlabel(r'Regularization strength $\lambda$', fontsize=12)
    axe.legend(loc='lower right')

    axe.plot(lamb[0], err[0], marker='v', c='k', markersize=10.)
    axe1 = plt.subplot(gs[1,0])
    axe1.matshow(absfft(res[0]), cmap='gray_r')
    axe1.plot(L/6, L/6, marker='v', c='k', markersize=10.)
    axe1.axis('off')
    axe1.text(-L/1.8, L/2, "Fourier\nresiduals",
              fontdict={'rotation': 'vertical', 'verticalalignment': 'center'})
    axe2 = plt.subplot(gs[2,0])
    axe2.matshow(res[0], cmap='RdBu', **vlim)
    axe2.plot(L/6, L/6, marker='v', c='k', markersize=10.)
    axe2.text(-L/3.9, L/2, "Residuals",
              fontdict={'rotation': 'vertical', 'verticalalignment': 'center'})
    axe2.axis('off')
    axe2.set_ylabel("Residuals")

    axe.plot(lamb[opt_index], err[opt_index], marker='*', c='b', markersize=13.)
    axe3 = plt.subplot(gs[1,1])
    axe3.matshow(absfft(res[opt_index]), cmap='gray_r')
    axe3.plot(L/6, L/6, marker='*', c='b', markersize=13.)
    axe3.axis('off')
    axe4 = plt.subplot(gs[2,1])
    axe4.matshow(res[opt_index], cmap='RdBu', **vlim)
    axe4.plot(L/6, L/6, marker='*', c='b', markersize=13.)
    axe4.axis('off')
    
    axe.plot(lamb[best], err[best], marker='X', c='k', markersize=10.)
    axe5 = plt.subplot(gs[1,2])
    axe5.matshow(absfft(res[best]), cmap='gray_r')
    axe5.plot(L/6, L/6, marker='X', c='k', markersize=10.)
    axe5.axis('off')
    axe6 = plt.subplot(gs[2,2])
    axe6.matshow(res[best], cmap='RdBu', **vlim)
    axe6.plot(L/6, L/6, marker='X', c='k', markersize=10.)
    axe6.axis('off')

    axe.plot(lamb[-1], err[-1], marker='H', c='k', markersize=10.)
    axe7 = plt.subplot(gs[1,3])
    axe7.matshow(absfft(res[-1]), cmap='gray_r')
    axe7.plot(L/6, L/6, marker='H', c='k', markersize=10.)
    axe7.axis('off')
    axe8 = plt.subplot(gs[2,3])
    axe8.matshow(res[-1], cmap='RdBu', **vlim)
    axe8.plot(L/6, L/6, marker='H', c='k', markersize=10.)
    axe8.axis('off')

    fig = plt.gcf()
    fig.set_size_inches([5.3, 4.7])

    p = axe.get_position()
    axe.set_position([p.x0+0.065, p.y0+.08, p.width-.1, p.height-.03])
    
    return plt.gcf(), axe

def compare_truth(uni_result, uni_tester, para_result, para_tester):
    letters = string.ascii_lowercase
    fig = plt.figure(figsize=(10., 11.))
    L = len(uni_result)
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 1 + L), axes_pad=0.1, 
                     share_all=True, cbar_location="right", 
                     cbar_mode="each",)
    vlim = get_j_density_range(uni_tester.g, uni_result)
    uniform_truth = j_density(uni_tester.g)
    grid[0].matshow(uniform_truth, cmap='gray_r', **vlim)
    subfig = "({}) ".format(letters[0])
    letters = letters[1:]
    grid[0].set_title(subfig + "Ground truth\nuniform $|\mathbf{j}|$", pad=0)

    for i in range(len(grid)):
        if not i%(L+1) == L:
            grid.cbar_axes[i].axis('off')

    for i, (label, res) in enumerate(uni_result.items()):
        im = grid[i+1].matshow(j_density(res['gsol']), cmap='gray_r', **vlim)
        subfig = "({}) ".format(letters[0])
        letters = letters[1:]
        grid[i+1].set_title(subfig + label, pad=0)
    grid.cbar_axes[L].colorbar(im)

    vlim = get_j_density_range(uni_tester.g, uni_result, uniform_truth)
    for i, (label, res) in enumerate(uni_result.items()):
        im = grid[L+i+2].matshow(j_density(res['gsol']) - uniform_truth, 
                                 cmap='RdBu', **vlim)
        if not i:
            grid[L+i+2].set_title("Reconstruction error", pad=0)
    grid.cbar_axes[2 * L + 1].colorbar(im)

    vlim = get_j_density_range(para_tester.g, para_result)
    parabolic_truth = j_density(para_tester.g)
    grid[2*L+2].matshow(parabolic_truth, cmap='gray_r', **vlim)
    subfig = "({}) ".format(letters[0])
    letters = letters[1:]
    grid[2*L+2].set_title(subfig + "Ground truth \nparabolic $|\mathbf{j}|$", pad=0)

    for i, (label, res) in enumerate(para_result.items()):
        im = grid[2*L+3+i].matshow(j_density(res['gsol']), cmap='gray_r', **vlim)
        subfig = "({}) ".format(letters[0])
        letters = letters[1:]
        grid[2*L+3+i].set_title(subfig + label, pad=0)
    grid.cbar_axes[3 * L + 2].colorbar(im)

    vlim = get_j_density_range(para_tester.g, para_result, parabolic_truth)
    for i, (label, res) in enumerate(para_result.items()):
        im = grid[3*L+4+i].matshow(j_density(res['gsol'])-parabolic_truth, 
                                   cmap='RdBu', **vlim)
        if not i:
            grid[3*L+4+i].set_title("Reconstruction error", pad=0)
    grid.cbar_axes[4 * L + 3].colorbar(im)

    for i in range(len(grid)):
        grid[i].axis('off')

    grid.set_axes_pad((0.1, 0.55))
    return fig, grid

def diagnostic(results, tester, protocols):
    s = tester.sigma
    g = tester.g
    fig, axes = plt.subplots(3, len(results))
    for ax, pro, (k, v) in zip(axes[0], protocols, results.items()):
        ax.matshow(results[k]['residual'])
        ax.set_title(r"$\lambda\sigma$={}, ".format(pro['sigma']) + k)
        ax.axis('off')
    for ax, pro, (k, v) in zip(axes[1], protocols, results.items()):
        ax.matshow(absfft(results[k]['residual']))
        ax.axis('off')
    for ax, (k, v) in zip(axes[2], results.items()):
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

params = np.array([4.0, 1e-3, 1e-3])  # very small gaussian so no PSF
kernel = GaussianKernel((L, L), params)

# rescale g-field so flux is of order 1
g_uniform /= kernel.applyM(g_uniform).ptp()
g_parabolic /= kernel.applyM(g_parabolic).ptp()

admm_kwargs = {'iprint': 1, 'eps_rel': 1e-8, 'eps_abs': 1e-6, 'itnlim': 200,
               'rho': 1e-1}
TV_factor = 1.4
L_factor = 2.

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
D2h, D2v = makeD2(kernel._padshape)
protocols.append(
    dict(label="gaussian laplace prior", decon="LinearDeconvolver",
         sigma=L_factor * sigma, gamma=D2h + D2v)
)
protocols.append(
    dict(label="gaussian and finite support prior",
         decon="LinearDeconvolver", 
         sigma=L_factor * sigma, support_mask=mask)
)

# Ground truth flux
phi_uniform = kernel.applyM(g_uniform)
phi_parabolic = kernel.applyM(g_parabolic)

uniform_tester = Tester(g_uniform, phi_uniform, kernel, sigma)
parabolic_tester = Tester(g_parabolic, phi_parabolic, kernel, sigma)

sigma = estimate_noise(uniform_tester.phi.reshape(kernel._shape))
reg_protocol = []
factors = np.linspace(0.0001, 10, 100)
for fac in factors:
    reg_protocol.append(
        dict(label='factor = {}'.format(fac), decon='LinearDeconvolver',
             sigma=fac * sigma, #support_mask=mask, 
             deconv_kwargs=dict(solver='gcrotmk', atol=1e-10, tol=1e-10))
    )

if False:
    print("Performing uniform annulus tests")
    uniform_results = uniform_tester.test_protocols(protocols)
    
    print("Performing parabolic annulus tests")
    parabolic_results = parabolic_tester.test_protocols(protocols)
    
    fig, grid = compare_truth(
        {
            'Gaussian \nLaplacian': uniform_results['gaussian laplace prior'],
            'Gaussian \nFrobenius': uniform_results['gaussian prior'],
            'TV \nFrobenius': uniform_results['TV prior'],
        },
        uniform_tester, 
        {
            'Gaussian \nLaplacian': parabolic_results['gaussian laplace prior'],
            'Gaussian \nFrobenius': parabolic_results['gaussian prior'],
            'TV \nFrobenius': parabolic_results['TV prior'],
        },
        parabolic_tester
    )
    
    fig, grid = compare_truth(
        {
            'Gaussian finite\nsupport (FS)': 
              uniform_results['gaussian and finite support prior'],
            'TV and FS': uniform_results['TV and finite support prior'],
        },
        uniform_tester, 
        {
            'Gaussian and FS': 
              parabolic_results['gaussian and finite support prior'],
            'TV and FS': parabolic_results['TV and finite support prior'],
        },
        parabolic_tester
    )

#print("Performing parabolic regularization test")
#parabolic_reg_results = parabolic_tester.test_protocols(reg_protocol)
#fig, axe = plot_regularization(parabolic_reg_results, parabolic_tester,
#                               reg_protocol, 9, yshift_b=.02)

#print("Performing uniform regularization test")
#uniform_reg_results = uniform_tester.test_protocols(reg_protocol)
#fig, axe = plot_regularization(uniform_reg_results, uniform_tester,
#                               reg_protocol, 14)
plt.show()

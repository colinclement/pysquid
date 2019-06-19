import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

from pysquid.kernels.magpsf import GaussianKernel
from pysquid.util.linear_operators import curl

from annular_currents import annular_gfield
from tester import Tester

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

def j_density(g):
    return np.hypot(*curl(g))

def show_current_density(results, tester, cmap='gray_r'):
    fig, axes = plt.subplots(1, len(results)+1)
    fig.suptitle(
        'Reconstructed Current Density, sigma = {}'.format(tester.sigma)
    )
    axes[0].matshow(j_density(tester.g), cmap=cmap)
    axes[0].set_title('Ground truth')
    axes[0].axis('off')
    vmin, vmax = np.inf, -np.inf
    for v in results.values():
        j = j_density(v['gsol'])
        vmin = min(vmin, j.min())
        vmax = max(vmax, j.max())

    for ax, (k, v) in zip(axes[1:], results.items()):
        ax.matshow(j_density(v['gsol']), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(k)
        ax.axis('off')
    return fig, axes

L = 100
inner_rad = 10
width = 20
sigma = 0.05

g_uniform = annular_gfield(L, inner_rad, width)
g_parabolic = annular_gfield(L, inner_rad, width, profile="parabolic")

print("Making mask")
mask = np.zeros_like(g_uniform)
mask[g_uniform == 0.] = 1.
mask[g_uniform == g_uniform.max()] = 1.
mask = 1 * binary_erosion(mask == 1., border_value=True)

params = np.array([4.0, 1e-3, 1e-3])  # very small gaussian basically no PSF
kernel = GaussianKernel((L, L), params)

# rescale g-field so flux is of order 1
g_uniform /= kernel.applyM(g_uniform).ptp()
g_parabolic /= kernel.applyM(g_parabolic).ptp()

admm_kwargs = {'iprint': 0, 'eps_rel': 1e-7, 'eps_abs': 1e-5, 'itnlim': 50,
               'rho': 1e-2}
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

print("Performing uniform annulus tests")
uniform_results = uniform_tester.test_protocols(protocols)

print("Performing parabolic annulus tests")
parabolic_results = parabolic_tester.test_protocols(protocols)

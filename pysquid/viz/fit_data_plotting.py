import numpy as np
import matplotlib.pyplot as plt
from pysquid.util.helpers import curl

def fit_diagnostic(model, ref_data, asp, title=None): 
    jx, jy = curl(model.g_sol, model.dx, model.dy)
    fit_flux = (model.kernel.applyM(model.gfield).real + 
                model.extmodel.ext_flux)
    slx = np.s_[:,model.Lx/2-1:model.Lx/2+1]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    flux_arrays = [ref_data, fit_flux, ref_data-fit_flux]
    flux_lim = np.abs(np.array(flux_arrays)).max()
    flux_labels = ['Measured flux', 'Reconstructed flux', 'Residuals']
    
    j_arrays = [model.crop(jx), model.crop(jy)]
    j_labels = ['Recovered $J_x$','Recovered $J_y$']
    j_lim = np.abs(np.array(j_arrays)).max()
    
    for axrow in range(3):
        for axcol in range(3):
            ax = axes[axrow, axcol]
            if axrow == 0:
                if axcol < 2:
                    ax.matshow(flux_arrays[axcol], aspect = asp,
                               vmin=-flux_lim, vmax = flux_lim)
                else:
                    rlim = 4*flux_arrays[axcol].std()
                    ax.matshow(flux_arrays[axcol], aspect = asp,
                               vmin = -rlim, vmax=rlim)
                ax.axis('off')
                ax.set_title(flux_labels[axcol], fontsize=30)
            elif axrow == 1:
                if axcol < 2:
                    ax.matshow(j_arrays[axcol], aspect = asp,
                               vmin=-j_lim, vmax = j_lim)
                    ax.axis('off')
                    ax.set_title(j_labels[axcol], fontsize=30)
                if axcol == 2:
                    res = flux_arrays[2]
                    hist, bins = np.histogram(res, bins=60, normed=True, range=(-4*res.std(), 4*res.std()))
                    p = lambda x: np.exp(-(x/model.sigma)**2/2)/np.sqrt(2*np.pi*model.sigma**2)
                    resp = 0.5*(bins[1:]+bins[:-1])
                    ax.plot(resp, hist, label='Fit residuals')
                    ax.plot(resp, p(resp), label='Estimated noise')
                    ax.legend(loc='lower center', fontsize=16)
                    ax.set_title("Residual distribution", fontsize=30)
                    ax.set_yscale('log')
                    ax.set_ylabel('log $P(r)$', fontsize=20)
                    ax.set_xlabel('$r$', fontsize=20)
            else:
                if axcol == 1:
                    ax2 = ax.twinx()
                    lj = ax.plot(j_arrays[0][slx].mean(1), c = 'b', label='Horizontal current')
                    lf = ax2.plot(ref_data[slx].mean(1), c = 'g', label='Flux data')
                    lab = lj + lf
                    ax.legend(lab, [l.get_label() for l in lab], 
                              loc='upper right', fontsize=16)
                    ax.set_title("Vertical Cross sections", fontsize=30)
                    ax.set_xlabel("Vertical distance", fontsize=20)
                    ax.set_ylabel("$J_x$", fontsize=20)
                    ax2.set_ylabel("Flux", fontsize=20)
                    ax.set_xlim([0, model.Ly])
                else:
                    ax.axis('off')
    title = title if title is not None else "Reconstruction with $\mu$={}".format(model.linearModel.mu_reg)
    plt.suptitle(title, fontsize=25) 
    return fig, axes



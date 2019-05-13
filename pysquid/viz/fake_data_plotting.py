import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from pysquid.util.helpers import curl

def simpleaxis(ax):
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
def diagnostic(model, ref_g, ref_flux, admm, asp = 0.5,
               title = None):
    """
    Create fancy diagnostic plot for fake data fitting.
    input:
        model : instance of LinearFluxModel evaluated at
                current best gfield
        ref_g : The reference (ground truth) g-field
        ref_flux : The (noisy) data shown to the optimizer
        admm : Instance of LinearModelTV_ADMM
        asp : aspect ratio of images
        title : str title of plot

    returns:
        fig, axes
    """
    jx, jy = curl(model.g_sol)#model.gfield + model.extmodel.ext_g)
    jx, jy = model.crop(jx), model.crop(jy)
    true_jx, true_jy = curl(ref_g)
    true_jx, true_jy = model.crop(true_jx), model.crop(true_jy)
    fit_flux = model.modelflux.real.reshape(model.Ly, -1)
    fig, axes = plt.subplots(3, 4, figsize=(21, 13))
    
    _, sx, sy = model.pmap['psf_params']
    shx, shy = int(3*sx), int(3*sy)
    ker = np.fft.fftshift(model.kernel.PSF.real)[model.Ly_pad-shy:model.Ly_pad+shy,
                                                 model.Lx_pad-shx+1:model.Lx_pad+shx]

    residuals = ref_flux - fit_flux
    sly, slx = np.s_[model.Ly_pad/2,:],  np.s_[:,model.Lx_pad/2]
    tv_ref_g = admm.TV(ref_g).reshape(model.Ly_pad, -1)
    tv_fit_g = admm.TV(model.g_sol).reshape(model.Ly_pad, -1)

    reference = [ref_flux, true_jx, true_jy]
    ref_label = ['Reference flux', 'True $J_x$', 'True $J_y$']
    reconst = [fit_flux, jx, jy]
    rec_label = ['Reconstructed flux', 'Recovered $J_x$', 'Recovered $J_y$']
    
    jxlim = min(np.abs(jx).max(), np.abs(true_jx).max())
    jylim = min(np.abs(jy).max(), np.abs(true_jy).max())
    flim = min(np.abs(ref_flux).max(), np.abs(fit_flux).max())
    
    ref_lim = [flim, jxlim, jylim]
    rec_lim = [flim, jxlim, jylim]
    
    for axrow in range(3):
        for axcol in range(4):
            axe = axes[axrow, axcol]
            if axcol == 0: #reference
                lim = ref_lim[axrow]
                axe.matshow(reference[axrow], aspect=asp, vmin=-lim, vmax=lim)
                axe.axis('off')
                axe.set_title(ref_label[axrow], fontsize=30)
                if axrow == 1: #Kernel inset
                    yy, xx = reference[axrow].shape
                    yk, xk = ker.shape
                    py, px = yk/(yy+0.), xk/(xx+0.)
                    insetax = inset_axes(axe,
                                         width="{}%".format(px*100), 
                                         height="{}%".format(py*100),
                                         loc=8)
                    insetax.matshow(ker, cmap='Greys')
                    simpleaxis(insetax)
                    insetax.set_ylabel('PSF', fontsize=18, rotation='horizontal',
                                       labelpad=20, y=.2)
                if axrow == 1:
                    axe.axvline(model.Lx/2, color='k', alpha = 0.1)
                if axrow == 2:
                    axe.axhline(model.Ly/2, color='k', alpha = 0.1)
            elif axcol == 1:
                lim = rec_lim[axrow]
                axe.matshow(reconst[axrow], aspect=asp, vmin=-lim, vmax=lim)
                axe.axis('off')
                axe.set_title(rec_label[axrow], fontsize=30)
                if axrow == 1:
                    axe.axvline(model.Lx/2, color='k', alpha = 0.1)
                if axrow == 2:
                    axe.axhline(model.Ly/2, color='k', alpha = 0.1)
            elif axcol == 2:
                if axrow == 0:
                    axe.matshow(residuals, aspect=asp, vmin = -4*model.sigma,
                                vmax = 4*model.sigma)
                    axe.axis('off')
                    axe.set_title("Residuals", fontsize=30)
                elif axrow == 1:
                    axe.plot(true_jx[:,model.Lx/2], label=r'Reference')
                    axe.plot(jx[:,model.Lx/2], label=r'Reconstructed')
                    axe.set_title(r'$J_x$ cross section', fontsize=30)
                    axe.set_xlim([0, model.Ly])
                elif axrow == 2:
                    axe.plot(true_jy[model.Ly/2], label=r'Reference')
                    axe.plot(jy[model.Ly/2], label=r'Reconstructed')
                    axe.set_title(r'$J_y$ cross section', fontsize=30)
                    axe.set_xlim([0, model.Lx])
                axe.legend(loc='best', fontsize=16)
                axe.set_ylabel("Current", fontsize=18)
            elif axcol == 3: #histogram, g/TV cross sections
                if axrow == 0:
                    pass #Histogram of residuals
                    hist, bins = np.histogram(residuals.ravel(), bins=60, density=True)
                    p = lambda x: np.exp(-(x/model.sigma)**2/2)/np.sqrt(2*np.pi*model.sigma**2)
                    resp = 0.5*(bins[1:]+bins[:-1])
                    axe.plot(resp, hist, label = 'Residuals')
                    axe.plot(resp, p(resp), label = 'True noise')
                    axe.set_yscale('log') 
                    axe.legend(loc='best', fontsize=16)
                    axe.set_title("Residual distribution", fontsize=30)

                elif axrow == 1 or axrow == 2:
                    sl = {1: slx, 2: sly}[axrow] 
                    axe2 = axe.twinx()

                    gslice, ref_g_slice = model.g_sol[sl], ref_g[sl]
                    axe.plot(gslice, label='Fit $g$', lw = 4, c = 'k')
                    axe.plot(ref_g_slice, label = 'True $g$', lw = 3, c = 'r', alpha = 0.6)
                    axe.set_ylabel(r"$g$", fontsize=24)
                    axe.set_ylim([3*gslice.min(), 1.4*gslice.max()])
                   
                    tv_fit_slice, tv_ref_slice = tv_fit_g[sl], tv_ref_g[sl] 
                    axe2.plot(tv_fit_slice, lw = 2, label = "TV of fit $g$", c = 'k', alpha = 0.4)
                    axe2.plot(tv_ref_slice, ":", lw = 3, label = "TV of true $g$", c = 'r', alpha = 0.6)
                    axe2.set_ylabel("TV(g)", fontsize=24)
                    axe2.set_ylim([0, 2*tv_ref_slice.max()])
                    
                    if axrow == 1: 
                        axe.set_title("Vertical Cross Section", 
                                      fontsize=20)
                        axe.axvline(model.py, color = 'k', lw=0.4)
                        axe.axvline(model.py+model.Ly, color = 'k', lw=0.4)
                        axe.set_xlim([0,model.Ly_pad])
                        axe2.legend(loc='upper right', fontsize=16)
                        axe.legend(loc='lower left', fontsize=16) 
                    else:
                        axe.set_title("Horizontal Cross Section",
                                      fontsize=20)
                        axe.axvline(model.px, color = 'k', lw=0.4)
                        axe.axvline(model.px+model.Lx, color = 'k', lw=0.4)
                        axe.set_xlim([0,model.Lx_pad])
                        axe2.legend(loc='upper right', fontsize=16)
                        axe.legend(loc='upper left', fontsize=16) 

    title = title if title is not None else "Reconstruction with $\mu$ = {}".format(admm.mu_reg) 
    plt.suptitle(title, fontsize=25)
    return fig, axes


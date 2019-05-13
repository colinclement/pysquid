import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

from pysquid.rnet import ResistorNetworkModel
from pysquid.model import FluxModelTVPrior
import pysquid.kernels.magpsf as kernel
import pysquid.viz.fit_data_plotting as fdp
import pysquid.util.helpers as hpr

loc = os.path.dirname(os.path.realpath(__file__))

datarec = np.load(os.path.join(loc, 
                               '../gendata/flux_and_reconstructions.npz'))['data']
Ly, Lx = datarec[0][1]['scan'].shape

mask = np.load(os.path.join(loc,'../gendata/hallprobe_interpolated.npy'))
imgspacing = [.16, .73] #y,x pixel spacing in micrometers
dy, dx = imgspacing

interpmask = 2*mask-1
xcorner, ycorner = 224, 840

fitpsf = np.load(os.path.join(loc, '../gendata/psf_params_2_gaussians.npz'))['psf_params']

scaledpsf = hpr.changeUnits(fitpsf.ravel(), dx = 3.2, dy = 2.6).ravel()

pdict = {'psf_params': np.r_[.8, scaledpsf], 'J_ext': np.array([700]), 
         'sigma': np.array([0.013])}

netmodel = ResistorNetworkModel(mask, phi_offset = [ycorner, xcorner], 
                                gshape=(Ly, Lx), electrodes=[50,550])

model = FluxModelTVPrior(datarec[0][1]['scan'].copy(), 
                         param_dict = pdict,
                         extmodel = netmodel, 
                         kerneltype = kernel.MogKernel,
                         padding = (40, 10),
                         mu_reg = 2.2,
                         dy = dy, dx = dx)

#Find external currents for each imaage
topedge = model.extmodel._unitJ_flux[model.extmodel._cut][0].copy()
today = datetime.date(datetime.now()).isoformat()
 
mu = model.linearModel.mu_reg
filename = os.path.join(loc,
                        'HgTe/{}-mu_{}-HgTe-reconstruction.npz'.format(today, mu))
g0 = np.random.randn(model.N_pad)/np.sqrt(model.N_pad)

for n, sc in enumerate(datarec):
    print("{}: Reconstructing for V={}".format(n, sc[0]))
    fluxdata = sc[1]['scan'].copy()
    J_ext = topedge.dot(fluxdata[0])/topedge.dot(topedge)
    
    model.fluxdata[:,:] = fluxdata
    sigma_est = hpr.noise_estimate(fluxdata)/2.
    model.computeNLL(np.array([J_ext, sigma_est]), ['J_ext', 'sigma'])
    
    subflux = fluxdata - model.extmodel.ext_flux
    gsol = model.linearModel.solve(subflux.ravel(), g0.copy(),
                                   extmodel = model.extmodel,
                                   itnlim = 150, iprint = 1)
    
    model.computeNLL(gsol, ['gfieldflat'])
    newsigma = model.residuals.std()
    print("Re-fitting with sigma={}".format(newsigma))
    model.computeNLL(np.array([newsigma]), ['sigma'])
    gsol = model.linearModel.solve(subflux.ravel(), gsol.copy(),
                                   extmodel = model.extmodel,
                                   itnlim = 250, iprint = 1)
    model.computeNLL(gsol, ['gfieldflat'])
    
    sc[1]['gfield'] = gsol.copy()
    sc[1]['g_sol'] = model.g_sol.copy()
    sc[1]['ext_flux'] = model.extmodel.ext_flux.copy()
    sc[1]['ext_g'] = model.extmodel.ext_g.copy()
    sc[1]['g_sol_flux'] = model.kernel.applyM(model.gfield).real
    sc[1]['model_flux'] = sc[1]['g_sol_flux'] + model.extmodel.ext_flux
    sc[1]['J_ext'] = model.extmodel.J_ext
    sc[1]['psf_params'] = model.pmap['psf_params'].copy()
    sc[1]['sigma'] = model.sigma
    sc[1]['mu_reg'] = model.pmap['mu_reg'].copy()
    
    #Save updated datarec
    with open(filename, 'w') as outfile:
        np.savez(outfile, data = datarec)

    #Make figures
    title = "V = {}, $mu$ = {}".format(sc[0], model.linearModel.mu_reg)
    fig, axes = fdp.fit_diagnostic(model, fluxdata, dy/dx, title = title)
                                   
    figname = "HgTe/plots/{t}-V_{:.1e}-diagnostic.pdf".format(sc[0], t=today)
    plt.savefig(os.path.join(loc, figname))
    plt.close()    
    g0 = gsol.copy().ravel()



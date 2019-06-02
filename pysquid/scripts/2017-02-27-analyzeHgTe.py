import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

from pysquid.rnet import ResistorNetworkModel
from pysquid.model import FluxModelTVPrior
import pysquid.kernels.magpsf as kernel
import pysquid.viz.fit_data_plotting as fdp
import pysquid.util.helpers as hpr

import pysquid.deconvolve as deconvolve

loc = os.path.dirname(os.path.realpath(__file__))

datarec = np.load(os.path.join(loc, 
                               '../gendata/flux_and_reconstructions.npz'))['data']
Ly, Lx = datarec[0][1]['scan'].shape

mask = np.load(os.path.join(loc,'../gendata/hallprobe_interpolated.npy'))
imgspacing = [.16, .73] #y,x pixel spacing in micrometers
dy, dx = imgspacing

interpmask = 2*mask-1
xcorner, ycorner = 224, 840

fitpsf = np.load(os.path.join(loc, '../gendata/2016-03-12-psf_params_2_gaussians.npz'))['psf_params']

scaledpsf = hpr.changeUnits(fitpsf.ravel(), dx = 3.2, dy = 2.6).ravel()

pdict = {'psf_params': np.r_[.8, scaledpsf], 'J_ext': np.array([700]), 
         'sigma': np.array([0.013])}

kern = kernel.MogKernel(datarec[0][1]['scan'].shape,
                        pdict['psf_params'], padding=(40, 10),
                        dy=dy, dx=dx)

# This is a mess... TODO: Clean up dependencies
kern_netmodel = kernel.MogKernel(mask.shape, pdict['psf_params'], 
                                 dy=dy, dx=dx)
netmodel = ResistorNetworkModel(mask, kernel=kern_netmodel, padding=(40, 10),
                                phi_offset=[ycorner, xcorner], 
                                gshape=kern._shape, electrodes=[50,550])

solver = deconvolve.TVDeconvolver(kern, 1E-4)

#Find external currents for each imaage
topedge = netmodel.ext_flux[0].copy()
today = datetime.date(datetime.now()).isoformat()
 
mu = 2.2
nameformat = 'HgTe/{}-mu_{}-HgTe-reconstruction.npz'
filename = os.path.join(loc, nameformat.format(today, mu))

for n, sc in enumerate(datarec):
    print("{}: Reconstructing for V={}".format(n, sc[0]))
    fluxdata = sc[1]['scan'].copy()
    J_ext = topedge.dot(fluxdata[0])/topedge.dot(topedge)
    sigma_est = hpr.noise_estimate(fluxdata)

    solver.gamma = mu * sigma_est**2
    print("Gamma = {}".format(solver.gamma))
    solver.set_g_ext(J_ext * netmodel.ext_g)
    subflux = (fluxdata - J_ext * netmodel.ext_flux).ravel()
    # Indent printing for easier reading
    solver.appendprint = "\t"
    gsol0 = solver.deconvolve(subflux, iprint=1, rho=solver.gamma/10.,
                             algorithm='minimize_fastrestart', itnlim=100)

    #res = solver.M.dot(gsol) - subflux
    #sigma_meas = res.std()
    #solver.gamma = mu * sigma_meas**2

    #print("Re-fitting with gamma={}".format(solver.gamma))
    #gsol = solver.deconvolve(subflux, iprint=1, rho=solver.gamma/10.,
    #                         algorithm='minimize_fastrestart', itnlim=80)
    
    sc[1]['g_sol'] = gsol0.reshape(*kern._padshape)
    sc[1]['ext_flux'] = J_ext * netmodel.ext_flux
    sc[1]['ext_g'] = J_ext * netmodel.ext_g.copy()
    sc[1]['model_flux'] = solver.M.dot(gsol0).reshape(*kern._shape) + J_ext*netmodel.ext_flux
    sc[1]['J_ext'] = J_ext 
    sc[1]['psf_params'] = pdict['psf_params'].copy()
    sc[1]['sigma'] = (solver.M.dot(gsol0)-subflux.ravel()).std()
    sc[1]['gamma'] = solver.gamma
    
    #Save updated datarec
    with open(filename, 'w') as outfile:
        np.savez(outfile, data = datarec)

    #Make figures
    #title = "V = {}, $gamma$ = {}".format(sc[0], gamma) 
    #fig, axes = fdp.fit_diagnostic(model, fluxdata, dy/dx, title = title)
    #                               
    #figname = "HgTe/plots/{t}-V_{:.1e}-diagnostic.pdf".format(sc[0], t=today)
    #plt.savefig(os.path.join(loc, figname))
    #plt.close()    
    #g0 = gsol.copy().ravel()



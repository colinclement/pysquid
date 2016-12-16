import numpy as np
import scipy as sp
import os
from scipy.io import savemat, loadmat

from pysquid.rnet import ResistorNetworkModel
from pysquid.model import FluxModelTVPrior
from pysquid.kernel.magpsf import GaussianKernel
from pysquid.util.helpers import *

loc = os.path.dirname(os.path.realpath(__file__))
mask = np.load(os.path.join(loc, 'fake_data_hallprobe_interpolated.npy'))
Ly, Lx = 300, 200
y_by_x_ratio = 0.5

true_params = {'J_ext': np.array([1000]), 
              'sigma': np.array([1.40803307e-02])}

#true_params['psf_params'] =  p.array([3.26043651e+00,   3.40755272e+00,   5.82311678e+00]
true_params['psf_params'] =  np.array([3.,  6.,  10.])

fake_data_offset = [840-100, 185]#fake_offset

netmodel = ResistorNetworkModel(mask, phi_offset = fake_data_offset, 
                                gshape=(Ly, Lx), electrodes=[50,550])

kernel = GaussianKernel(mask.shape, psf_params=true_params['psf_params'],
                       dx = 1./y_by_x_ratio)
                        
netmodel.kernel = kernel
netmodel.updateParams('J_ext', np.array([1000]))
jx, jy = curl(netmodel.ext_g, dx = kernel.dx, dy = kernel.dy)

if __name__ == '__main__':

    np.savez(os.path.join(loc, 'fake_data.npz'), 
             offset = fake_data_offset, 
             psf_params = true_params['psf_params'],
             J_ext = true_params['J_ext'], all_g = netmodel.gfield,
             unitJ_flux = netmodel._unitJ_flux,
             image_g = netmodel.ext_g,
             image_flux = netmodel.ext_flux)
    
    
    savemat(os.path.join(loc,'fake_data.mat'),
            {'scan': netmodel.ext_flux,
            'scan_plus_noise': (netmodel.ext_flux + 
                                np.random.randn(Ly, Lx)*0.02),
             'true_jx': jx, 'true_jy': jy,
             'true_g_field': netmodel.ext_g,
             'true_PSF': kernel.PSF.real})





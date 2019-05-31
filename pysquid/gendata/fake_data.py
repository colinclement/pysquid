from pysquid.rnet import ResistorNetworkModel
from pysquid.model import FluxModelTVPrior
from pysquid.kernels.magpsf import GaussianKernel
from pysquid.gendata.hall_probes_to_fake_data import make_mask
from pysquid.util.helpers import *

from scipy.io import savemat, loadmat
import os
import numpy as np
import scipy as sp


def make_fake_data():
    loc = os.path.dirname(os.path.realpath(__file__))
    maskfile = os.path.join(loc, 'fake_data_hallprobe_interpolated.npy')
    if not os.path.exists(maskfile):
        print("Making new mask")
        make_mask()
    
    mask = np.load(maskfile)
    Ly, Lx = 300, 200
    y_by_x_ratio = 0.5
    
    true_params = {'J_ext': np.array([1000]), 
                  'sigma': np.array([1.40803307e-02])}
    
    #true_params['psf_params'] =  p.array([3.26043651e+00,   3.40755272e+00,
    #5.82311678e+00])
    true_params['psf_params'] =  np.array([3.,  6.,  10.])
    
    fake_data_offset = [840-100, 185]#fake_offset
    
    netmodel = ResistorNetworkModel(mask, phi_offset = fake_data_offset, 
                                    gshape=(Ly, Lx), electrodes=[50,550])
    
    kernel = GaussianKernel(mask.shape, params=true_params['psf_params'],
                            rxy=1./y_by_x_ratio)
                            
    netmodel.kernel = kernel
    netmodel.updateParams('J_ext', np.array([1000]))
    jx, jy = curl(netmodel.g_ext, dx=kernel.rxy)

    np.savez(os.path.join(loc, 'fake_data.npz'), 
             offset = fake_data_offset, 
             psf_params = true_params['psf_params'],
             J_ext = true_params['J_ext'], all_g = netmodel.gfield,
             unitJ_flux = netmodel._unitJ_flux,
             image_g = netmodel.g_ext,
             image_flux = netmodel.ext_flux)
    
    #savemat(os.path.join(loc,'fake_data.mat'),
    #        {'scan': netmodel.ext_flux,
    #        'scan_plus_noise': (netmodel.ext_flux + 
    #                            np.random.randn(Ly, Lx)*0.02),
    #         'true_jx': jx, 'true_jy': jy,
    #         'true_g_field': netmodel.g_ext,
    #         'true_PSF': kernel.PSF.real})

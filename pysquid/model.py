"""
model.py

author: Colin Clement
Date: 2015-9-23

"""

import sys
import numpy as np
from scipy.optimize import leastsq
from copy import copy
from collections import defaultdict

from pysquid.param import ParameterMap
from pysquid.component import ModelComponent
import pysquid.linearModel as lm
from pysquid.kernels.magpsf import GaussianKernel
from pysquid.util.linear_operators import curl


class FluxModel(ModelComponent):
    def __init__(self, fluxdata, param_dict=defaultdict(list),
                 kerneltype=None, extmodel=None, padding=None, **kwargs):
        """
        input:
            fluxdata : float array of shape (ly, lx) of flux data to model
            param_dict : (dict) keys are parameter names, values are numpy
                        arrays
            kerneltype : Subclass of Kernel for computing PSF/M matrix products

            extmodel : ResistorNetworkModel for modeling currents outside
                       fluxdata image

            padding : (py, px) tuple of ints for padding g-field outside image
        **kwargs are passed into Kernels and LinearModel,
                see kwargs for those classes
        """
        self.fluxdata = fluxdata
        super(FluxModel, self).__init__(fluxdata.shape, padding, **kwargs)

        self.pmap = ParameterMap()
        self.gfieldflat = np.zeros(self.N_pad)

        self.kerneltype = kerneltype or GaussianKernel
        self.kernel = self.kerneltype(self._shape, param_dict['psf_params'],
                                      padding, **kwargs)
        psf_dependence = [self.kernel]

        self.extmodel = extmodel
        if self.extmodel:
            self.extmodel.padding = self._padding
            # NOTE: rnet does not use padding in kernel!
            self.extmodel.kernel = self.kerneltype(self.extmodel._shape,
                                                   param_dict['psf_params'],
                                                   **kwargs)
            J_ext = param_dict['J_ext'] or np.array([1.])
            self.pmap.register('J_ext', [self.extmodel], J_ext)
            psf_dependence += [self.extmodel.kernel, self.extmodel]

        self._setLinearModel()

        self.pmap.register('psf_params', psf_dependence,
                           param_dict['psf_params'])
        self.pmap.register('sigma', [self.linearModel, self],
                           param_dict['sigma'])
        self.pmap.register('mu_reg', [self.linearModel],
                           np.array([kwargs.get('mu_reg', 2.5)]))
        self.pmap.register('gfieldflat', [self], self.gfieldflat)

    def _setLinearModel(self, shape, kernel, padding, **kwargs):
        pass

    @property
    def params(self):
        return self.pmap.params.copy()

    @property
    def gfield(self):
        return self.gfieldflat.reshape(self.Ly_pad, -1)

    @property
    def fluxdataflat(self):
        return self.fluxdata.ravel()

    def getParams(self, names=None):
        if names is None:
            names = self.pmap.names
        p = np.array([])
        for name in names:
            p = np.r_[p, self.pmap[name].copy()]
        return p

    def updateParams(self, name, value):
        if name == 'sigma':
            self.sigma = value[0]
        if name == 'gfieldflat':
            self.gfieldflat = value

    def updateParamVector(self, param_vec, param_names=None):
        param_names = param_names or self.pmap.names
        N = 0
        for name in param_names:
            l = len(self.pmap[name])
            self.pmap.updateParams([name], [param_vec[N:N+l]])
            N += l

    def solveLinearModel(self, params=None, g0=None, ext_flux=None,
                         ext_g=None, verbose=0, **kwargs):
        if params is not None:
            self.updateParamVector(params)
        if g0 is None:
            g0 = np.sqrt(self.N_pad)*np.random.randn(self.N_pad)
        self.updateParamVector(g0.ravel(), ['gfieldflat'])
        self.computeNLL()
        if verbose > 0:
            print("Initial NLL = {:<6.4f}".format(self.NLL))

        if self.extmodel:
            ext_g = ext_g if ext_g is not None else self.extmodel.ext_g.ravel()
            if ext_flux is None:
                ext_flux = self.extmodel.ext_flux.ravel()
            # Subtract external loop field from padded flux image
            Ly_pad, Lx_pad, py, px = self.Ly_pad, self.Lx_pad, self.py, self.px
            self._flux0 = np.zeros((Ly_pad, Lx_pad))
            self._flux0[py:py+self.Ly, px:px+self.Lx] = self.fluxdata.copy()[:, :]
            self._flux0 -= self.extmodel.ext_flux

            gfieldflat = self.linearModel.solve(self._flux0.ravel(), g0 = g0,
                                                verbose = verbose, 
                                                ext_g = ext_g, **kwargs)
        else:
            self._flux0 = self.fluxdataflat
            gfieldflat = self.linearModel.solve(self._flux0, g0 = g0, **kwargs)
        
        self.updateParamVector(gfieldflat, ['gfieldflat'])
        self.computeNLL()
        if verbose>0:
            print("Final NLL = {:<6.4f}".format(self.NLL))
        
        return self.gfieldflat 

    def computeResiduals(self, params = None, param_names = None,
                         gfield = None, **kwargs):
        pass

    def computeNLL(self, params = None, param_names = None, **kwargs):
        pass   
    
    def computeGradResiduals(self, params = None, param_names = None,
                             **kwargs):
        pass

    def computeGradNLL(self, params = None, param_names = None, 
                       **kwargs):
        pass 

    def fitModel(self, init_params = None, param_names = None,
                 verbose=False, linearkwargs = None,
                 **kwargs):       
        pass
                 
    @property
    def g_sol(self):
        if self.extmodel:
            return self.gfield + self.extmodel.ext_g
        else:
            return self.gfield

    @property
    def g(self):
        return self.crop(self.g_sol)

    @property
    def J_sol(self):
        jx, jy = curl(self.g_sol)
        Ly, Lx = self.Ly, self.Lx
        py, px = self.py, self.px 
        return jx[py:py+Ly+1, px:px+Lx], jy[py:py+Ly,px:px+Lx+1] 


class FluxModelTVPrior(FluxModel):
    def __init__(self, fluxdata, param_dict = defaultdict(list),
                 kerneltype = None, extmodel = None, 
                 padding = None, **kwargs):
        """
        input:
            fluxdata : float array of shape (ly, lx) of flux data to model 
            param_dict : (dict) keys are parameter names, values are numpy
                        arrays
            kerneltype : Subclass of Kernel for computing PSF/M matrix products

            extmodel : ResistorNetworkModel for modeling currents outside
                       fluxdata image 
                            
            padding : (py, px) tuple of ints for padding g-field outside image
        **kwargs are passed into Kernels and LinearModel, 
                see kwargs for those classes
        """
        self._order = kwargs.get('order', 1.)
        super(FluxModelTVPrior, self).__init__(fluxdata, param_dict,
                                               kerneltype, extmodel,
                                               padding, **kwargs)

    def _setLinearModel(self):
        self.linearModel = lm.LinearModelTV_ADMM(self)

    def computeResiduals(self, params = None, param_names = None,
                         gfield = None, **kwargs):
        if params is not None:
            self.updateParamVector(params, param_names)
        gfield = gfield if gfield is not None else self.gfield
        ext_flux = self.extmodel.ext_flux.ravel() if self.extmodel else 0.
        ext_g = self.extmodel.ext_g.ravel() if self.extmodel else None
        #self.modelflux = (self.kernel.applyM(gfield).ravel() + ext_flux).real
        self.residuals = self.linearModel.computeResiduals(self.fluxdataflat - 
                                                           ext_flux, self.gfield) 
        return self.residuals

    def computeNLL(self, params = None, param_names = None, **kwargs):
        self.computeResiduals(params, param_names, **kwargs)
        ext_g = self.extmodel.ext_g if self.extmodel else 0.
        mu = self.linearModel.mu
        self.NLL = 0.5*(((self.residuals**2).sum() +
                         mu*self.linearModel.TV(self.gfield +
                                                ext_g).sum())/self.sigma**2 + 
                        self.N*np.log(2*np.pi*self.sigma**2))
        return self.NLL


class FluxModelGaussianPrior(FluxModel):
    def __init__(self, fluxdata, param_dict = defaultdict(list),
                 kerneltype = None, extmodel = None, 
                 padding = None, **kwargs):
        """
        input:
            fluxdata : float array of shape (ly, lx) of flux data to model 
            param_dict : (dict) keys are parameter names, values are numpy
                        arrays
            kerneltype : Subclass of Kernel for computing PSF/M matrix products

            extmodel : ResistorNetworkModel for modeling currents outside
                       fluxdata image 
                            
            padding : (py, px) tuple of ints for padding g-field outside image
        **kwargs are passed into Kernels and LinearModel, 
                see kwargs for those classes
        """
        super(FluxModelGaussianPrior, self).__init__(fluxdata, param_dict,
                                                     kerneltype, extmodel,
                                                     padding, **kwargs)

    def _setLinearModel(self, shape, kernel, padding, **kwargs):
        self.linearModel = lm.LinearModelOrthProj(shape, kernel, 
                                               padding, **kwargs)

    def computeResiduals(self, params = None, param_names = None,
                         gfield = None, **kwargs):
        if params is not None:
            self.updateParamVector(params, param_names)
        gfield = gfield if gfield is not None else self.gfield
        ext_flux = self.extmodel.ext_flux.ravel() if self.extmodel else 0.
        ext_g = self.extmodel.ext_g.ravel() if self.extmodel else None
        self.modelflux = (self.kernel.applyM(gfield).ravel() + ext_flux).real
        self.residuals = self.linearModel.computeResiduals(self.fluxdataflat - 
                                                           ext_flux,
                                                           self.gfield, ext_g) 
        return self.residuals

    def computeNLL(self, params = None, param_names = None, **kwargs):
        self.computeResiduals(params, param_names, **kwargs)
        self.NLL = 0.5*((self.residuals**2).sum() +
                         self.N*np.log(2*np.pi*self.sigma**2))
        return self.NLL

    def computeGradResiduals(self, params = None, param_names = None,
                             **kwargs):
        if params is not None:
            self.computeNLL(params, param_names, **kwargs)
        
        N_p = len(self.kernel.psf_params)
        d_modelflux = self.kernel.computeGradients(self.gfield)
    
        if self.extmodel:
            ext_flux = self.extmodel.ext_flux.ravel()
            ext_g = self.extmodel.ext_g.ravel()
            d_extflux = self.extmodel.computeGradients()
            d_flux_residuals = (- d_extflux - 
                                  d_modelflux).reshape(-1, N_p)/self.sigma
            self.d_residuals = np.r_[d_flux_residuals,
                                     np.zeros((self.N_pad, N_p))]

            Gammadot = self.linearModel.Gammadot
            J_ext = self.pmap['J_ext']
            self.d_residuals = np.c_[self.d_residuals, 
                                     np.r_[-ext_flux/J_ext/self.sigma,
                                           -Gammadot(ext_g/J_ext)]]
        else:
            self.d_residuals = np.r_[d_modelflux/self.sigma,
                                     np.zeros((self.N_pad, N_p))]
        return self.d_residuals

    def computeGradNLL(self, params = None, param_names = None, 
                       **kwargs):
        if params is not None:
            self.computeNLL(params, param_names, **kwargs)
        
        d_residuals = self.computeGradResiduals(params, param_names)
        self.d_NLL = self.residuals.T.dot(d_residuals)
        return self.d_NLL

    def fitModel(self, init_params = None, param_names = None,
                 verbose=False, linearkwargs = None,
                 **kwargs):
        if init_params is not None:
            self.computeNLL(init_params, param_names, **kwargs)
        if linearkwargs is None:
            linearkwargs = {'atol': 1E-3, 'btol': 1E-3}
        nlimit = kwargs.get('nlimit', 40)
        nll_tol = kwargs.get('nll_tol', 1E1)
        maxfev = kwargs.get('maxfev', 8)
        ftol = kwargs.get('ftol', 1E-3)
        sigstepsize = kwargs.get('sigstepsize', 0.001)
        minNLL = copy(self.NLL)
        minparams = self.getParams()
        for itn in range(nlimit):
            if verbose:
                print("Starting iteration {}\n".format(itn))
            NLL0 = copy(self.NLL)
            #Nonlinear params
            nonlinear_names = ['psf_params', 'J_ext']
            self.lsqdict = leastsq(self.computeResiduals,
                                   self.getParams(nonlinear_names), 
                                   args = (nonlinear_names,),
                                   Dfun = self.computeGradResiduals,
                                   full_output = True, 
                                   maxfev=maxfev, ftol=ftol)
            self.updateParamVector(self.lsqdict[0], nonlinear_names)
            #Linear params
            self.solveLinearModel(**linearkwargs)
            self.computeNLL()
            if self.NLL < minNLL:
                minNLL = copy(self.NLL)
                minparams = self.getParams()

            deltaNLL = self.NLL - NLL0
            if verbose:
                print("After {} NLL = {}, deltaNLL = {}\n".format(itn, self.NLL,
                                                                  deltaNLL))
                print("Parameters:\n")
                for name in self.pmap.names:
                    print("\t{} = {}\n".format(name, self.pmap[name]))
                sys.stdout.flush() 
            if np.abs(deltaNLL) < nll_tol:
                if verbose:
                    print("Change in NLL is less than {}\n".format(nll_tol))
                break
        
        self.updateParamVector(minparams)
        self.solveLinearModel()
        return self.params





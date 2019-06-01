"""
finite-k-support.py

    author: Colin Clement
    email: colin.clement@gmail.com
    date: 2016-03-25


"""


from __future__ import division
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spl

from pysquid.util.linear_operators import MyLinearOperator
from pysquid.linearModel import LinearModel
from pysquid.util.fftw import WrapFFTW


class LinearModel_finitek_ADMM(LinearModel):
    """
    LinearModelTV computes the maximum-likelihood estimate of a
    deconvolution problem with a  Total Variation (TV) regularizing
    prior on the current dipole field g. I.e.,
    log p(data | g, z) = -1/2\sigma^2 || \phi - Mg || ^2 
                         - \mu \sigma^2 TV(g)
        where TV(g) = \sum_i ((d g_i dx)^p + (d g_i dy)^p)^1/p
        for some positive real number p
    This routine assumes constant z (psf_params, etc) and optimizes
    this probability w.r.t. g
    """

    def __init__(self, model, mask, **kwargs):
        """
        input:
            model: Instance of FluxModel

            kwargs:
                p : float (default 0.5) for TV regularization power
                mu_reg : float imposing the regularization
                order : Order of derivative operatore in TV regularizer
        """
        # Could now implement p for any power (might help down the line...)
        super(LinearModel_finitek_ADMM, self).__init__(model._shape, model.kernel, 
                                                 model._padding, dx = model.dx, 
                                                 dy = model.dy, **kwargs)
        self.mask = mask
        self.sigma = model.sigma
        self.rho = kwargs.get('rho', 1E-2)
        self.fftw = WrapFFTW(model._padshape)
       
    def _makeLinearOperators(self):
        """
        Constructs the LinearOperators and solver instances from pykrylov
        for solving the linear least squares problem.
        """
        M = lambda g: self.kernel.applyM(g).real.ravel()
        Mt = lambda g: self.kernel.applyMt(g).real.ravel()
        self.M = MyLinearOperator((self.N, self.N_pad),
                                  matvec = M, rmatvec = Mt)
        self.A = MyLinearOperator((self.N_pad, self.N_pad),
                                  matvec = self._g_kernel_apply)
        self._unPickleable = ['M', 'A']

    def _g_kernel_apply(self, g):
        M, sigma, rho = self.M, self.sigma, self.rho
        return M.T.dot(M.dot(g))/sigma**2 + rho*g 

    def computeResiduals(self, flux, gfield):
        return flux - self.M.dot(gfield.ravel())
 
    def computeNLL(self, flux, gfield):
        res = self.computeResiduals(flux, gfield)
        nll = 0.5*((res**2).sum()/self.sigma**2 +
                   self.N * np.log(2*np.pi*self.sigma**2))
        return nll
        
    def _initialize_lambda(self, g0, Mtphi, h0):
        """
        Initialize lambda so that if we start at the right answer ADMM stays
        there.
        You can find this result by taking the g-update equation
        and solving for lambda.
        """
        #return (self.M.T.dot(self.M.dot(g0)) - Mtphi)/self.sigma**2
        return (self.M.T.dot(self.M.dot(g0)) -Mtphi)/self.sigma**2+ self.rho*(g0-h0)

    def _update_g(self, h, lamb, **kwargs):
        """
        Evaluate the g-update proximal map for ADMM.
        input:
            h : g-like array 
            lamb : lagrange multiplier enforcing D.dot(g) = z
                    (same shape as z)
        kwargs are passed to spl.minres. tol and maxiter control how hard it
        tries
        output:
            updated g : (N_pad)-shaped
        """
        self._oldg = self._oldg if self._oldg is not None else np.zeros(self.N_pad)
        oldAg = self.A.dot(self._oldg)
        self._c = self._Mtphi/self.sigma**2 + lamb + self.rho * h - oldAg
        maxiter = kwargs.get('maxiter', 200)
        tol = kwargs.get('tol', 1E-12)
        self._gminsol = spl.minres(self.A, self._c, maxiter = maxiter, tol = tol)
        self._newg = self._gminsol[0] + self._oldg
        self._oldg = self._newg.copy()
        return self._newg

    def _update_h(self, h0):
        """
        Evaluate the z-update proximal map for ADMM.
        input:
            h0 : current h-value
        output:
            h0 projected onto 
        """
        h_k = self.fftw.fft(h0.reshape(self.Ly_pad,-1))
        return self.fftw.ifft(self.mask * h_k).real.ravel()

    def solve(self, flux, g0 = None, iprint = 0, itnlim = 200, 
              eps_abs = 1E-6, eps_rel = 1E-6, 
              tau_inc = 2, tau_dec = 2, res_ratio = 10, **kwargs):
        """ 
            Use ADMM to solve the TV regularized optimization problem.
            See Foundations and Trends in Machine Learning Vol. 3, No. 1 (2010)
            1-122

            Also implemented: fast ADMM from <citation needed> 
        """
        self._oldg, N_pad = None, self.N_pad #Clear out warm start
        g0 = g0 if g0 is not None else np.random.randn(N_pad)/np.sqrt(N_pad)
        self._Mtphi = self.M.T.dot(flux.ravel())
        
        #Initialize parameters
        g1 = g0.copy()
        h0 = g0.copy()#self._update_h(g1)
        lamb0 = np.zeros(N_pad)#
        #lamb0 = self._initialize_lambda(g1, self._Mtphi, h0)
        h1 = h0.copy()
        lamb1 = lamb0.copy()
        r, s = np.zeros_like(lamb0), np.zeros_like(g1)
        self._r, self._s = r, s

        if iprint:
            print("Initial NLL = {}".format(self.computeNLL(flux, g0)))

        for i in range(itnlim):
            g1[:] = self._update_g(h0, lamb0, **kwargs)
            h1[:] = self._update_h(g1)
          
            r[:] = h1 - g1#Primal residual
            s[:] = - self.rho * (h1 - h0)#Dual residual
            lamb1[:] = lamb0 + self.rho * r
               
            h0[:], lamb0[:] = h1.copy(), lamb1.copy()
            self._g = g1.copy()
            
            eps_primal = (np.sqrt(N_pad)*eps_abs + 
                          eps_rel*max(np.sum(g1**2), np.sum(h1**2))) 
            eps_dual = (np.sqrt(N_pad)*eps_abs +
                        eps_rel*np.sum(lamb1**2))
    
            if r.dot(r) > res_ratio * s.dot(s):
                self.rho *= tau_inc
            elif s.dot(s) > res_ratio * r.dot(r):
                self.rho /= tau_dec

            if iprint > 1:
                pstr = ("\tItn {:1}: NLL = {:e}, r = {:.3e}, "+
                        " s = {:.3e}, eps_p = {:.2e}, eps_d = {:.2e}")
                print(pstr.format(i, self.computeNLL(flux, g1), 
                                  r.dot(r), s.dot(s), eps_primal, eps_dual)) 
            
            if r.dot(r) <= eps_primal and s.dot(s) <= eps_dual:
                if iprint:
                    print("Convergence criterion satisfied")
                break

            self._lamb = lamb1
            self._h0 = h0
        if iprint:
            print("Final NLL = {}".format(self.computeNLL(flux, g1)))
        return g1



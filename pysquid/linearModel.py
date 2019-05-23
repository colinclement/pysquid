"""
linearModel.py

    author: Colin Clement
    email: colin.clement@gmail.com
    date: 2015-9-3

This script contains a class for applying M and M.T matrix-vector products
where M is the convolution of the Biot-Savart law acting on a current-loop
field g.

"""


from __future__ import division
import numpy as np
import numexpr as nu
import scipy as sp
import scipy.sparse.linalg as spl
from scipy.sparse import vstack
from scipy.optimize import minimize
from copy import copy
from itertools import product

# from pykrylov.lls import LSQRFramework, LSMRFramework
# from pykrylov.linop import * #LinearOperator, BlockLinearOperator, null_log
# import logging #Quiet pykrylov.linop logger
# null_log.setLevel(logging.WARNING)

# TODO: Replace use of pykrylov with scipy or remove

from pysquid.kernels.magpsf import *
from pysquid.util.helpers import makeD2_operators, makeD_operators
from pysquid.component import ModelComponent
from pysquid.util.linearOperator import MyLinearOperator


class LinearModel(ModelComponent):
    """
    LinearModel is a class which solves matrix-free linear least-square
    problems.    
    """

    def __init__(self, shape, kernel=None, padding=None, **kwargs):
        """
        input:
            shape : (ly, lx) tuple of (ints) shape of flux field
            kernel : instance of Kernel class for construction M matrix
                        Default is Gaussian Kernel.
            padding : (py, px)
            kwargs:
                mu_reg : float imposing the regularization
        """
        super(LinearModel, self).__init__(shape, padding, **kwargs)

        self.nargin = self.Lx_pad * self.Ly_pad
        self.nargout = self.Lx_pad * self.Ly_pad

        self.sigma = 1.0  # set later
        self.mu_reg = kwargs.get("mu_reg", 0.5)

        self.kernel = kernel if kernel is not None else GaussianKernel(shape, **kwargs)

        self._makeLinearOperators()

    def __setstate__(self, d):
        self.__dict__ = d
        self._makeLinearOperators()

    @property
    def mu(self):
        """
        This is the parameter in front of the g-field regularization prior.
        Dividing by sigma^2 means the solution mean is constant as you change
        sigma
        """
        return self.mu_reg / self.sigma ** 2

    def _makeLinearOperators(self):
        """
        Constructs the LinearOperators and solver instances from pykrylov
        for solving the linear least squares problem.
        """
        pass

    def computeResiduals(self, flux, gfield):
        """
        Non-Gaussian regularization will only calculate the data
        part of the log-likelihood
        """
        pass

    def updateParams(self, name, value):
        """
        Updates module parameters sigma and mu_reg. 
        input: 
            name : string ('sigma' or 'mu_reg')
            value : np array of one element
        """
        if name == "sigma":
            self.sigma = value[0]
        if name == "mu_reg":
            self.mu_reg = value[0]

    def solve(self, flux, **kwargs):
        """
        Find the minimum of negative-log-Likelihood
        """
        pass


class LinearModelTV_ADMM(LinearModel):
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

    def __init__(self, model, **kwargs):
        """
        input:
            model: Instance of FluxModel

            kwargs:
                p : float (default 0.5) for TV regularization power
                mu_reg : float imposing the regularization
                order : Order of derivative operatore in TV regularizer
        """
        # Could now implement p for any power (might help down the line...)
        self._p = kwargs.get("p", 0.5)
        self.rho = kwargs.get("rho", 1e-2)
        super(LinearModelTV_ADMM, self).__init__(
            model._shape, model.kernel, model._padding, rxy=model.rxy, **kwargs
        )
        # self.sigma = model.sigma
        self.lamb = np.random.randn(self.N_pad * 2) / (2 * self.N_pad)

        self.Dh, self.Dv = makeD2_operators((self.Ly_pad, self.Lx_pad), dx=self.rxy)
        self.D = vstack([self.Dh, self.Dv])

    def _makeLinearOperators(self):
        """
        Constructs the LinearOperators and solver instances from pykrylov
        for solving the linear least squares problem.
        """
        M = lambda g: self.kernel.applyM(g).real.ravel()
        Mt = lambda g: self.kernel.applyMt(g).real.ravel()
        self.M = MyLinearOperator((self.N, self.N_pad), matvec=M, rmatvec=Mt)
        self.A = MyLinearOperator((self.N_pad, self.N_pad), matvec=self._g_kernel_apply)
        self._unPickleable = ["M", "A"]

    def computeResiduals(self, flux, gfield):
        return flux - self.M.dot(gfield.ravel())

    def computeNLL(self, flux, gfield, g_ext=None):
        g_ext = g_ext if g_ext is not None else np.zeros_like(gfield)
        res = self.computeResiduals(flux, gfield)
        nll = 0.5 * (
            ((res ** 2).sum() + self.mu * self.TV(gfield + g_ext).sum())
            / self.sigma ** 2
            + self.N * np.log(2 * np.pi * self.sigma ** 2)
        )
        return nll

    @property
    def mu(self):
        """
        We defined the regularization parameter to be 2 mu sigma^2
        to keep things around order 1.
        """
        return 2 * self.mu_reg * self.sigma ** 2

    def TV(self, g):
        """
        Total variation elements. NOTE: may want to JIT this
        in the future 
        """
        p = self._p
        dh = self.Dh.dot(g.ravel())
        dv = self.Dv.dot(g.ravel())
        if p == 0.5:
            return nu.evaluate("sqrt(dh*dh+dv*dv)")
        else:
            return nu.evaluate("(dh*dh+dv*dv)**p")

    def _g_kernel_apply(self, g):
        """
        Computes (M^TM + rho*D^TD)g
        """
        M, D, flatg = self.M, self.D, g.ravel()
        return M.T.dot(M.dot(flatg)) + self.rho * D.T.dot(D.dot(flatg))

    def _initialize_lambda(self, g0, Mtphi):
        """
        Initialize lambda so that if we start at the right answer ADMM stays
        there.
        You can find this result by taking the g-update equation
        and solving for lambda.
        """
        self._RHS = Mtphi - self.M.T.dot(self.M.dot(g0))
        self._minlamb = spl.lsqr(self.D.T, self._RHS, damp=1e-5, atol=1e-6, btol=1e-6)
        return self._minlamb[0]

    def _update_g(self, z, lamb, D2g_ext, **kwargs):
        """
        Evaluate the g-update proximal map for ADMM.
        input:
            z : current z-value (2*N_pad)-shaped
            lamb : lagrange multiplier enforcing D.dot(g) = z
                    (same shape as z)
            D2g_ext : D.T.dot(D.dot(g_ext)) the fourth-derivative of the
                      loop field
        kwargs are passed to spl.minres. tol and maxiter control how hard it
        tries
        output:
            updated g : (N_pad)-shaped
        """
        self._oldg = self._oldg if self._oldg is not None else np.zeros(self.N_pad)
        self._oldAg = self.A.dot(self._oldg)
        self._c = (
            self._Mtphi - self.D.T.dot(lamb - self.rho * z) - self.rho * D2g_ext
        ) - self._oldAg
        maxiter = kwargs.get("maxiter", 200)
        tol = kwargs.get("tol", 1e-6)
        self._gminsol = spl.minres(self.A, self._c, maxiter=maxiter, tol=tol)
        self._newg = self._gminsol[0] + self._oldg
        self._oldg = self._newg.copy()
        return self._newg

    def _update_z(self, z0, lamb, Dg, zsteps=40):
        """
        Evaluate the z-update proximal map for ADMM.
        input:
            z0 : current z-value (2*N_pad)-shaped
            lamb : lagrange multiplier same shape as z0
            Dg : D.dot(g) second derivatives of g
            zsteps : how many LBFGS steps to allow
        output:
            updated z : (2*N_pad)-shaped
        """
        self._zminsol = minimize(
            self._T_dTdz,
            z0,
            args=(lamb, Dg),
            method="l-bfgs-b",
            jac=True,
            options={"maxiter": zsteps},
        )
        return self._zminsol["x"]

    def _T_dTdz(self, z, lamb, Dg):
        """
        Evaluate the z-update proximal map and its derivatives.
        """
        N_pad, mu, p = self.N_pad, self.mu, self._p
        x, y, Dhg, Dvg = z[:N_pad], z[N_pad:], Dg[:N_pad], Dg[N_pad:]
        zsq = nu.evaluate("x*x + y*y")
        vari = nu.evaluate("(x*x + y*y)**p")
        d_vari = vari / zsq
        x_con, y_con = Dhg - x, Dvg - y
        T = (
            mu * vari.sum()
            - lamb.dot(z)
            + (self.rho / 2.0) * (x_con.dot(x_con) + y_con.dot(y_con))
        )
        dTdx = (2 * p * mu) * x * d_vari - lamb[:N_pad] - self.rho * x_con
        dTdy = (2 * p * mu) * y * d_vari - lamb[N_pad:] - self.rho * y_con
        dTdz = np.r_[dTdx, dTdy]
        return T, dTdz

    def solve(
        self,
        flux,
        g0=None,
        extmodel=None,
        iprint=0,
        itnlim=200,
        eps_abs=1e-6,
        eps_rel=1e-6,
        zsteps=20,
        eta=0.999,
        tau_inc=2,
        tau_dec=2,
        res_ratio=10,
        **kwargs
    ):
        """ 
            Use ADMM to solve the TV regularized optimization problem.
            See Foundations and Trends in Machine Learning Vol. 3, No. 1 (2010)
            1-122

            Also implemented: fast ADMM from <citation needed> 
        """
        self._oldg, N_pad = None, self.N_pad  # Clear out warm start
        g0 = g0 if g0 is not None else np.random.randn(N_pad) / np.sqrt(N_pad)
        self._Mtphi = self.M.T.dot(flux.ravel())
        if extmodel is not None:
            g_ext = extmodel.ext_g.ravel()
            Dg_ext = self.D.dot(g_ext)
            D2g_ext = self.D.T.dot(Dg_ext)
        else:
            g_ext, Dg_ext = np.zeros(N_pad), np.zeros(2 * N_pad)
            D2g_ext = np.zeros(N_pad)

        # Initialize parameters
        g1 = g0.copy()
        z0 = self.D.dot(g0 + g_ext)
        lamb0 = self._initialize_lambda(g1, self._Mtphi)
        alpha0, alpha1, ck0, ck1 = 1.0, 1.0, np.inf, 1.0
        # Make copies for fast ADMM
        z1_hat, z1 = z0.copy(), z0.copy()
        lamb_hat, lamb1 = lamb0.copy(), lamb0.copy()
        r, s = np.zeros_like(lamb0), np.zeros_like(g1)
        self._r, self._s = r, s

        if iprint:
            print("Initial NLL = {}".format(self.computeNLL(flux, g0, g_ext)))

        for i in range(itnlim):
            g1[:] = self._update_g(z1_hat, lamb_hat, D2g_ext, **kwargs)
            Dg = self.D.dot(g1)
            z1[:] = self._update_z(z1_hat, lamb_hat, Dg + Dg_ext, zsteps)

            r[:] = Dg + Dg_ext - z1  # Primal residual
            s[:] = self.rho * self.D.T.dot(z1 - z1_hat)  # Dual residual
            lamb1[:] = lamb_hat + self.rho * r
            ck1 = np.sum((lamb1 - lamb_hat) ** 2) / self.rho + self.rho * np.sum(
                (z1 - z1_hat) ** 2
            )
            if ck1 < eta * ck0:
                alpha1 = (1.0 + np.sqrt(1 + 4 * alpha0 ** 2)) / 2.0
                mom = (alpha0 - 1) / alpha1
                z1_hat = z1 + mom * (z1 - z0)
                lamb_hat = lamb1 + mom * (lamb1 - lamb0)
            else:
                if iprint > 1:
                    print("\t\tRestarted {} > {} * {}".format(ck1, eta, ck0))
                alpha1, z1_hat[:], lamb_hat[:] = 1.0, z0.copy(), lamb0.copy()
                ck1 = ck0 / eta

            z0[:], lamb0[:] = z1.copy(), lamb1.copy()
            ck0, alpha0 = ck1, alpha1
            self._g = g1.copy()

            eps_primal = np.sqrt(2 * N_pad) * eps_abs + eps_rel * max(
                np.sum(Dg ** 2), np.sum(z1 ** 2), np.sum(Dg_ext ** 2)
            )
            eps_dual = np.sqrt(N_pad) * eps_abs + eps_rel * np.sum(
                (self.D.T.dot(lamb1)) ** 2
            )
            if r.dot(r) <= eps_primal and s.dot(s) <= eps_dual:
                if iprint:
                    print("Convergence criterion satisfied")
                break

            if r.dot(r) > res_ratio * s.dot(s):
                self.rho *= tau_inc
            elif s.dot(s) > res_ratio * r.dot(r):
                self.rho /= tau_dec

            if iprint > 1:
                pstr = (
                    "\tItn {:1}: NLL = {:e}, r = {:.3e}, "
                    + " s = {:.3e}, eps_p = {:.3e}, eps_d = {:.3e}"
                )
                print(
                    pstr.format(
                        i,
                        self.computeNLL(flux, g1, g_ext),
                        r.dot(r),
                        s.dot(s),
                        eps_primal,
                        eps_dual,
                    )
                )
        if iprint:
            print("Final NLL = {}".format(self.computeNLL(flux, g1, g_ext)))
        return g1


class LinearModelOrthProj(LinearModel):
    """
    LinearModelOrthProj is an object which contains methods for computing
    matrix-vector and matrix-transpose-vector products of 
    Biot-Savart convolutions using an FFT. These functions are passed to
    LinearOperator instances used in a Krylov-subspace iteration least-square
    solver in the pykrylov module.
    """

    def __init__(self, shape, kernel=None, padding=None, **kwargs):
        """
        input:
            shape : (ly, lx) tuple of (ints) shape of flux field
            kernel : instance of Kernel class for construction M matrix
                        Default is Gaussian Kernel.
            padding : (py, px)
            kwargs:
                mu_reg : float imposing the regularization
                x_ord, y_ord : int arrays of x/y-ordered polynomials
                        for the orthogonal projection regularizer
        """
        self.x_ord = kwargs.get("x_ord", np.arange(12))
        self.y_ord = kwargs.get("y_ord", np.arange(20))
        super(LinearModelOrthProj, self).__init__(shape, kernel, padding, **kwargs)

    def Gammadot(self, g):
        """
        Compute product P.dot(g) where P = 1 - W.dot(W.T)
        where W is the basis of the null space chosen by x_ord/y_ord
        """
        return np.sqrt(self.mu) * (g - self._W.dot(self._W.T.dot(g)))

    def _makeLinearOperators(self):
        """
        Constructs the LinearOperators and solver instances from pykrylov
        for solving the linear least squares problem.
        """
        self._W = makeOrthogonalBasis(self.Lx_pad, self.Ly_pad, self.x_ord, self.y_ord)
        M = lambda g: self.kernel.applyM(g).real.ravel() / self.sigma
        Mt = lambda phi: self.kernel.applyMt(phi).real.ravel() / self.sigma
        # self.M = LinearOperator(nargin = self.N_pad,
        #                        nargout = self.N_pad,
        #                        matvec = M,
        #                        matvec_transp = Mt)
        # self.muGamma = LinearOperator(nargin = self.N_pad,
        #                              nargout = self.N_pad,
        #                              matvec = self.Gammadot,
        #                              symmetric = True)
        self.A = BlockLinearOperator([[self.M], [self.muGamma]])
        # self.lsqr = LSMRFramework(self.A)
        self._unPickleable = ["M", "muGamma", "A", "lsqr"]

    def computeResiduals(self, flux, gfield, ext_g=None):
        gamma_ext_g = np.zeros(self.N_pad) if ext_g is None else self.muGamma * ext_g
        regularized_g = np.r_[flux.ravel() / self.sigma, -gamma_ext_g]
        return regularized_g - self.A * gfield.ravel()

    def solve(
        self, flux, oldx=None, ext_g=None, atol=1e-6, btol=1e-6, etol=0.0, **kwargs
    ):
        """
        Solve the linear least squares problem
        min_g 1/2 || flux - M.dot(g) ||^2 + mu * || Gamma.dot(g+ext_g) ||^2
        input:
            flux : float array of shape (ly, lx) where ly = Ly/ay,
                    lx - Lx/ax
        output:
            solution array of g, flattened and with length (Ly*Lx)
        
        kwargs:
            kwargs are passed to LSQRFramework.solve, e.g.
            itnlim : (int) iteration limit (default is 3*nargin)
            atol, btol, etol : (float) convergence criterion
        """
        self._oldx = oldx if oldx is not None else copy(self.lsqr.x)
        gamma_ext_g = (
            np.zeros(self.N_pad) if ext_g is None else -1 * self.muGamma * ext_g
        )
        self._oldAx = self.A * self._oldx if self._oldx is not None else 0.0
        fluxpad = np.r_[flux.ravel() / self.sigma, gamma_ext_g] - self._oldAx
        self.lsqr._output = self.lsqr.solve(
            fluxpad, etol=etol, atol=atol, btol=btol, **kwargs
        )
        # Because there are bugs in lsqr
        self.lsqr.x = self.lsqr._output[0]
        self.lsqr.istop = self.lsqr._output[1]
        self.lsqr.itn = self.lsqr._output[2]

        self.solution = self.lsqr.x + (self._oldx if self._oldx is not None else 0.0)
        self.residuals = flux - self.M * self.solution
        self.residuals_with_reg = fluxpad - self.A * self.solution
        return self.solution


############### Regularization projection ###################


def makeOrthogonalBasis(Lx, Ly, x_orders=np.arange(4), y_orders=np.arange(4)):
    """
    Construct operater with a nullspace containing orthogonal
    polynomials of specified orders in the x and y directions.
    input:
        Lx, Ly : (int)'s specifying sample shape
        x_orders : array of ints of polynomial order in x-dir
        y_orders : same as x_orders but for y-direction
    output:
        W : (float) array of shape (Lx*Ly, x_orders+y_orders) 
            which is an orthonormal basis for polynomials
    NOTE: only x_order OR y_order should have a zero, otherwise
    there will be degenerate eigenvalues which will cause 
    numerical problems
    """
    crossterms = np.array(list(product(x_orders, y_orders)))
    rc = np.zeros((Ly * Lx, len(crossterms)))
    for i in range(Ly * Lx):
        x, y = ((i % Lx + 1) - Lx / 2) / Lx, ((i // Lx + 1) - Ly / 2) / Ly
        rc[i, :] = np.prod(np.array([x, y]) ** crossterms, 1)
    W, R = sp.linalg.qr(rc, mode="economic")
    return W

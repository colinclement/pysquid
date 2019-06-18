"""
deconvolve.py

author: Colin Clement
date: 2016-12-06

Classes which perform deconvolution. Implemented priors include total
variation in current (second derivative in g) and finite support
(constant g) constraints.
"""

from scipy.optimize import minimize
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as ssl
from scipy.sparse import vstack
from functools import partial
import numexpr as nu
import numpy as np

from pysquid.util.linear_operators import MyLinearOperator, makeD2, finite_support
from pysquid.opt.admm import ADMM


LINEAR_SOLVERS = ["bicg", "bicgstab", "cg", "cgs", "gmres", "lgmres", "minres", "qmr"]
SOLVER_MSG = "Solver must be one of the following\n" + ", ".join(LINEAR_SOLVERS)


class LinearDeconvolver:
    r"""
    An object which solves a deconvolution problem subject to a Gaussian 
    (quadratic) prior, also known as Tikhonov regularized deconvolution.
    Specifically, for a given flux image \phi, kernel M, regularization
    operator \Gamma, and regularization strength \sigma^2, it solves

        min_g 1/2||Mg - \phi||^2 + \sigma^2 ||\Gamma(g + g_ext)||^2.

    In other words, it solves the linear equation

        (M^T M + 2 \sigma^2\Gamma^T \Gamma) g = M^T \phi -
            \sigma^2\Gamma^T\Gamma c.

    If support_mask is provided, then there are effectively fewer parameters
    as the regions where current is set to zero (where support_mask is 1) are
    a single parameter. In this case we define an operator F, which takes us
    from the free parameter \tilde{g} to the full image plane in g as
    follows: g = F \tilde{g}. In this case M -> MF and \Gamma -> \Gamma F

    Parameters
    ----------
    kernel: pysquid.Kernel object which can compute
        M.dot and M.T.dot matrix-vector products
        via the methods `kernel.applyM` and `kernel.applyMt`
    sigma : float
        estimate of noise in the data or otherwise chosen regularization
        strength
    g_ext : array_like, optional
        portion of external current loop model inside field of view,
        provided to not penalize total variation due to subtracted global loop
    gamma : MyLinearOperator object
        the regularization operator \Gamma, by default it is the laplacian
    support_mask : array_like
        mask with binary entries, where 0 indicates free current can flow,
        and 1 indicates contiguous regions which must have constant g, and
        thus zero internal current
    """

    def __init__(self, kernel, sigma, g_ext=None, gamma=None, support_mask=None):
        """
        Initialize LinearDeconvolver
        """
        self.kernel = kernel
        assert np.isscalar(sigma), "sigma must be a single scalar number"
        self.sigma = sigma

        if gamma is None:
            D2h, D2v = makeD2(self.kernel._padshape, dx=self.kernel.rxy)
            self.G = D2h + D2v
        else:
            msg = "gamma must have `dot` and methods"
            assert hasattr(gamma, "dot"), msg
            msg = "gamma must have `T` and methods"
            assert hasattr(gamma, "T"), msg
            self.G = gamma

        if g_ext is not None:
            self._gg_g_ext = sigma ** 2 * self.G.dot(g_ext)

        if support_mask is None:
            self._F = None

            def M(g):
                return self.kernel.applyM(g).real.ravel()

            def Mt(g):
                return self.kernel.applyMt(g).real.ravel()

        else:
            self._F = finite_support(support_mask, self.kernel._padshape)

            def M(g):
                return self.kernel.applyM(self._F.dot(g)).real.ravel()

            def Mt(p):
                return self._F.T.dot(self.kernel.applyMt(p).real.ravel())

            self.G = self.G.dot(self._F)

        # G.T after G -> GF in case shape changes
        if g_ext is not None:
            self._gg_g_ext = self.G.T.dot(self._gg_g_ext)

        n = self.kernel.N_pad if self._F is None else self._F.shape[1]
        self.M = MyLinearOperator((self.kernel.N, n), matvec=M, rmatvec=Mt)

    def _apply_regularized_kernel(self, x, sigma):
        return self.M.T.dot(self.M.dot(x)) + 2 * sigma ** 2 * self.G.T.dot(
            self.G.dot(x)
        )

    def _regularized_operator(self, sigma):
        n = self.kernel.N_pad if self._F is None else self._F.shape[1]
        return MyLinearOperator(
            (n, n), matvec=partial(self._apply_regularized_kernel, sigma=sigma)
        )

    def deconvolve(self, phi, **kwargs):
        """
        Solve the regularized deconvolution problem with specific
        flux image \phi and regularization strength sigma

        Parameters
        ----------
        phi : ndarray
            image of size N total elements, matching initialized kernel
        (optional kwargs)
        solver : str
            specifying one of the linear solvers from the `scipy.sparse.linalg`
            package
        iprint : int
            if larger than 0 will print messages

        Returns
        -------
        gsol : ndarray of size N_pad
            solution to the regularized deconvolution problem
        """
        solver_str = kwargs.pop("solver", "cg")
        iprint = kwargs.pop("iprint", 0)
        assert solver_str in LINEAR_SOLVERS, SOLVER_MSG
        solver = getattr(ssl, solver_str)

        A = self._regularized_operator(self.sigma)
        b = self.M.T.dot(phi.ravel()) - getattr(self, "_gg_g_ext", np.zeros_like(b))

        xsol, info = solver(A, b, **kwargs)
        if iprint:
            print("Convergence info {}".format(info))

        if self._F is not None:
            return self._F.dot(xsol)
        return xsol


class Deconvolver(ADMM):
    """
    Base class which specifies the first function in the ADMM
    problem to be a sum of quadratics of linear functions.
    In particular, this sets f(x) in standard
    ADMM problem to be 1/2||Mx-phi||^2 + nu/2||x-x^hat||^2,
    which is a deconvolution problem when M is a blurring matrix.

    The first term is the 'fidelity' term for reproducing data
    the second term is for finding the proximal operator
    associated with our deconvolution problem which is useful for
    solving constrained optimzation problems.

    This class must subclassed to be useful. The following must be
    defined in a subclass:
        g
        z_update
    (See class ADMM for argument signatures of these functions)

    Parameters
    ----------
    kernel: pysquid.Kernel object which can compute
        M.dot and M.T.dot matrix-vector products
        via the methods `kernel.applyM` and `kernel.applyMt`
    support_mask : array_like
        mask with binary entries, where 0 indicates free current can flow,
        and 1 indicates contiguous regions which must have constant g, and
        thus zero internal current
    """

    def __init__(self, kernel, support_mask=None, **kwargs):
        """
        input:
            kernel: pysquid.Kernel object which can compute
                M.dot and M.T.dot matrix-vector products

        kwargs:
            (Useful for calculating proximal operator of deconvolution,
            see TVFiniteSupportDeconvolve)
            nu      : float, strength of proximal term
            xhat    : ndarray of shape self.n, proximal goal
        """
        self.kernel = kernel
        if support_mask is None:
            self._F = None

            def M(g):
                return self.kernel.applyM(g).real.ravel()

            def Mt(g):
                return self.kernel.applyMt(g).real.ravel()

        else:
            self._F = finite_support(support_mask, self.kernel._padshape)

            def M(g):
                return self.kernel.applyM(self._F.dot(g)).real.ravel()

            def Mt(p):
                return self._F.T.dot(self.kernel.applyMt(p).real.ravel())

        # set shape of matrices and arrays
        n = kernel.N_pad if support_mask is None else self._F.shape[1]
        m = 2 * kernel.N_pad  # x and y derivatives
        p = m

        # Setup M matrix for Mg = phi
        self.M = MyLinearOperator((self.kernel.N, n), matvec=M, rmatvec=Mt)

        super(Deconvolver, self).__init__(n, m, p)

        self._oldx = None  # warm start for linear solver
        self._newx = None

        # proximal operator parameters
        self.nu = kwargs.get("nu", 0.0)
        self.xhat = kwargs.get("xhat", np.zeros(n))
        assert len(self.xhat) == n, "xhat must be length {}".format(n)

    def f(self, x, phi, **kwargs):
        """
        Fidelity term and proximity term in generalized deconvolution
        problem.

        input:
            x   : ndarray of shape self.n
            phi : ndarray of shape self.kernel.N
        returns:
            f(x): float, fidelity + proximal terms
        """
        assert len(phi) == self.kernel.N, "phi is incorrect size"
        res = self.M.dot(x) - phi
        prox = x - self.xhat
        return res.dot(res) / 2 + self.nu * prox.dot(prox) / 2

    def _apply_x_update_kernel(self, x, rho):
        """
        Function which returns the matrix-vector product of the kernel
        solved in the x_update function.

        input:
            x           : ndarray of shape self.n
            rho         : float, strength of augmented lagrangian term
        returns:
            K.dot(x)    : result of kernel operating on x
        """
        M, A = self.M, self.A
        return M.T.dot(M.dot(x)) + rho * A.T.dot(A.dot(x)) + self.nu * x

    def _get_x_op(self, rho):
        """
        Create linear operator object which applies the function
        self._apply_x_update_kernel
        input:
            rho : floatm strength of augmented lagrangian term
        returns:
            Op  : Linear operator which applies kernel for x_update
        """
        n = self.kernel.N_pad if self._F is None else self._F.shape[1]

        def apply_kernel(x):
            return self._apply_x_update_kernel(x, rho)

        return MyLinearOperator((n, n), matvec=apply_kernel)

    def start_lagrange_mult(self, x0, z0, rho, phi, **kwargs):
        """
        Calculates the initial lagrange multiplier which ensures that
        ADMM is stable if started at the correct x0, z0. The result of
        this function can be calculated by taking the x_update solution
        and solving for y.

        input:
            x0  : ndarray of shape self.n, initial x point
            z0  : ndarray of shape self.m, initial z point
            rho : float, weighting of augmented part of lagrangian
            phi : float array of shape self.
        returns:
            y0  : ndarray of shape self.p, initial y
        """
        Op = self._get_x_op(rho)
        A, B, c = self.A, self.B, self.c
        self._y0rhs = (
            -Op.dot(x0)
            + self.M.T.dot(phi)
            + self.nu * self.xhat
            - rho * A.T.dot(B.dot(z0) - c)
        )
        maxiter = kwargs.get("y0_maxiter", None)
        atol = kwargs.get("atol", 1e-5)
        btol = kwargs.get("atol", 1e-5)
        self._y0minsol = ssl.lsqr(
            self.A.T, self._y0rhs, iter_lim=maxiter, atol=atol, btol=btol, damp=1e-5
        )
        return self._y0minsol[0]

    def x_update(self, z, y, rho, x0=None, phi=None, **kwargs):
        """
        Calculates x_{k+1} = argmin_x 1/2||Mx-phi||^2 + nu/2||x-x^hat||^2
                                      + y^T Ax + rho/2||Ax+Bz-c||^2

        by solving (M^TM+rho*A^T A+nu)x = M^Tphi+nu*x^hat-A^T(y+rho*(Bz-c))

        input:
            z   : ndarray of shape self.m current z-value
            y   : ndarray of shape self.m current y-value
            rho : float, strength of augmented lagrangian term
            phi : ndarray of shape self.kernel.N, data to be fit
            (optional)
            x0  : ndarray of shape self.n optional starting point
        returns:
            x   : ndarray of shape self.n, updated x value

        kwargs:
            solver: string of iterative linear solver method. Choose from
                list LINEAR_SOLVERS or solvers in scipy.sparse.linalg.
                default is 'minres'
            maxiter: maximum iterations for linear solver, default 250
            tol: tolerance for convergence criterion of linear solver,
                default is 1E-6. See docs for solver for definitions

        #TODO: writeup warm start of linear equation solvers
        """
        A, B, c = self.A, self.B, self.c
        self._oldx = self._oldx if self._oldx is not None else np.zeros(self.n)
        maxiter = kwargs.get("maxiter", 250)
        tol = kwargs.get("tol", 1e-6)
        solver_str = kwargs.get("solver", "cg")

        assert phi is not None, "Must provide phi to deconvolve!"
        assert solver_str in LINEAR_SOLVERS, SOLVER_MSG
        solver = getattr(ssl, solver_str)

        Op = self._get_x_op(rho)
        self._oldOpx = Op.dot(self._oldx)
        self._rhs = (
            self.M.T.dot(phi) + self.nu * self.xhat - A.T.dot(y + rho * (B.dot(z) - c))
        )
        self._rhs -= self._oldOpx  # warm start

        self._xminsol = solver(Op, self._rhs, maxiter=maxiter, tol=tol)

        self._newx = self._xminsol[0] + self._oldx
        self._oldx = self._newx.copy()
        return self._newx

    def callstart(self, x0=None, **kwargs):
        """
        Ensures that ADMM is started with a warm start for x_update
        that is consistent with the initial guess
        input:
            x0  : ndarray of shape self.n, initial x guess
        """
        self._oldx = x0  # Reset warm start


class TVDeconvolver(Deconvolver):
    """
    Performs deconvolution of flux image with a
    total variation prior on currents by minimizing the function
        1/2||Mx-phi||^2 + nu/2||x-x^hat||^2 + sigma^2 TV(x)

    Parameters
    ----------
    kernel: pysquid.Kernel object which can compute
        M.dot and M.T.dot matrix-vector products
        via the methods `kernel.applyM` and `kernel.applyMt`

    sigma : float
        strength of the TV prior, and basically an estimate of the noise of phi
    g_ext : array_like, optional
        portion of external current loop model inside field of view,
        provided to not penalize total variation due to subtracted global loop
    support_mask : array_like, optional
        mask with binary entries, where 0 indicates free current can flow,
        and 1 indicates contiguous regions which must have constant g, and
        thus zero internal current

    kwargs
    ------
    see `Deconvolver.__init__`

    Usage
    -----
        deconvolver = TVDeconvolver(kernel, sigma, g_ext)
        deconvolver.deconvolve(phi, x0, **kwargs)
    """

    def __init__(self, kernel, sigma, g_ext=None, support_mask=None, **kwargs):
        """ Initialize TVDeconvolver """
        super(TVDeconvolver, self).__init__(kernel, support_mask, **kwargs)
        assert np.isscalar(sigma), "sigma must be a single scalar number"
        self.sigma = sigma

        self.A = vstack(makeD2(self.kernel._padshape, dx=self.kernel.rxy))
        self._set_g_ext(g_ext)

        if self._F is not None:  # NOTE: self._F defined in Deconvolver init
            self.A = self.A.dot(self._F)

        self.B = MyLinearOperator((self.m, self.m), matvec=lambda x: -x)

    def _set_g_ext(self, g_ext=None):
        if g_ext is None:
            self.c = np.zeros(self.p)
        else:  # No penalty for TV of edge made by exterior loop subtraction
            A = vstack(makeD2(self.kernel._padshape, dx=self.kernel.rxy))
            self.c = -A.dot(g_ext.ravel())

    def g(self, z):
        """
        Total variation function, recall that derivatives are imposed in
        the constraint Ax + Bz = c.
        input:
            z   : ndarray of shape self.m
        returns:
            g(z): float, value of total variation of z
        """
        dx, dy = z[: len(z) // 2], z[len(z) // 2 :]
        return self.sigma ** 2 * nu.evaluate("sum(sqrt(dx*dx + dy*dy))")

    def _lagrangian_dz(self, z, x, y, rho):
        """
        Evaluate the augmented lagrangian (up to a constants not depending
        on z) and its derivative with respect to z.
        input:
            z   : ndarray of shape self.m
            x   : ndarray of shape self.n
            y   : ndarray of shape self.p
        returns:
            lagrangian (float), d_lagrangian (m-shaped ndarray)
        """
        r = self.primal_residual(x, z)
        xx, yy = z[: len(z) // 2], z[len(z) // 2 :]
        tv = nu.evaluate("sqrt(xx*xx + yy*yy)")
        lagrangian = self.sigma ** 2 * tv.sum() + r.dot(y) + rho * r.dot(r) / 2
        d_tv = np.concatenate((xx / (tv + 1e-8), yy / (tv + 1e-8)))
        d_lagrangian = self.sigma ** 2 * d_tv + self.B.T.dot(y + rho * r)
        return lagrangian, d_lagrangian

    def z_update(self, x, y, rho, z0, **kwargs):
        """
        Evaluate the z-update proximal map by solving
        z_{k+1} = argmin_z - y^T Bz + rho/2||Ax+Bz-c||^2 + g(z)
        with the scipy LBFGS minimizer.

        input:
            x   : ndarray of shape self.n, current x-value
            y   : ndarray of shape self.p, current y-value
            rho : float, strength of augmented lagrangian term
            z0  : ndarray of shape self.m, initial guess for z
        returns:
            z   : ndarray of shape self.m, updated z-value
        kwargs:
            zsteps  : int, maximum number of attempted LBFGS steps,
                        default is 20
        """
        options = {"maxiter": kwargs.get("zsteps", 20)}
        self._zminsol = minimize(
            self._lagrangian_dz,
            z0,
            args=(x, y, rho),
            method="l-bfgs-b",
            jac=True,
            options=options,
        )
        return self._zminsol["x"]

    def nlnprob(self, phi, g):
        """
        Evaluates the negative log probability of g given phi
        """
        z = self.A.dot(g.ravel())
        return self.cost(g.ravel(), z, f_args=(phi.ravel(),))

    def deconvolve(self, phi, x0=None, **kwargs):
        """
        Perform a deconvolution of data phi with provided kernel.

        Parameters
        ----------
            phi : ndarray of shape self.kernel.N, data to be analyzed
            x0 : ndarray of shape self.n, initial guess for x (or g)
                default is random vector roughly normalized
        Returns
        -------
            x (or g)    : ndarray of shape self.kernel.N_pad

        kwargs
        ------
            maxiter     : int, maximum number of steps for linear solver in
                            x_update, default is 200
            tol         : float, tolerance for linear solver in x_update,
                            default is 1E-6
            solver      : str, specific linear solver for x_update.
                            default is 'minres', more in scipy.sparse.linalg
            zsteps      : float, number of iterations allowed for z_update
                            default is 20
            algorithm   : str, either 'minimize' (default) for standard ADMM
                            or 'minimize_fastrestart' for fast ADMM
            (All other kwargs are passed to the minimizer, either ADMM.minimize
             or ADMM.minimize_fastrestart. See for documentation)

        """
        x0 = x0 if x0 is not None else np.random.randn(self.n) / np.sqrt(self.n)

        f_args = (phi.ravel(),)
        f_kwargs = {}
        f_kwargs["maxiter"] = kwargs.get("maxiter", 200)
        f_kwargs["tol"] = kwargs.get("tol", 1e-6)
        f_kwargs["solver"] = kwargs.get("solver", "minres")

        g_args = ()
        g_kwargs = {}
        g_kwargs["zsteps"] = kwargs.get("zsteps", 20)

        algorithm = kwargs.get("algorithm", "minimize")
        minimizer = getattr(self, algorithm)

        xmin, _, msg = minimizer(x0, f_args, g_args, f_kwargs, g_kwargs, **kwargs)

        if self._F is not None:
            return self._F.dot(xmin)
        return xmin

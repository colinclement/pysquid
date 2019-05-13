"""
deconvolve.py

author: Colin Clement
date: 2016-12-06

Classes which perform deconvolution. Implemented priors include total
variation in current (second derivative in g) and finite support
(constant g) constraints.
"""

from scipy.optimize import minimize
import scipy.sparse.linalg as ssl
from scipy.ndimage import label
from scipy.sparse import vstack
import numexpr as nu
import numpy as np

from pysquid.util.linearOperator import MyLinearOperator
from pysquid.util.helpers import makeD2_operators
from pysquid.opt.admm import ADMM


linear_solvers = ['bicg', 'bicgstab', 'cg', 'cgs', 'gmres', 'lgmres',
                  'minres', 'qmr']
solver_msg = ("Solver must be one of the following\n" +
              ', '.join(linear_solvers))


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

    """

    def __init__(self, kernel, **kwargs):
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
        n = kernel.N_pad
        m = 2 * n  # x and y derivatives
        p = m
        super(Deconvolver, self).__init__(kernel.N_pad, m, p)

        self._oldx = None  # warm start for linear solver
        self._newx = None

        # proximal operator parameters
        self.nu = kwargs.get('nu', 0.)
        self.xhat = kwargs.get('xhat', np.zeros(n))
        assert len(self.xhat) == n, "xhat must be length {}".format(n)

        # Setup M matrix for Mg = phi

        def M(g): return self.kernel.applyM(g).real.ravel()

        def Mt(g): return self.kernel.applyMt(g).real.ravel()
        N, N_pad = self.kernel.N, self.kernel.N_pad
        self.M = MyLinearOperator((N, N_pad), matvec=M, rmatvec=Mt)

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
        return res.dot(res)/2 + self.nu*prox.dot(prox)/2

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
        return M.T.dot(M.dot(x)) + rho * A.T.dot(A.dot(x)) + self.nu*x

    def _get_x_op(self, rho):
        """
        Create linear operator object which applies the function
        self._apply_x_update_kernel
        input:
            rho : floatm strength of augmented lagrangian term
        returns:
            Op  : Linear operator which applies kernel for x_update
        """
        N_pad = self.kernel.N_pad

        def apply_kernel(x): return self._apply_x_update_kernel(x, rho)
        return MyLinearOperator((N_pad, N_pad), matvec=apply_kernel)

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
        self._y0rhs = (- Op.dot(x0) + self.M.T.dot(phi) + self.nu * self.xhat -
                       rho*A.T.dot(B.dot(z0)-c))
        maxiter = kwargs.get('y0_maxiter', None)
        atol = kwargs.get('atol', 1E-5)
        btol = kwargs.get('atol', 1E-5)
        self._y0minsol = ssl.lsqr(self.A.T, self._y0rhs, iter_lim=maxiter,
                                  atol=atol, btol=btol, damp=1E-5)
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
                    list linear_solvers or solvers in scipy.sparse.linalg.
                    default is 'minres'
            maxiter: maximum iterations for linear solver, default 250
            tol: tolerance for convergence criterion of linear solver,
                    default is 1E-6. See docs for solver for definitions

        #TODO: writeup warm start of linear equation solvers
        """
        A, B, c = self.A, self.B, self.c
        self._oldx = self._oldx if self._oldx is not None else np.zeros(self.n)
        maxiter = kwargs.get('maxiter', 250)
        tol = kwargs.get('tol', 1E-6)
        solver_str = kwargs.get('solver', 'cg')

        assert phi is not None, "Must provide phi to deconvolve!"
        assert solver_str in linear_solvers, solver_msg
        solver = getattr(ssl, solver_str)

        Op = self._get_x_op(rho)
        self._oldOpx = Op.dot(self._oldx)
        self._rhs = (self.M.T.dot(phi) + self.nu * self.xhat -
                     A.T.dot(y + rho*(B.dot(z) - c)))
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
        1/2||Mx-phi||^2 + nu/2||x-x^hat||^2 + gamma*TV(x)

    usage:
        deconvolver = TVDeconvolver(kernel, gamma, g_ext)
        deconvolver.deconvolve(phi, x0, **kwargs)
    """

    def __init__(self, kernel, gamma, g_ext=None, **kwargs):
        """
        input:
            kernel  : pysquid.Kernel object which can compute
                        M.dot and M.T.dot matrix-vector products
            gamma   : float, strength of TV prior
            (optional)
            g_ext   : ndarray of shape self.n, g-field of exterior
                        loop, if data has a subtracted flux field
        kwargs:
            see Deconvolver class __init__
        """
        self.gamma = gamma
        super(TVDeconvolver, self).__init__(kernel, **kwargs)

        Ly_pad, Lx_pad = self.kernel._padshape
        dy, dx = 1., self.kernel.rxy
        self.Dh, self.Dv = makeD2_operators((Ly_pad, Lx_pad), dx, dy)

        self.A = vstack([self.Dh, self.Dv])
        self.B = MyLinearOperator((self.m, self.m), matvec=lambda x: -x)
        self.set_g_ext(g_ext)

    def set_g_ext(self, g_ext=None):
        if g_ext is None:
            self.c = np.zeros(self.p)
        else:  # No penalty for TV of edge made by exterior loop subtraction
            self.c = -self.A.dot(g_ext.ravel())

    def g(self, z):
        """
        Total variation function, recall that derivatives are imposed in
        the constraint Ax + Bz = c.
        input:
            z   : ndarray of shape self.m
        returns:
            g(z): float, value of total variation of z
        """
        dx, dy = z[:self.n], z[self.n:]
        return self.gamma * nu.evaluate('sum(sqrt(dx*dx + dy*dy))')

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
        gamma = self.gamma
        xx, yy = z[:self.n], z[self.n:]
        tv = nu.evaluate('sqrt(xx*xx + yy*yy)')
        lagrangian = gamma * tv.sum() + r.dot(y) + rho*r.dot(r)/2
        d_tv = np.concatenate((xx/tv, yy/tv))
        d_lagrangian = gamma * d_tv + self.B.T.dot(y + rho*r)
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
        options = {'maxiter': kwargs.get('zsteps', 20)}
        self._zminsol = minimize(self._lagrangian_dz, z0, args=(x, y, rho,),
                                 method='l-bfgs-b', jac=True,
                                 options=options)
        return self._zminsol['x']

    def nlnprob(self, phi, g):
        """
        Evaluates the negative log probability of g given phi
        """
        z = self.A.dot(g.ravel())
        return self.cost(g.ravel(), z, f_args=(phi.ravel(),))

    def deconvolve(self, phi, x0=None, **kwargs):
        """
        Perform a deconvolution of data phi with provided kernel.

        input:
            phi : ndarray of shape self.kernel.N, data to be analyzed
            x0  : ndarray of shape self.n, initial guess for x (or g)
                default is random vector roughly normalized
        returns:
            x (or g)    : ndarray of shape self.n, solution of deconvolution

        kwargs:
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
        x0 = x0 if x0 is not None else np.random.randn(self.n)/np.sqrt(self.n)

        f_args = (phi.ravel(),)
        f_kwargs = {}
        f_kwargs['maxiter'] = kwargs.get('maxiter', 200)
        f_kwargs['tol'] = kwargs.get('tol', 1E-6)
        f_kwargs['solver'] = kwargs.get('solver', 'minres')

        g_args = ()
        g_kwargs = {}
        g_kwargs['zsteps'] = kwargs.get('zsteps', 20)

        algorithm = kwargs.get("algorithm", "minimize")
        minimizer = getattr(self, algorithm)

        xmin, _, msg = minimizer(x0, f_args, g_args, f_kwargs, g_kwargs,
                                 **kwargs)
        return xmin


class TVFiniteSupportDeconvolve(ADMM):
    """
    Deconvolve with total variation and finite support prior.

    This class solves the optimization problem

    x* = argmin_x I_c(x) + g(x)

    where I_c(x) is an indicator function for set c, i.e.
    I_c(x) = 0 if x is in c and infinity otherwise, and
    g(x) = 1/2||Mx-phi||^2 + TV(x), the standard TV deconvolution.

    usage:
        deconvolver = TVFiniteSupportDeconvolve(kernel, gamma, g_mask,
                                                g_ext)
        deconvolver.deconvolve(phi, x0, **kwargs)
    """

    def __init__(self, kernel, gamma, g_mask, g_ext=None, **kwargs):
        n = kernel.N_pad
        m = n
        p = m
        super(TVFiniteSupportDeconvolve, self).__init__(n, m, p, **kwargs)

        self.kernel = kernel
        self.gamma = gamma
        self.set_mask(g_mask)

        self._zminsol = None  # Warm start
        self.A = MyLinearOperator((self.m, self.m), matvec=lambda x:  x)
        self.B = MyLinearOperator((self.m, self.m), matvec=lambda x: -x)
        self.c = np.zeros(self.p)

        # proximal operator for g(z) solves
        # argmin_z 1/2||Mx-phi||^2 + TV(x) + nu/2||x-x^hat||^2
        self.prox_g = TVDeconvolver(kernel, gamma, g_ext, **kwargs)

    def set_mask(self, mask):
        """
        Set mask which defines regions of the model which have some constant
        value. Contiguous regions of value 1 will be forced to be constant,
        separated regions of 1 will be allowed to have different constants.
        input:
            mask : ndarray of shape (self.kernel._padshape)
        """
        assert mask.shape == self.kernel._padshape, "mask is incorrect shape"
        self.g_mask = mask
        indices = np.arange(mask.size).reshape(*mask.shape)
        labels, num = label(mask)
        self.region_indices = [indices[labels == i] for i in range(1, num+1)]

    def project_onto_c(self, x):
        """
        Projects x onto the set c, which is the set of images which have a
        constant value in the regions defined by self.g_mask.

        input:
            x   : ndarray of shape (self.n)
        output:
            Px  : ndarray same shape as x, projected onto set c
        """
        for ind in self.region_indices:
            x[ind] = x[ind].mean()
        return x

    def f(self, x, *args, **kwargs):
        """
        Value of indicator function, but since it will almost
        always be infinity, I set this to return zero so that
        the minimizer will print meaningful numbers.

        input:
            x   : ndarray of shape self.n, x-value
        """
        return 0.

    def g(self, z, *args, **kwargs):
        """
        Fidelity and TV prior function, the thing we really want to optimize
        input:
            z   : ndarray of shape self.n
            args:
                tuple, first element should be data phi of shape self.kernel.N
        returns:
            cost: float, cost function to be minimized
        """
        Dz = self.prox_g.A.dot(z)
        dx, dy = Dz[:self.n], Dz[self.n:]
        tv = nu.evaluate('sum(sqrt(dx*dx + dy*dy))')
        phi = args[0]
        res = self.prox_g.M.dot(z) - phi
        return res.dot(res)/2 + self.gamma * tv

    def x_update(self, z, y, rho, x0=None, *args, **kwargs):
        """
        Solves argmin_x I_c(x) + y^T x + rho/2||x-z||^2
                (complete the square)
            =  argmin_x I_c(x) + rho/2||x-(z-y/rho)||^2
            =  P_c(z-y/rho)
                (Project onto constants set c)
        input:
            z   : ndarray of shape self.m, current z-value
            y   : ndarray of shape self.p, current y-value
            rho : float, strength of augmentation term
            (optional)
            x0  : ndarray of shape self.n, initial guess of x
        """
        self._xminsol = self.project_onto_c(z - y/rho)
        return self._xminsol

    def z_update(self, x, y, rho, z0=None, *args, **kwargs):
        """
        Solves argmin_z -y^T z + g(z) where g(z) is our TV deconvolution
        function. Simply uses TVDeconvolver for this optimization.

        input:
            x   : ndarray of shape self.n, current x-value
            y   : ndarray of shape self.p, current y-value
            rho : float, strength of augmentation term
            (optional)
            z0  : ndarray of shape self.m, initial guess of z
            args:
                tuple, first element must be data phi, of shape
                self.kernel.N
            kwargs:
                z_rho   : float, initial strength of augmentation term
                            for z_update deconvolution

        """
        phi = args[0]
        self.prox_g.nu = rho
        self.prox_g.xhat = x + y/rho
        if self._zminsol is None:
            z0 = np.random.randn(self.n)/np.sqrt(self.n)
        else:  # warm start
            z0 = self._zminsol.copy()
        z_rho = kwargs.get('z_rho', 1E-2)
        self._zminsol = self.prox_g.deconvolve(phi, z0, rho=z_rho, **kwargs)
        return self._zminsol

    def callstart(self, x0=None, **kwargs):
        self._zminsol = None  # Reset warm start

    def start_lagrange_mult(self, x0, z0, rho, *f_args, **f_kwargs):
        """
        Initializes lagrange multiplier for stability. This statement can be
        found by writing the x_update x = P(z-y/rho), multiplying both sides
        by P, solving for P(y).
        input:
            x0  : ndarray of shape self.n, initial x-value
            z0  : ndarray of shape self.m, initial z-value
            rho : float, strength of augmentation term
        returns:
            y0  : ndarray of shape self.p, initial y-value
        """
        return rho * self.project_onto_c(z0 - x0)

    def deconvolve(self, phi, x0=None, options={}, **kwargs):
        """
        Perform a deconvolution of data phi with provided kernel.

        input:
            phi     : ndarray of shape self.kernel.N, data to be analyzed
            x0      : ndarray of shape self.n, initial guess for x (or g)
                default is roughly normalized random vector
            options : dictionary which is passed as kwargs to self.z_update
                        for setting kwargs of TVDeconvolver.deconvolve
        returns:
            x (or g)    : ndarray of shape self.n, solution of deconvolution
        kwargs:
            algorithm   : str, either 'minimize' (default) for standard ADMM
                            or 'minimize_fastrestart' for fast ADMM
            gcolor      : str, color for printing inner ADMM loop for z_update.
                            default is 'purple', can choose from
                            ADMM.printcolors
            (All other kwargs are passed to the minimizer, either ADMM.minimize
             or ADMM.minimize_fastrestart. See for documentation)

        """
        x0 = x0 if x0 is not None else np.random.randn(self.n)/np.sqrt(self.n)
        x0 = self.project_onto_c(x0)

        f_args, f_kwargs = (), {}

        g_args = (phi,)
        g_kwargs = {}
        g_kwargs["algorithm"] = options.get("algorithm", "minimize")
        g_kwargs['maxiter'] = options.get('maxiter', 20)
        g_kwargs['tol'] = options.get('tol', 1E-6)
        g_kwargs['solver'] = options.get('solver', 'minres')
        g_kwargs['zsteps'] = options.get('zsteps', 20)
        g_kwargs['z_rho'] = options.get("rho", 1E-2)
        g_kwargs['iprint'] = options.get('iprint', 1)
        g_kwargs['eps_abs'] = options.get('eps_abs', 1E-5)
        g_kwargs['eps_rel'] = options.get('eps_rel', 1E-5)

        algorithm = kwargs.get("algorithm", "minimize")
        minimizer = getattr(self, algorithm)
        self.prox_g.defaultprintcolor = kwargs.get("gcolor", "purple")
        self.prox_g.appendprint = "\t"  # indent inner admm messages

        xmin, zmin, msg = minimizer(x0, f_args, g_args, f_kwargs, g_kwargs,
                                    **kwargs)
        return xmin

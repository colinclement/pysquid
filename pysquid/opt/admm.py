"""
admm.py

author: Colin Clement
date: 2016-12-06

This defines an abstract class for implementing
the Alternating Difference Method of Multipliers
based on Stephen Boyd's 2010 paper

    http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf

Also implemented is an accelerated version of ADMM as described
in Fast Alternating Direction Optimization Methods, Tom Goldstein et al.,
SIAM J. Imaging Sciences, Vol.7, No. 3, pp 1588-1623 (2014)

This class must be subclassed to be useful. A proper subclass
must define the methods ADMM.f, ADMM.g, ADMM.x_update, ADMM.z_update,
and it must define the attributes A, B, and c, defined below.
"""

import numpy as np
import scipy.sparse.linalg as ssl


def defaultcallback(x, z, y, **kwargs):
    pass


class ADMM(object):
    """
    Solves the optimization problem

    x* = argmin_x f(x) + g(z)
        subject to Ax + Bz = c

    where x is in R^n, y is in R^m, and C is in R^p.
    Matrices A and B are shaped appropriately and f and
    g are convex.

    The algorithm is as follows:
        define the augmented lagrangian
        L(x, z, y, p) := f(x) + g(z) + y.dot(Ax+Bz-c)
                         + rho/2 ||Ax+bz-c||^2
        repeat until convergence criterion met:
            x_{k+1} := argmin_x L(x_{k}, z_{k}, y_{k}, p)
            z_{k+1} := argmin_z L(x_{k+1}, z, y_{k}, p)
            y_{k+1} := y_{k} + rho(Ax_{k+1} + Bz_{k+1} - c)

    The notation and most of the algorithm are implementations
    of the paper "Distributed Optimization and Statistical
    Learning via the Alternating Direction Method of Multipliers"
    by Stephen Boyd et al. Foundations and Trends in Machine
    Learning Vol. 3, No. 1 (2010)
    http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
    """

    def __init__(self, n, m, p, **kwargs):
        """
        n : (int) size of x
        m : (int) size of z
        p : (int) size of c

        kwargs:
            rho : (float) augmented regularization
        """
        self.n = n
        self.m = m
        self.p = p

        self.A = None
        self.B = None
        self.c = None

        self.msg = {1: "Convergence criterion satisfied", 2: "Iteration limit reached"}
        self.printcolors = {
            "red": "\033[1;31;49m",
            "green": "\033[1;32;49m",
            "blue": "\033[1;34;49m",
            "purple": "\033[1;35;49m",
            "gray": "\033[1;30;49m",
            "black": "",
        }
        self.defaultprintcolor = "black"
        self.appendprint = ""

    def cprint(self, string, color=None):
        """
        Print in red text
        """
        color = color if color is not None else self.defaultprintcolor
        printstr = self.appendprint + self.printcolors[color] + string + "\033[0m"
        print(printstr)

    # -----------------------------------------------------------------
    # These functions must be subclassed for ADMM to function
    # -----------------------------------------------------------------

    def f(self, x, *args, **kwargs):
        """
        One of the functions to be minimized.
        input:
            x: numpy array of shape n
        """
        pass

    def g(self, z, *args, **kwargs):
        """
        Second functions to be minimized.
        input:
            z: numpy array of shape m
        """
        pass

    def x_update(self, z, y, rho, x0=None, *args, **kwargs):
        """
        Perform the optimization
            x_{k+1} := argmin_x L(x_{k}, z_{k}, y_{k}, p)
        """
        pass

    def z_update(self, x, y, rho, z0=None, *args, **kwargs):
        """
        Perform the optimization
            z_{k+1} := argmin_z L(x_{k+1}, z, y_{k}, p)
        """
        pass

    def start_lagrange_mult(self, x0, z0, rho, *f_args, **f_kwargs):
        """
        Set the initial lagrange multiplier so that if the algorithm
        starts at the true x it stays there.
        NOTE: This accepts f_args and f_kwargs as it is assumed that the
        x-update (minimizing over f) is performed first.
        """
        pass

    # ----------------------------------------------------------------
    # End functions necesarry for ADMM to function
    # -----------------------------------------------------------------

    def callstart(self, x0=None, *args, **kwargs):
        """
        Call at the beginning of minize for custom child classes
        """
        pass

    @staticmethod
    def check_callback(callback):
        cbackstr = "callback function must have at least 3 args (x, z, y)"
        assert callback.__code__.co_argcount > 2, cbackstr
        kstr = "callback should accept kwargs"
        assert "kwargs" in callback.__code__.co_varnames, kstr

    def y_update(self, y, x, z, rho):
        """
        Update lagrange multiplier
        """
        A, B, c = self.A, self.B, self.c
        return y + rho * (A.dot(x) + B.dot(z) - c)

    def cost(self, x, z, f_args=(), g_args=()):
        """
        Compute objective function
        """
        return self.f(x, *f_args) + self.g(z, *g_args)

    def check_shape(self, arr, name, shape):
        errstr = "{} should have shape {}".format(name, shape)
        assert arr.shape == shape, errstr

    def primal_residual(self, x, z):
        """
        Compute the primal residual, the violation
        of the condition Ax+Bz=c
        """
        A, B, c = self.A, self.B, self.c
        return A.dot(x) + B.dot(z) - c

    def dual_residual(self, z1, z0, rho):
        """
        Compute the dual residual, another quantity
        guaranteed to converge in ADMM
        """
        A, B = self.A, self.B
        return rho * A.T.dot(B.dot(z1 - z0))

    def stop(self, r, s, x, z, y, eps_rel, eps_abs, iprint=0):
        """
        Compute the stopping criterion of ADMM as discussed
        in section 3.3.1 of Boyd 2010.
        """
        A, B, c = self.A, self.B, self.c
        Ax = A.dot(x)
        Bz = B.dot(z)
        ATy = A.T.dot(y)
        eps_pri = np.sqrt(self.p) * eps_abs + eps_rel * max(
            Ax.dot(Ax), Bz.dot(Bz), c.dot(c)
        )
        eps_dual = np.sqrt(self.n) * eps_abs + eps_rel * ATy.dot(ATy)
        if r.dot(r) <= eps_pri and s.dot(s) <= eps_dual:
            return True
        else:
            return False

    def update_rho(self, rho, t_inc, t_dec, mu, r, s):
        """
        See equation 3.13, this update tries to keep
        the dual and primal residuals within a factor of
        mu of each other
        """
        r2, s2 = r.dot(r), s.dot(s)
        if r2 > mu * s2:
            return rho * float(t_inc)
        elif s2 > mu * r2:
            return rho / float(t_dec)
        else:
            return rho

    def start_z(self, x0, **kwargs):
        """
        Set the initial z0 of ADMM to satisfy the constrain
        Ax + Bz = c by solving Bz = c - Ax
        """
        self._z0rhs = self.c - self.A.dot(x0)
        maxiter = kwargs.get("z0_maxiter", None)
        atol = kwargs.get("z0_atol", 1e-6)
        btol = kwargs.get("z0_btol", 1e-6)
        self._z0minsol = ssl.lsqr(
            self.B, self._z0rhs, iter_lim=maxiter, atol=atol, btol=btol, damp=1e-5
        )
        return self._z0minsol[0]

    def minimize(
        self, x0=None, f_args=(), g_args=(), f_kwargs={}, g_kwargs={}, **kwargs
    ):
        """
        Implementation of the ADMM algorithm as described
        in Boyd 2010. Includes the scheme for updating
        rho as described in section 3.4.1
        input:
            x0: ndarray of length n, initial x. Defaults
                to random array
        returns:
            optimal x0, message integer

        kwargs:

        z0 : ndarray of length m, initial z. Defaults to
            random array
        y0 : ndarray of length p, initial y. Defaults to
            random array
        rho : float, initial weighting of augmented term

        Parameters effecting convergence of algorithm

        eps_abs : float, tolerance for absolute size of residuals
        eps_rel : float, tol for relative size of residuals

        See equation 3.13 in Boyd 2010 for definition of these.
        t_inc : Amount to increase rho by when primal residual is large
        t_dec : Amount to decrease rho by when dual residual is large
        mu : Algorithm tries to keep ||dual||^2 = mu * ||primal||^2

        z0_solver : string from scipy.sparse.linalg for linear solver
            to initialize z to satisfy constraints given x0.
        """
        self.callstart(x0, **kwargs)
        callback = kwargs.get("callback", defaultcallback)
        self.check_callback(callback)

        itnlim = kwargs.get("itnlim", 40)
        eps_abs = kwargs.get("eps_abs", 1e-4)
        eps_rel = kwargs.get("eps_rel", 1e-6)
        t_inc = kwargs.get("t_inc", 2.0)
        t_dec = kwargs.get("t_dec", 2.0)
        mu = kwargs.get("mu", 10)
        rho = kwargs.get("rho", 1e-2)

        x0 = x0 if x0 is not None else np.random.randn(self.n)
        z0 = self.start_z(x0, **kwargs)
        y0 = self.start_lagrange_mult(x0, z0, rho, *f_args, **f_kwargs)
        iprint = kwargs.get("iprint", 1)

        self.check_shape(x0, "x0", (self.n,))
        self.check_shape(z0, "z0", (self.m,))
        self.check_shape(y0, "y0", (self.p,))

        # Check types
        itnmsg = "itnlim must be positive int"
        assert isinstance(itnlim, int) and itnlim > 0, itnmsg
        assert t_inc > 0, "t_inc must > 0"
        assert t_dec > 0, "t_dec must be 0"
        assert mu > 0, "mu must be 0"
        assert rho > 0, "rho must be 0"

        cost = self.cost(x0, z0, f_args, g_args)

        if iprint:
            self.cprint("Initial cost = {:.3e}".format(cost))

        for itn in range(itnlim):
            x1 = self.x_update(z0, y0, rho, x0, *f_args, **f_kwargs)
            z1 = self.z_update(x1, y0, rho, z0, *g_args, **g_kwargs)
            y1 = self.y_update(y0, x1, z1, rho)

            r1 = self.primal_residual(x1, z1)
            s = self.dual_residual(z1, z0, rho)
            rho = self.update_rho(rho, t_inc, t_dec, mu, r1, s)

            x0, z0, y0 = x1.copy(), z1.copy(), y1.copy()

            callback(x1, z1, y1, **kwargs)

            cost = self.cost(x1, z1, f_args, g_args)
            if iprint > 1:
                pstr = "Itn {:1}: cost = {:.3e}, rho = {}"
                self.cprint(pstr.format(itn, cost, rho))
                pstr = "\tr = {:.3e}, s = {:.3e} "
                self.cprint(pstr.format(r1.dot(r1), s.dot(s)))

            if self.stop(r1, s, x1, z1, y1, eps_rel, eps_abs, iprint):
                msg = 1
                break

        if itn == itnlim - 1:
            msg = 2

        if iprint:
            self.cprint(self.msg[msg])
            self.cprint("Final cost = {:.3e}".format(cost))

        return x1, z1, msg

    def combined_residual(self, y, y_hat, z, z_hat, rho):
        prim = y - y_hat
        dual = self.B.dot(z - z_hat)
        return prim.dot(prim) / rho + rho * dual.dot(dual)

    def minimize_fastrestart(
        self, x0=None, f_args=(), g_args=(), f_kwargs={}, g_kwargs={}, **kwargs
    ):
        """
        Implementation of the ADMM algorithm as described
        in Boyd 2010. Includes the scheme for updating
        rho as described in section 3.4.1. Includes Nesterov-type
        acceleration and a restarting scheme as described in
        Goldstein 2014.
        input:
            x0: ndarray of length n, initial x. Defaults
                to random array
        returns:
            optimal x0, message integer

        kwargs:

        z0 : ndarray of length m, initial z. Defaults to
            random array
        y0 : ndarray of length p, initial y. Defaults to
            random array

        Parameters effecting convergence of algorithm
        rho : float, initial weighting of augmented term
        eta : float [0,1) for setting how often restarts occur.
            Closer to 1 leads to fewer restarts (generally preferred)

        eps_abs : float, tolerance for absolute size of residuals
        eps_rel : float, tol for relative size of residuals

        See equation 3.13 in Boyd 2010 for definition of these.
        t_inc : Amount to increase rho by when primal residual is large
        t_dec : Amount to decrease rho by when dual residual is large
        mu : Algorithm tries to keep ||dual||^2 = mu * ||primal||^2

        z0_solver : string from scipy.sparse.linalg for linear solver
            to initialize z to satisfy constraints given x0.
        """
        self.callstart(x0, **kwargs)
        callback = kwargs.get("callback", defaultcallback)
        self.check_callback(callback)

        itnlim = kwargs.get("itnlim", 20)
        eps_abs = kwargs.get("eps_abs", 1e-4)
        eps_rel = kwargs.get("eps_rel", 1e-6)
        t_inc = kwargs.get("t_inc", 2.0)
        t_dec = kwargs.get("t_dec", 2.0)
        mu = kwargs.get("mu", 10)
        rho = kwargs.get("rho", 1e-1)
        eta = kwargs.get("eta", 0.999)

        x0 = x0 if x0 is not None else np.random.randn(self.n)/np.sqrt(self.n)
        z0 = self.start_z(x0, **kwargs)
        y0 = self.start_lagrange_mult(x0, z0, rho, *f_args, **f_kwargs)
        iprint = kwargs.get("iprint", 1)

        self.check_shape(x0, "x0", (self.n,))
        self.check_shape(z0, "z0", (self.m,))
        self.check_shape(y0, "y0", (self.p,))

        # Check types
        itnmsg = "itnlim must be positive int"
        assert isinstance(itnlim, int) and itnlim > 0, itnmsg
        assert t_inc > 0, "t_inc must > 0"
        assert t_dec > 0, "t_dec must be 0"
        assert mu > 0, "mu must be 0"
        assert rho > 0, "rho must be 0"
        assert 0 <= eta < 1, "eta must be in [0,1)"

        cost = self.cost(x0, z0, f_args, g_args)

        z_hat, y_hat = z0.copy(), y0.copy()
        alpha0, c0 = 1.0, np.inf  # always accept first accelerated step

        if iprint:
            self.cprint("Initial cost = {:.3e}".format(cost))

        for itn in range(itnlim):
            x1 = self.x_update(z_hat, y_hat, rho, x0, *f_args, **f_kwargs)
            z1 = self.z_update(x1, y_hat, rho, z_hat, *g_args, **g_kwargs)
            y1 = self.y_update(y_hat, x1, z1, rho)
            c1 = self.combined_residual(y1, y_hat, z1, z_hat, rho)

            r1 = self.primal_residual(x1, z1)
            s = self.dual_residual(z1, z_hat, rho)
            rho = self.update_rho(rho, t_inc, t_dec, mu, r1, s)

            if c1 < eta * c0:
                alpha1 = (1 + np.sqrt(1 + 4 * alpha0 ** 2)) / 2
                vel = (alpha0 - 1) / alpha1
                z_hat = z1 + vel * (z1 - z0)
                y_hat = y1 + vel * (y1 - y0)
            else:  # Restart acceleration
                alpha1, z_hat, y_hat = 1.0, z0.copy(), y0.copy()
                c1 = c0 / eta
                if iprint > 1:
                    self.cprint("\t\tRestarted acceleration", color="red")

            x0, z0, y0 = x1.copy(), z1.copy(), y1.copy()
            alpha0, c0 = alpha1, c1

            callback(x1, z1, y1, **kwargs)

            cost = self.cost(x1, z1, f_args, g_args)
            if iprint > 1:
                pstr = "Itn {:1}: cost = {:.3e}, rho = {}"
                self.cprint(pstr.format(itn, cost, rho))
                pstr = "\tr = {:.3e}, s = {:.3e} "
                self.cprint(pstr.format(r1.dot(r1), s.dot(s)))

            if self.stop(r1, s, x1, z1, y1, eps_rel, eps_abs, iprint):
                msg = 1
                break

        if itn == itnlim - 1:
            msg = 2

        if iprint:
            self.cprint(self.msg[msg])
            self.cprint("Final cost = {:.3e}".format(cost))

        return x1, z1, msg

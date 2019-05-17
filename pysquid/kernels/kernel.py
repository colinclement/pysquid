"""
kernel.py

    author: Colin Clement
    email: colin.clement@gmail.com
    date: 2018-06-13

This is a class for managing the construction of the M matrix for computing
magnetic flux due to g-fields. It includes the ability to subtract the fields
due to the edges of the currents. This 

"""


from __future__ import division, print_function
from copy import copy
import numexpr as nu
import numpy as np

from pysquid.component import ModelComponent
from pysquid.util.fftw import FFT


class Kernel(ModelComponent):
    def __init__(self, shape, params=None, padding=None, edges=True, **kwargs):
        """
        Kernel class instance stores point spread functions and the flux of
        point (square) sources of constant g-field. 

        input:
            shape: tuple of ints (ly, lx),  shape of measured flux field
            params: list of parameters for point spread function. 
                     First element is always height above plane.
                     Length is dependent on subclass implementations
            padding: tuple of ints (py, px) for padding on g-field outside
                      flux image
            edges: True/False. False causes the kernel to subtracts fields due
                to image edge currents.

        kwargs:
            rxydx and dy: floats for units of x and y. Choose either dx or dy
            to be 1, and then the other should be the appropriate ratio.
            NOTE: You should choose the smaller of x and y units to be 1,
            so that the larger is a number greater than 1, which will promote
            numerical stability in the linear solvers.
            
            fftw_plan and threads (see FFT object)

            cutoff: Default is true, will cutoff PSF.
        """
        super(Kernel, self).__init__(shape, padding, **kwargs)
        self.edges = edges

        self._fftw_plan = kwargs.get("fftw_plan", "FFTW_MEASURE")
        self._fftw_threads = kwargs.get("threads", 4)

        self._updategrids()

        self.fft = FFT(self._fftshape, plan=self._fftw_plan, threads=self._fftw_threads)
        self._unPickleable += ["fft"]

        self._doubleg = np.zeros(self._fftshape)
        self._doublefluxpad = np.zeros(self._fftshape)

        self.updateParams("params", params if params is not None else [1.0])

    def __setstate__(self, d):
        self.__dict__ = d
        self.fft = FFT(self._fftshape, plan=self._fftw_plan, threads=self._fftw_threads)

    def _updategrids(self):
        self.x = self.rxy * np.arange(0.5, 2 * self.Lx_pad, 1.0)
        self.y = np.arange(-0.5, -2 * self.Ly_pad, -1.0)
        self.xg, self.yg = np.meshgrid(self.x, self.y)
        self.d_xg = np.fft.fftshift(self.xg - (self.rxy * 2 * self.Lx_pad) / 2)
        self.d_yg = np.fft.fftshift(self.yg + (2 * self.Ly_pad) / 2)

    def updateParams(self, name, values):
        """
        params = list of at least one number
        set up for lazy evaluation
        """
        if name == "params":
            self.params = copy(values)
            self._updatem()
            self._updatepsf()
            if not self.edges:
                self._updateE()

    def _updatem(self):
        self.mg = _gGreensFunction(
            self.rxy / 2.0,
            -1 / 2.0,
            self.d_xg,
            self.d_yg,
            self.params[0],
            self.rxy,
            1.0,
        )
        self.mg_k = self.fft.fft2(self.mg)

    def _updateE(self):
        height, rxy = self.params[0], self.rxy
        Ly, Lx = self.Ly_pad, self.Lx_pad
        x = np.fft.fftshift(rxy * np.arange(-Lx, Lx)[None, :])
        y = np.fft.fftshift(np.arange(-Ly, Ly)[:, None])
        self.edgefields = [
            _bzlinecurrent(x, y, 0.0, -0.5, height, 1.0, False),
            -_bzlinecurrent(x, y, 0.0, 0.5, height, 1.0, False),
            _bzlinecurrent(x, y, -rxy / 2.0, 0.0, height, rxy),
            -_bzlinecurrent(x, y, rxy / 2.0, 0.0, height, rxy),
        ]
        self.edgefields_k = [self.fft.fft2(b) for b in self.edgefields]

    def _updatepsf(self):
        """
        """
        self.psf = np.zeros(self._fftshape)
        self.psf[0, 0] = 1.0
        self.psf_k = self.fft.fft2(self.psf)

    def edgeslice(self, pad=True):
        """ top bottom left right """
        if pad:
            Ly, Lx = self.Ly_pad, self.Lx_pad
        else:
            Ly, Lx = self.Ly, self.Lx
        return [
            (0, slice(None, Lx, None)),
            (Ly - 1, slice(None, Lx, None)),
            (slice(None, Ly, None), 0),
            (slice(None, Ly, None), Lx - 1),
        ]

    def edgeproject(self, g, pad=True):
        output = []
        for s in self.edgeslice(pad):
            edge = np.zeros_like(g)
            edge[s] = g[s]
            output.append(edge)
        return output

    def applyM(self, arr):
        """
        Compute product M.dot(g) using FFT.
        input:
            arr : (g) (Ly_pad, Lx_pad)-shaped or (N_pad)-shaped float array
        output:
            M.dot(arr) : (Ly_pad, Lx_pad)-shaped array
        """
        Ly, Lx = self.Ly_pad, self.Lx_pad
        g = arr.reshape(Ly, Lx)
        self._doubleg[:Ly, :Lx] = g[:, :]
        g_k = self.fft.fft2(self._doubleg)
        out = self.crop(self.fft.ifft2(self.psf_k * self.mg_k * g_k))
        if not self.edges:
            for ek, lg in zip(self.edgefields_k, self.edgeproject(self._doubleg)):
                lg_k = self.fft.fft2(lg)
                out -= self.crop(self.fft.ifft2(self.psf_k * ek * lg_k))
        return out

    def applyMt(self, arr):
        """
        Compute product M.T.dot(arr) using FFT.
        input:
            arr : (flux) (Ly, Lx)-shaped or (Ly*Lx)-shaped float array
        output:
            M.T.dot(arr) : (Ly_pad,Lx_pad)-shaped array where, e.g.
        """
        Ly, Lx, py, px = self.Ly, self.Lx, self.py, self.px
        flux = arr.reshape(Ly, Lx)
        self._doublefluxpad[py : py + Ly, px : px + Lx] = flux[:, :]
        flux_k = self.fft.fft2(self._doublefluxpad)
        out = self.fft.ifft2(self.psf_k * self.mg_k * flux_k)
        if not self.edges:
            for ek, sl in zip(self.edgefields_k, self.edgeslice()):
                out[sl] -= self.fft.ifft2(self.psf_k * ek * flux_k)[sl]
        return out[: self.Ly_pad, : self.Lx_pad]

    @property
    def m_k(self):
        return self.psf_k * self.mg_k


def _indefIntegral(x, y, d):
    """
    Indefinite integral of g-field flux for thin sheet of current 
    """
    d2, x2, y2 = d * d, x * x, y * y
    return nu.evaluate("x*y*(2*d2+x2+y2)/((d2+x2)*(d2+y2)*sqrt(d2+x2+y2))")


def _gGreensFunction(x0, y0, x, y, z, ax, ay):
    """
    Flux field from a square of constant g (square loop of
    constant current), and derivative of said flux with respect to 
    the sample plane height z

    input:
        x0, y0 : (floats) coordinates of the center of the current
        x, y: float arrays of equal shape giving the points at which
                the flux is to be evaluated.
        z : (float) height of measurement plane above currents
        ax, ay : (float) width of current loop

    output:
        phi : array of shape (N, M) where x.shape=y.shape=(N,M),
        dphi_dz : same shape and type as phi
        
    """
    ax2, ay2 = ax / 2.0, ay / 2.0
    A = lambda x1, y1: _indefIntegral(x1, y1, z)
    xc, yc = x - x0, y - y0
    Bz = (
        A(ax2 - xc, ay2 - yc)
        - A(ax2 - xc, -ay2 - yc)
        - A(-ax2 - xc, ay2 - yc)
        + A(-ax2 - xc, -ay2 - yc)
    ) / (4 * np.pi)
    return Bz


def _lineindefintegral(l, x, y, z):
    b, r = x * x + z * z, l - y
    return x * r / (b * np.sqrt(b + r * r)) / (4 * np.pi)


def _bzlinecurrent(x, y, x0, y0, z, l, ydir=True):
    """
    Computes the z-component of the magnetic field due to a line of unit current
    of length l, centered at (x0, y0). If ydir is True, the current points in
    the y-direction. otherwise ir points in the x-direction. Sampled at points
    x, y
    """
    if ydir:
        return _lineindefintegral(l / 2.0, x - x0, y - y0, z) - _lineindefintegral(
            -l / 2.0, x - x0, y - y0, z
        )
    else:
        return _lineindefintegral(l / 2.0, y - y0, x - x0, z) - _lineindefintegral(
            -l / 2.0, y - y0, x - x0, z
        )

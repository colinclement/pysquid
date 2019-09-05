"""
magpsf.py

    author: Colin Clement
    email: colin.clement@gmail.com
    date: 2015-9-10

This contains classes which contain kernels with the Biot-Savart law
information and a point-spread function (PSF).

"""

import numpy as np

from pysquid.kernels.kernel import Kernel


class GaussianKernel(Kernel):
    def __init__(self, shape, params=None, padding=None, **kwargs):
        """
        Single Gaussian PSF kernel. params are x and y-standard deviations
        input:
            shape : tuple of ints (ly, lx),  shape of measured flux field
            params : 3-element numpy array.  list of parameters for 
                    point spread function. 
                    First element is always height above plane. Second and third
                    elements are sigma_x and sigma_y of Gaussian
            padding : tuple of ints (py, px) for padding on g-field outside
                      flux image

        kwargs:
            rxy: ratio of x pixel width to y pixel width (assumed to be 1)
            mirror: True/False, impose mirror boundary conditions
            _fftw_flags and _fftw_threads (see FFTW object)
        """
        if not len(params) == 3:
            raise RuntimeError("len(params) must be 3")
        self.params = params if params is not None else np.array([1.0, 1.0, 1.0])
        super().__init__(shape, self.params, padding, **kwargs)

    def _updatepsf(self):
        z, sx, sy = self.params
        # Shifted to center at 0,0 (to match fourier method)
        x0, y0 = self.d_xg[0, 0], self.d_yg[0, 0]
        d_xg, d_yg = self.d_xg - x0, self.d_yg - y0
        self.psf = np.exp(-((d_xg / sx) ** 2 + (d_yg / sy) ** 2) / 2)
        self.psf /= self.psf.sum() * self.rxy
        self.psf_k = self.fft.fft2(self.psf)


class MogKernel(Kernel):
    def __init__(self, shape, params, padding=None, **kwargs):
        """
        Mixture of Gaussians PSF kernel.
        input:
            shape : tuple of ints (ly, lx),  shape of measured flux field
            params : list of parameters for 
                    point spread function. 
                    First element is always height above plane. What follows
                    is 5n parameters for n gaussians. the gaussian
                    parameters are (amplitude, mean_x, mean_y, sigma_x, sigma_y)
            padding : tuple of ints (py, px) for padding on g-field outside
                      flux image

        kwargs:
            dx and dy: floats for units of x and y. Choose either dx or dy
            to be 1, and then the other should be the appropriate ratio.
            NOTE: You should choose the smaller of x and y units to be 1,
            so that the larger is a number greater than 1, which will promote
            numerical stability in the linear solvers.
            
            _fftw_flags and _fftw_threads (see FFTW object)

        """
        self.params = params.ravel()
        if (self.params.size - 1) % 5:
            raise RuntimeError("params must be of size 5n+1")
        super().__init__(shape, self.params, padding, **kwargs)
        self.N_g = (self.params.size - 1) // 5

    def _updatepsf(self):
        # Log params for a, sx, sy
        p = self.params[1:].reshape(self.N_g, 5)
        x, y, sx, sy = p[:, 1], p[:, 2], p[:, 3], p[:, 4]
        xg, yg = self.d_xg - self.d_xg[0, 0], self.d_yg - self.d_yg[0, 0]
        Dx, Dy = xg[:, :, None] - x[None, None, :], yg[:, :, None] - y[None, None, :]
        g = np.exp(
            -Dx ** 2 / (2 * sx[None, None, :] ** 2)
            - Dy ** 2 / (2 * sy[None, None, :] ** 2)
        )
        g = p[:, 0][None, None, :] * g / np.sqrt(2 * np.pi * sx * sy[None, None, :])
        self.psf = g.sum(2)
        self.psf /= self.psf.sum() * self.rxy
        self.psf_k = self.fft.fft2(self.psf)


from scipy.special import expit, logit


class PlatonicKernel(Kernel):
    def __init__(self, shape, params=None, padding=None, **kwargs):
        """
        This is an implementation of a platonic approximation
        to the PSF of a key-shaped squid-loop. It is a solid,
        smoothed circle and rectangle with reduced amplitude.
        input:
            shape: tuple (Ly, Lx) of data shape
            params: ndarray of log-parameters
                        [height, radius, length, width, keyamp, amp]

            NOTE: this model PSF transforms parameters to enforce
            a number of conditions. radius and length are made positive
            by using logarithmic parameters, i.e. the model
            takes their exponential. width and keyamp are restricted
            to be in the range (0,1) by taking the expit. width
            is then understood as fraction of the diameter (2*radius).
            amp is enforced to be positive. Use the methods
            self.get_p to transform to the constraint-satisfying
            parameters, and self.get_logp to transform to the
            unconstrained parameters.

            **kwargs:
                d: transition rate for sigmoidal smoothing of 
                    the platonic shapes
                h: step-size for central difference derivatives
                    of PSF with respect to parameters
        """
        assert len(self.params) == 6, "len(params) must be 6"
        default = np.log(np.array([1.0, 5.0, 5.0, 0.7, 0.33, 1.0]))
        self.d = kwargs.get("d", 0.25)
        self.h = kwargs.get("h", 1e-7)
        self.params = params if params is not None else default
        super().__init__(shape, self.params, padding, **kwargs)

    def _updatepsf(self):
        p0 = self.params.copy()
        self.psf = self._psf(p0)
        self.PSF_k = self.fft.fft2(self.psf)

        for ip in range(len(p0)):
            p0[ip] += self.h / 2.0
            plus_h = self._psf(p0)
            p0[ip] -= self.h
            minus_h = self._psf(p0)
            self.d_PSF[:, :, ip] = (plus_h - minus_h) / self.h
            p0[ip] += self.h / 2.0
            self.d_PSF_k[:, :, ip] = self.fft.fft2(self.d_PSF[:, :, ip])

    @staticmethod
    def get_logp(params):
        """
        transform params back to unconstrained domains
        """
        logp = np.log(params)
        logp[0] = params[0]  # let height be negative
        p2, p3 = params[2:4]
        logp[2], logp[3] = logit(params[2:4])
        return logp

    @staticmethod
    def get_p(params):
        """
        transform params into the constrained domains
        """
        p = np.exp(params)
        p[0] = params[0]  # let height be negative
        p[2:4] = expit(params[2:4])
        return p

    @property
    def p(self):
        """
        Get the constrained parameters
        """
        return self.get_p(self.params)

    def _circle(self, r):
        """
        Make image of smoothed circle with radius r
        """
        x0, y0 = self.d_xg[0, 0], self.d_yg[0, 0]
        d_xg, d_yg = self.d_xg - x0, self.d_yg - y0
        radial = np.hypot(d_xg, d_yg)
        return expit(-(radial - r) / self.d)

    def _rectangle(self, L, w, offset):
        """
        Make smoothed rectangle of height (length) L,
        width w, offset from the origin by offset.
        """
        x0, y0 = self.d_xg[0, 0], self.d_yg[0, 0] - offset
        d_xg, d_yg = self.d_xg - x0, self.d_yg - y0
        sigx = expit(-(np.abs(d_xg) - w / 2.0) / self.d)
        sigy = expit(-(np.abs(d_yg + L / 2) - L / 2) / self.d)
        return sigx * sigy

    def _psf(self, params):
        """
        Make the platonic keyhole-shaped psf with
        unconstrained parameters params
        """
        _, radius, length, width, kamp, amp = self.get_p(params)
        circ = self._circle(radius)
        offset = radius * np.sqrt(1 - width ** 2)
        rect = self._rectangle(length, width * 2 * radius, offset)
        out = np.maximum(circ, kamp * rect)
        return out * amp

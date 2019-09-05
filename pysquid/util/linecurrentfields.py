"""
linecurrentfields.py

author: Colin Clement
date: 2017-07-11

This module computes the z-component of the magnetic field 
of lines of current of finite length. It uses these fields
to calculate the contribution to the field from the edges of
a gfield (for which current is conserved), so that this contribution
can be subtracted away to reveal the magnetic field of a gfield
which does not have current conservation on the edges.
"""


import numpy as np


def _indefintegral(l, x, y, z):
    b, r = x * x + z * z, l - y
    return x * r / (b * np.sqrt(b + r * r)) / (4 * np.pi)


def bzlinecurrent(x, y, x0, y0, z, l, ydir=True):
    """
    Computes the z-component of the magnetic field due to a
    line of unit current length l, centered at (x0, y0). If
    ydir is True, the current points in the y-direction,
    otherwise it points in the x-direction. Sampled at points
    x, y
    """
    if ydir:
        return _indefintegral(l / 2.0, x - x0, y - y0, z) - _indefintegral(
            -l / 2.0, x - x0, y - y0, z
        )
    else:
        return _indefintegral(l / 2.0, y - y0, x - x0, z) - _indefintegral(
            -l / 2.0, y - y0, x - x0, z
        )


def bzfield_edgecurrents(gfield, height, rxy=1.0):
    Ly, Lx = gfield.shape
    x = np.fft.fftshift(rxy * np.arange(-Lx, Lx)[None, :])
    y = np.fft.fftshift(np.arange(-Ly, Ly)[:, None])
    top = bzlinecurrent(x, y, 0.0, -0.5, height, 1.0, False)
    bottom = -bzlinecurrent(x, y, 0.0, 0.5, height, 1.0, False)
    left = bzlinecurrent(x, y, -rxy / 2.0, 0.0, height, rxy)
    right = -bzlinecurrent(x, y, rxy / 2.0, 0.0, height, rxy)

    edge = np.zeros((2 * Ly, 2 * Lx))
    kernels = [top, bottom, left, right]
    slices = [np.s_[0, :Lx], np.s_[Ly - 1, :Lx], np.s_[:Ly, 0], np.s_[:Ly, Lx - 1]]
    edgefield_k = np.zeros((2 * Ly, Lx + 1), dtype="complex128")
    for k, sl in zip(kernels, slices):
        edge[sl] = gfield[sl].copy()
        edgefield_k += np.fft.rfftn(edge) * np.fft.rfftn(k)
        edge[sl] = 0.0
    return np.fft.irfftn(edgefield_k)[:Ly, :Lx]

"""
component.py

author: Colin Clement
date: 2015-10-18

This class contains boilerplate for defining array sizes and dealing with
padded g-fields in most modules of pysquid.

"""

import numpy as np


class ModelComponent(object):
    def __init__(self, shape, padding=None, **kwargs):
        """
        shape : (ly, lx) integer tuple of flux image shape
        padding : (py, px) integer tuple of padding for g-field
        """
        self._updateSizes(shape, padding, **kwargs)

    def _updateSizes(self, shape, padding=None, **kwargs):
        """
        shape : (ly, lx) integer tuple of flux image shape
        padding : (py, px) integer tuple of padding for g-field
        """
        self._shape = shape
        self.Ly, self.Lx = shape
        self.N = self.Lx*self.Ly
        self.rxy = kwargs.get('rxy', 1.)

        self._padding = padding if padding is not None else [0, 0]
        self.py, self.px = self._padding
        self.Ly_pad, self.Lx_pad = self.Ly+2*self.py, self.Lx+2*self.px
        self._padshape = (self.Ly_pad, self.Lx_pad)
        self.N_pad = self.Ly_pad * self.Lx_pad
        self._fftshape = (2*self.Ly_pad, 2*self.Lx_pad)
        self._unPickleable = []

    def __getstate__(self):
        d = self.__dict__.copy()
        for unpk in self._unPickleable:
            try:
                del d[unpk]
            except KeyError:
                pass
        return d

    def __setstate__(self, d):
        self.__dict__ = d

    def crop(self, arr):
        """
        input:
            arr : 2D numpy array

        return:
            arr cropped to remove padding
        """
        Ly, Lx, py, px = self.Ly, self.Lx, self.py, self.px
        if len(arr.shape) < 2:
            try:
                arr = arr.reshape(self.Ly_pad, self.Lx_pad)
            except ValueError as perror:
                return arr
            return arr[py:py+Ly, px:px+Lx].ravel()
        else:
            return arr[py:py+Ly, px:px+Lx]

    def updateParams(self, name, values):
        """
        Implemented for each component.
        name is a string naming the parameter,
        values is a numpy array
        """
        pass

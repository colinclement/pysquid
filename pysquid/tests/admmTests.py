import numpy as np
from pysquid.linearModel import *
from pysquid.util.helpers import makeM

class LinearModelTV_ADMM_NoFFT(LinearModelTV_ADMM):
    def __init__(self, shape, kernel, padding = None, **kwargs):
        super(LinearModelTV_ADMM_NoFFT, self).__init__(shape, kernel, padding,
                                                       **kwargs)


    def _makeLinearOperators(self):
        self.M = makeM(np.fft.fftshift(self.kernel.MPSF.real), self._padding)
        self.A = MyLinearOperator((self.N_pad, self.N_pad),
                                  matvec = self._g_kernel_apply)
        self._unPickleable = ['A']

    def _g_kernel_apply(self, g):
        M, D = self.M, self.D
        return (M.T.dot(M.dot(g.ravel())) + 
                self.rho * self.D.T.dot(self.D.dot(g.ravel())))
    


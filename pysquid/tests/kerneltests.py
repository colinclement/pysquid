import numpy as np
from pysquid.kernels.magpsf import *
from pysquid.kernels.psf import *
from pysquid.util.helpers import makeM

def testConvolution(kernelType, psf_params = None, shape = (10, 10), 
                    padding = (2, 2), gtest = None, **kwargs):
    """
    This function tests the validity of using a Fourier Transform with a cutoff
    kernel to perform the convolution of the Biot-Savart law and PSF.

    """
    kernel = kernelType(shape, psf_params, padding, cutoff = False, **kwargs)
    MPSF = np.fft.fftshift(kernel.MPSF.copy()).real
    M = makeM(MPSF, padding)
    gtest = gtest if gtest is not None else np.random.randn(*kernel._padshape)
    fftConv = kernel.applyM(gtest).real
    trueConv = M.dot(gtest.ravel()).reshape(*shape)
    rmserror = np.sqrt(np.mean((fftConv - trueConv)**2))
    return rmserror, kernel, M

def testTranspose(kernelType, psf_params = None, shape = (10,10),
                  padding = (2,2), gtest = None, phitest = None, **kwargs):
    kernel = kernelType(shape, psf_params, padding, cutoff = False, **kwargs)
    MPSF = np.fft.fftshift(kernel.MPSF.copy()).real
    M = makeM(MPSF, padding)
    gtest = gtest if gtest is not None else np.random.randn(kernel.N_pad)
    phitest = phitest if phitest is not None else np.random.randn(kernel.N)
    fftConv = kernel.applyM(gtest).real.ravel()
    fftMT = kernel.applyMt(phitest).real.ravel()
    trueConv = M.dot(gtest.ravel())
    trueMT = M.T.dot(phitest.ravel())
    rmserror = np.sqrt(np.mean((fftConv - trueConv)**2))
    rmserrorMt = np.sqrt(np.mean((fftMT - trueMT)**2))
    return rmserror, rmserrorMt

def testGradient(kernelType, psf_params = None, shape = (10, 10),
                 padding = (2,2), gtest = None, stepsize = 1E-7, **kwargs):
    kernel = kernelType(shape, psf_params, padding, cutoff = False, **kwargs)
    gtest = gtest if gtest is not None else np.random.randn(*kernel._padshape)
    kernel_grad = kernel.computeGradients(gtest)
    psf_params = kernel.psf_params.copy()
    check_grad = np.zeros_like(kernel_grad)
    ev = kernel.applyM(gtest).real
    for i in range(len(psf_params)):
        psf_params[i] += stepsize
        kernel.updateParams('psf_params', psf_params)
        check_grad[:,:,i] = (kernel.applyM(gtest).real - ev)/stepsize
        psf_params[i] -= stepsize
    rmsdiff = np.sqrt(np.mean((kernel_grad-check_grad)**2))
    print("RMS deviation of gradient is {}".format(rmsdiff))
    return rmsdiff, kernel_grad, check_grad

"""
fftw.py

author: Colin Clement
date: 2017-06-07

This module is a wrapper for pyfftw which makes use of aligned
memory and keeps track of wisdom so that continued use of highler
optimized ffts is only expensive once
"""

import pickle
import os
try:
    import pyfftw
    hasfftw = True
except ImportError as ierr:
    print("Install pyfftw for 20x speedup")
    hasfftw = False


class FFT(object):
    def __init__(self, shape, real=True, **kwargs):
        """
        A convenience wrapper which uses aligned memory to improve performance
        of FFTs
        input:
            (required)
            shape : tuple (Ly, Lx) shape of arrays to work with
        kwargs:
            threads : int, the number of threads to parallelize across
            planning : str, the setting for how hard FFTW should work to
                optimize performace. Default is "FFTW_PATIENT".
                Options are, in order of increasing time spent at the
                initialization: "FFTW_ESTIMATE", "FFTW_MEASURE", 
                "FFTW_PATIENT", and "FFTW_EXHAUSTIVE"

        usage:
            fft = FFT(shape)
            fft.fft2(a)  # a.shape == shape
        """
        self.real = real
        self.shape = shape
        self.n = shape[0]*shape[1]
        self.realshape = (shape[0], shape[1]//2+1)
        self.fftwkwargs = {"threads": kwargs.get("threads", 2),
                           "flags": [kwargs.get("plan", "FFTW_MEASURE")], 
                           "axes": (0,1)}
        self.alignargs = {"dtype": "complex128"}
        self.alignargs_forward = {"dtype": "float64" if real else "complex128"}
        self.wisdomfile = kwargs.get("wisdom", os.path.join(os.path.expanduser("~"), 
                                                            ".register_wisdom.pkl"))
        self._loadwisdom(self.wisdomfile)

        self.a = pyfftw.empty_aligned(shape, **self.alignargs_forward)
        self.b = pyfftw.empty_aligned(self.realshape if real else shape, 
                                      **self.alignargs)
        self.fftobj = pyfftw.FFTW(self.a, self.b, **self.fftwkwargs)
        self.ifftobj = pyfftw.FFTW(self.b, self.a, direction="FFTW_BACKWARD", 
                                   **self.fftwkwargs)
        self._savewisdom(self.wisdomfile)
    
    def _loadwisdom(self, infile):
        if infile is None:
            return
        try:
            pyfftw.import_wisdom(pickle.load(open(infile, 'rb')))
        except (IOError, TypeError) as e:
            self._savewisdom(infile)

    def _savewisdom(self, outfile):
        if outfile is None:
            return
        if outfile:
            pickle.dump(pyfftw.export_wisdom(), open(outfile, 'wb'),
                        protocol=0)

    def fft2(self, a):
        """ Perform 2D FFT on array a """
        self.a[:,:] = a
        return self.fftobj().copy()

    def ifft2(self, b):
        """ Perform 2D IFFT on array b """
        self.b[:,:] = b
        return self.ifftobj().copy()


class FFTNumpy(object):
   def __init__(self, shape, **kwargs):
       self.shape = shape

   def fft(self, inp):
       return np.fft.fftn(inp)

   def ifft(self, inp):
       return np.fft.ifftn(inp) 


if not hasfftw:
    FFT = FFTNumpy

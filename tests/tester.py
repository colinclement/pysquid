"""
tester.py

author: Colin Clement
date: 2019-06-18

This module provides a class for controlling numerical experiments
"""

import numpy as np
from pysquid.deconvolve import LinearDeconvolver, TVDeconvolver

RNG = np.random.RandomState(4810740)
DECON_DICT = {'LinearDeconvolver': LinearDeconvolver,
              'TVDeconvolver': TVDeconvolver}


class Tester:
    """
    Tester(g, kernel, sigma, phi_ext=None, g_ext=None)

    class for comparing multiple deconvolution methods and
    parameter combinations for a given experimental setup.
    Parameters
    ----------
    g : array_like
        ground truth g-field
    kernel : pysquid.kernel.Kernel
        kernel object for computing magnetic flux
    sigma : float
        magnitude of Gaussian noise to add to flux
    phi_ext : array_like, optional
        Flux in field of view due to external current model. Default is None.
    g_ext : array_like, optional
        g-field in field of view due to external current model. Default is None.
    """
    def __init__(self, g, kernel, sigma, phi_ext=None, g_ext=None):
        self.g = g
        self.kernel = kernel
        self.sigma = sigma
        self.phi_ext = phi_ext
        self.g_ext = g_ext

        self.phi = self.make_data(g, kernel, sigma)

    def make_data(self, g, kernel, sigma):
        """ Returns kernel.dot(g) + noise """
        flux = kernel.applyM(g.ravel()).ravel().real + sigma * RNG.randn(g.size)
        return flux - self.phi_ext if self.phi_ext is not None else flux

    def test_protocols(self, protocols):
        """
        Run test protocols
        Parameters
        ----------
        protocols : list of dicts
            List of dicts defining numerical experiments, e.g.

            # the first three keys 'label', 'decon', and 'sigma' are required
            [{'label': 'uniform annulus gaussian prior',
              'decon': 'LinearDeconvolver',
              'sigma': 0.07,
              # the following are optional
              'support_mask': None,
              'deconv_kwargs': {}  # kwargs for deconv.deconvolve
              }, ...
            ]

        Returns
        -------
        results : dict
            results[label] = g_field_solution
        """
        output = {}
        kwargs = {'g_ext': self.g_ext}
        for pro in protocols:
            results = {}
            print('Testing protocol {}'.format(pro['label']))
            kwargs['support_mask'] = pro.get('support_mask')
            kwargs['gamma'] = pro.get('gamma')
            decon = DECON_DICT[pro['decon']](self.kernel, pro['sigma'], **kwargs)
            gsol = decon.deconvolve(self.phi, **pro.get('deconv_kwargs', {}))

            results['residual'] = (self.kernel.applyM(gsol).ravel().real -
                                   self.phi).reshape(self.kernel._padshape)

            if self.g_ext is not None:
                gsol += self.g_ext.ravel()
            results['gsol'] = gsol.reshape(self.kernel._padshape)

            output[pro['label']] = results
        print('Finished all protocols')
        return output

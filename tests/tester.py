"""
tester.py

author: Colin Clement
date: 2019-06-18

This module provides a class for controlling numerical experiments
"""

import numpy as np
from pysquid.deconvolve import LinearDeconvolver, TVDeconvolver

RNG = np.random.RandomState(4810740)
DECON_DICT = dict('LinearDeconvolver'=LinearDeconvolver,
                  'TVDeconvolver'=TVDeconvolver)


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
        if self.phi_ext is not None:
        return kernel.applyM(g.ravel()) + sigma * rng.randn(g.size)

    def test_protocols(self, protocols):
        """
        Run test protocols
        Parameters
        ----------
        protocols : list of dicts
            List of dicts defining numerical experiments, each of which requires
            keys 'label' (string labeling experiment), 'decon' (string name of
            deconvolution method), and 'sigma' (the estimated noise magnitude or
            regularization strength). Each dict can optionally contain keys
            'support_mask' (the finite support mask, defaults None) and
            'deconv_kwargs' (the keyword arguments handed to deconv.deconvolve).

        Returns
        -------
        results : dict
            results[label] = g_field_solution
        """
        results = {}
        kwargs = {'g_ext': self.g_ext}
        for pro in protocols:
            print('Testing protocol {}'.format(pro['label']))
            kwargs['support_mask'] = pro.get('support_mask')
            decon = DECON_DICT[pro['decon']](self.kernel, pro['sigma'], **kwargs)
            gsol = decon.deconvolve(self.phi, **pro.get('deconv_kwargs', {}))
            results[pro['label']] = (gsol + self.g_ext).reshape(self.kernel._padshape)
        print('Finished all protocols')
        return results

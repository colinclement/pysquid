"""
annular_currents.py

author: Colin Clement
date: 2019-06-04

This script produces g-fields corresponding to annular current distributions
with uniform and parabolic profiles, for the purpose of showcasing the effects
of different priors in pysquid.
"""

import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

from pysquid.util.linear_operators import makeD

def uniform_profile(x, y, ir, w):
    r = np.hypot(x, y)
    return - np.clip((r - ir), 0, ir + w)  + ir + w  # counter-clockwise

def parabolic_profile(x, y, ir, w):
    r = np.hypot(x, y)
    mask = uniform_profile(x, y, ir, w)
    mask = (mask - (ir + w)/2)**2 - (ir + w) ** 2 / 4
    jx = mask * y / r
    return - np.cumsum(jx, 0)

profiles = {'uniform': uniform_profile, 'parabolic': parabolic_profile}

def annular_gfield(L, inner_rad, width, profile='uniform'):
    """
    Produce jx, jy current distribution which is an annulus
    with inner_rad and width as a fraction of image width L
    """
    x = y = np.arange(L, dtype='float64')
    xx, yy = np.meshgrid(x, y)
    xx -= xx.mean()
    yy -= yy.mean()
    return profiles[profile](xx, yy, inner_rad, width)

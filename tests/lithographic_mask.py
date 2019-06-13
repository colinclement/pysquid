"""
lithographic_mask.py

author: Colin Clement
date: 2019-06-05

"""

import os
import numpy as np
import matplotlib.pyplot as plt

from pysquid.rnet import ResistorNetworkModel

DIR = os.path.dirname(os.path.abspath(__file__))

MASK_FILE = os.path.join(DIR, 'data', 'litho-mask.png')

mask = 1 * (plt.imread(MASK_FILE)[:,:,3] > 0)[:1000]

if __name__ == '__main__':
    # slice up the mask so it doesn't take forever!
    model_slice = np.s_[400:, 151:1500]
    print('Solving currents for mask model')
    netmodel = ResistorNetworkModel(mask[model_slice])
    
    # select field of view
    window = np.s_[430:520, 610:740]
    g = netmodel.gfield[window]

    print('Saving currents')
    np.savez(os.path.join(DIR, 'data', 'g-field-litho.npz'), 
             g=g, model_slice=model_slice, window=window)


import numpy as np
import os
import matplotlib.pyplot as plt
from pysquid.util.helpers import curl

params = {'text.usetex': True, 
          'font.family': 'sans-serif', 'font.sans-serif': 'cmbright'}
plt.rcParams.update(params)

def crop(im, px, py):
    Ly, Lx = im.shape
    return im[py:py+Ly-2*py,
              px:px+Lx-2*px]

loc = os.path.dirname(os.path.realpath(__file__))

data = np.load(os.path.join(loc,
                            "../../../data/2016-03-09-HgTe-reconstruction.npz"))['data']
dy, dx = 0.16, 0.73 #Pixel distance in micrometers

fluxmax = max([np.abs(d[1]['scan']).max() for d in data])

maxcurrent = lambda jxjy: max(np.abs(jxjy[0]).max(), np.abs(jxjy[1]).max())
jmax = max([maxcurrent(curl(d[1]['g_sol'])) for d in data])

def plotfluxcurrents(flux, g_sol, dx, dy, voltage, jlim = jmax, flim = fluxmax):
    cmap = 'RdBu_r'
    fig, axes = plt.subplots(1,3, figsize = (12, 5))
    
    jx, jy = curl(g_sol, dx, dy)
    axes[0].matshow(flux, aspect=dy/dx, vmin=-flim, vmax=flim, cmap=cmap)
    axes[0].set_title("Measured Flux", fontsize=30)
    
    axes[1].matshow(crop(jx, 10, 40), aspect=dy/dx, vmin=-jlim, vmax=jlim, cmap=cmap)
    axes[1].set_title("$j_x$", fontsize=30)
    
    axes[2].matshow(crop(jy, 10, 40), aspect=dy/dx, vmin=-jlim, vmax=jlim, cmap=cmap)
    axes[2].set_title("$j_y$", fontsize=30)
    
    plt.suptitle("Gate voltage = {}".format(voltage), fontsize=30)
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()


if __name__ == '__main__':
    plot_loc = os.path.join(loc, "HgTe/plots/voltage_gif")

    for fn, da in enumerate(data):
        plotfluxcurrents(da[1]['scan'], da[1]['g_sol'], dx, dy, da[0])
        plt.savefig(os.path.join(plot_loc, "voltage_{}.png".format(fn)),
                    dpi=150)
        plt.close()

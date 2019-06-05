"""


NOTE: commit ca51b66d contains a MeasuredKernel which does the cleaning
and interpolation of the measured PSF. Repeat that work with the fitted PSF.

"""

import numpy as np
from scipy.io import loadmat
from scipy.optimize import leastsq


class Mog(object):
    def __init__(self, shape, N_g, params=None, rxy=None, **kwargs):
        self.shape = shape
        self.Ly, self.Lx = shape
        self.N_g = N_g
        self.rxy = rxy or 1.

        x = np.arange(self.Lx)
        y = np.arange(0, -self.Ly, -1)
        self.xg, self.yg = np.meshgrid(x, y)
        self.xg = self.dx*(self.xg-self.Lx/2.)
        self.yg = (self.yg+self.Ly/2.)
        self.mixture = np.zeros(self.shape)
        self.d_params = np.zeros((self.xg.size, self.N_g*5))
        self.params = params
        if self.params is None:
            #Initialize A, x, y, sx, sy, for each gaussian
            a = np.sqrt(self.rxy),
            self.params = np.c_[np.random.rand(N_g),
                                3*self.rxy*np.random.randn(N_g),
                                3*np.random.randn(N_g),
                                3*self.rxy + np.random.randn(N_g),
                                3 + np.random.randn(N_g)]
            self.params[:,0] /= self.params[:,0].sum()
        self.logparams = self.getlogp(self.params)
        self.update(self.logparams)
    
    def gaussian(self, params):
        #Log params for a, sx, sy
        x, y, sx, sy = params[:,1], params[:,2], np.exp(params[:,3]), np.exp(params[:,4])
        Dx, Dy = self.xg[:,:,None]-x[None,None,:], self.yg[:,:,None]-y[None,None,:]
        g = np.exp(- Dx**2/(2*sx[None,None,:]**2) - Dy**2/(2*sy[None,None,:]**2))
        g = np.exp(params[:,0])[None,None,:]*g/(2*np.pi*sx*sy[None,None,:])
        return g.sum(2).ravel()
 
    def d_gaussian(self, params):
        #Log params for a, sx, sy
        a = np.exp(params[:,0])
        x, y, sx, sy = params[:,1], params[:,2], np.exp(params[:,3]), np.exp(params[:,4])
        sx2, sy2 = sx*sx, sy*sy
        Dx, Dy = self.xg[:,:,None]-x[None,None,:], self.yg[:,:,None]-y[None,None,:]
        Dx2, Dy2 = Dx*Dx, Dy*Dy
        g = np.exp(- Dx2/(2*sx2[None,None,:]) - Dy2/(2*sy2[None,None,:]))
        g = g*a/(2*np.pi*sx*sy)[None,None,:]
        da = g #a = exp(alpha) 
        dx, dy = g*Dx/sx[None,None,:]**2, g*Dy/sy[None,None,:]**2
        dsx = g*(Dx-sx[None,None,:])*(Dx+sx[None,None,:])/(sx2[None,None,:])
        dsy = g*(Dy-sy[None,None,:])*(Dy+sy[None,None,:])/(sy2[None,None,:])
        #dsy = g*(2*Dy2-sy2[None,None,:])/(2*sy2[None,None,:])
        for i, d in enumerate([da, dx, dy, dsx, dsy]):
            self.d_params[:,i::5] = d.reshape(-1, self.N_g)
        return self.d_params

    def getp(self, logp):
        a = np.exp(logp[:,0])
        x, y = logp[:,1], logp[:,2]
        sx, sy =  np.exp(logp[:,3]), np.exp(logp[:,4])
        return np.vstack([a, x, y, sx, sy]).T

    def getlogp(self, p):
        a = np.log(p[:,0])
        x, y = p[:,1], p[:,2]
        sx, sy =  np.log(p[:,3]), np.log(p[:,4])
        return np.vstack([a, x, y, sx, sy]).T

    def update(self, params):
        self.mixture = self.gaussian(params.reshape(self.N_g, -1))

    def residual(self, params, data):
        self.update(params)
        return self.mixture - data.ravel()

    def jacobian(self, params, data):
        return self.d_gaussian(params.reshape(self.N_g, -1))

    def fit(self, data, p0 = None, maxfev = 200):
        p0 = p0 if p0 is not None else self.params.ravel()
        self.opt = leastsq(self.residual, self.logparams.ravel().copy(),
                           Dfun = self.jacobian, args = (data,),
                           full_output = 1, maxfev = maxfev)
        self.logparams = self.opt[0].reshape(self.N_g, -1)
        self.params = self.getp(self.logparams)
        return self.params


def changeUnits(params, dx = 1., dy = 1.):
    params = params.reshape(-1, 5)
    x, y = params[:,1], params[:,2]
    sx, sy = params[:,3], params[:,4]
    return np.vstack([params[:,0], x/dx, y/dy,
                      sx/dx, sy/dy]).T

def center(params):
    params = params.reshape(-1, 5)
    a, x, y = params[:,0], params[:,1], params[:,2]
    maxa = np.argmax(a)
    maxx, maxy = x[maxa], y[maxa]
    sx, sy = params[:,3], params[:,4]
    return np.vstack([a, x - maxx, y - maxy, sx, sy]).T


if __name__ == '__main__':
    # y, x pixel distances for the flux data and PSF 
    spacing = {'img': np.array([.16, .73]), 'psf': np.array([.48, .37])}
    
    dy, dx = spacing['psf']
    minx = spacing['img'].min()

    filename = '../../../katja/scan604_kernel.mat'
    psfdata = loadmat(filename)
    rawpsf = psfdata['kernel_structure'][0][0][0][::-1].real
    psf = rawpsf/rawpsf.sum()

    mog = Mog(psf.shape, 2, dx = dx, dy = dy)
    params = mog.fit(psf)

    s_params = center(params)
    np.savez('psf_params_{}_gaussians.npz'.format(mog.N_g), psf_params = s_params)



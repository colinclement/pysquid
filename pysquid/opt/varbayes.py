import numpy as np
from pysquid.opt.adam import Adam

class Varbayes(object):
    def __init__(self, kernel, sigma, gamma, dh, dv, 
                 **kwargs):
        self.kernel = kernel
        self.N = self.kernel.N_pad
        self.sigma = sigma
        self.gamma = gamma
        self.dh = dh
        self.dv = dv
        self.opt = Adam(self.Dkl_d_Dkl, 2*self.N, 
                        **kwargs)

    def fidelity(self, g, data):
        kernel = self.kernel
        Ly_pad, Lx_pad = kernel._padshape
        d_hat = kernel.applyM(g.reshape(Ly_pad,-1)).ravel()
        res = d_hat.real - data.ravel()
        return res.dot(res)/(2*self.sigma**2)
    
    def d_fidelity(self, g, data):
        kernel = self.kernel
        Ly, Lx = kernel._shape
        Ly_pad, Lx_pad = kernel._padshape
        d_hat = kernel.applyM(g.reshape(Ly_pad,-1)).ravel()
        res = (d_hat.real - data.ravel()).reshape(Ly,-1)
        return kernel.applyMt(res).ravel().real/self.sigma**2
        
    def TV(self, g, g_ext = None):
        g_ext = g_ext if g_ext is not None else 0.
        dhg, dvg = self.dh.dot(g+g_ext), self.dv.dot(g+g_ext)
        return np.sqrt(dhg**2+dvg**2)
    
    def d_TV(self, g, g_ext = None):
        g_ext = g_ext if g_ext is not None else 0.
        dhg, dvg = self.dh.dot(g+g_ext), self.dv.dot(g+g_ext)
        sqd2 = np.sqrt(dhg**2+dvg**2)
        return (self.dh.T.dot(dhg/sqd2) + self.dv.T.dot(dvg/sqd2))
    
    def nll(self, g, data, g_ext = None):
        return (self.fidelity(g, data)+self.gamma*self.TV(g, g_ext).sum()
                + self.N/2*np.log(2*np.pi*self.sigma**2))
    
    def Dkl_d_Dkl(self, lamb, data, batches, g_ext = None):
        N = len(lamb)/2
        mus, rhos = lamb[:N], lamb[N:]
        sigs = np.exp(rhos)
        eps = np.random.randn(batches, N)
        gs = mus[None,:] + eps * sigs[None,:]
        dkl = 0
        d_mus = np.zeros_like(mus)
        d_rhos = np.zeros_like(rhos)
        for g, e in zip(gs, eps):
            dkl += self.nll(g, data, g_ext)
            d_dg = self.d_fidelity(g, data)+self.gamma*self.d_TV(g, g_ext)
            d_mus += d_dg
            d_rhos += d_dg * e * sigs
        dkl /= batches
        d_mus /= batches
        d_rhos /= batches
        return dkl - rhos.sum(), np.concatenate([d_mus, d_rhos - 1])

    def fit(self, data, itn = 200, batches = 12, iprint=10,
            tol = 1E-3, **kwargs):
        datapad = np.zeros(self.N).reshape(*self.kernel._padshape)
        py, px = self.kernel._padding
        Ly, Lx = self.kernel._shape
        datapad[py:py+Ly, px:px+Lx] = data
        self.p = kwargs.get('p0', np.r_[datapad.ravel(), 
                                        np.log(np.random.rand(self.N))])
        lamb = self.opt.optimize(self.p, itn, args=(data, batches),
                                 iprint = iprint, tol = tol)  
        return lamb[:self.N], np.exp(lamb[self.N:])


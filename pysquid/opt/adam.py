"""
adam.py

author: Colin Clement
date: 2015-02-12

ADAM stochastic gradient descent optimizer.

usage:

opt = Adam(obj_grad_obj) 
sol = opt.optimize(np.random.randn(N))
"""

import numpy as np


class Adam(object):
    def __init__(self, obj_grad_obj, D, lr=0.001, 
                 beta1=0.9, beta2=0.999, eps=1e-8):
        """
        ADAM stochastic gradient descent optimizer.
        input:
            obj_grad_obj: Function which takes D (int) parameters
		and returns tuple (objective function value, its gradient)
            and returns (objective, grad_objective), can take other args
            D: int number of parameters
            lr: learning rate or step size
            beta1: exponential decay rate of first moment
            beta2: exponential decay rate of second moment
            eps: regularization to prevent divide-by-zero
        """
        self.obj_grad_obj = obj_grad_obj
        self.D = D
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._reset()
        
    def _reset(self):
        self.t = 0
        self.obj_list = []
        self.m = np.zeros(self.D)
        self.v = np.zeros(self.D)
        
    def update(self, params, args = ()):
        """
        Take one step of ADAM SGD.
        input:
            params: array of D floats
            args: tuple of extra arguments to obj_grad_obj
        """
        obj, g = self.obj_grad_obj(params, *args)
        self.t += 1
        self.m += (1-self.beta1)/(1-self.beta1**self.t) * (g - self.m)
        self.v += (1-self.beta2)/(1-self.beta2**self.t) * (g**2 - self.v)
        dparams = -self.lr * self.m / (np.sqrt(self.v) + self.eps)
        params += dparams
        return params, obj
    
    def optimize(self, p0, itn = 1000, tol = 1E-2,
                 iprint = 50, args = ()):
        """
        Run ADAM SGD.
        input:
            p0: array of D floats to start optimization
            itn : int number of iterations
            tol : change in objective below which algorithm terminates
            iprint : int for how often to print status of algorithm
            args : tuple of extra arguments to obj_grad_obj
        """
        self._reset()
        obj0, _ = self.obj_grad_obj(p0, *args)
        for i in range(itn):
            p0, obj = self.update(p0, args)
            try:
                if i % iprint == 0:
                    print("Itn {:6d}: obj = {:8e}".format(i, obj))
            except ZeroDivisionError as perror:
                pass
            if np.abs(obj - obj0) < tol:
                print("Change in objective less than tol")
                break
            self.obj_list += [obj]
            obj0 = obj
        return p0  

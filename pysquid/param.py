"""
params.py

author: Colin Clement
Date: 2015-10-7

ParameterMap is an object for managing parameters across modules
so that dependencies on those parameters can be updated and
accessing and optimizing subsets of parameters is easy.

"""

import numpy as np
from collections import defaultdict


class ParameterMap(object):
    def __init__(self):
        self.depend_dict = defaultdict(list)
        self.valuedict = defaultdict(list)
        self.N_params = 0
    
    def __getitem__(self, name):
        return self.valuedict[name]

    def register(self, name, depends, values):
        """
        Register parameters so updating across modules works.
        input:
            name : string, module attribute name
            depends : modules with depend on name, and which
                     have 'updateParams' method the know how to
                     deal with name
            values : numpy array containing values of name,
                    even if only one value
        """
        self.N_params += len(values)
        self.depend_dict[name] = depends
        self.valuedict[name] = values
        for depend in depends:
            depend.updateParams(name, values)
    
    def updateParams(self, names=None, values=None):
        names = names if names is not None else self.names
        values = values if values is not None else self.values
        for name, val in zip(names, values):
            self.valuedict[name] = val
            for depend in self.depend_dict[name]:
                depend.updateParams(name, val)

    @property
    def params(self):
        p = np.array([])
        for name in self.valuedict:
            p = np.r_[p, self.valuedict[name]]
        return p

    @property
    def names(self):
        return self.valuedict.keys()

    @property
    def values(self):
        return self.valuedict.values()

    @property
    def len_values(self):
        out = []
        for name in self.valuedict:
            out += [len(self.valuedict[name])]
        return out

